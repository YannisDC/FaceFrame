import CoreGraphics
import Foundation
import ImageIO
import simd
import Vision

/// Measures the pupils from the colour images instead of ARKit's eye model (whose eye positions come
/// from a symmetric face template and inherited a ~6 mm sideways offset on a real scan).
///
/// For each near-frontal frame: Vision finds the pupils, each pupil pixel becomes a camera ray, the
/// ray is placed in the head frame with the depth-corrected pose and intersected with the scanned eye
/// surface. Pupils are averaged over frames and corrected from near (eyes converged on the phone)
/// to distance PD.
struct PupilEstimator {
    struct Sample {
        var image: CGImage
        /// Colour intrinsics in raw sensor pixels.
        var intrinsics: simd_float3x3
        /// Raw ARKit head-from-camera pose (camera: x right, y up, looking down -z).
        var arPose: simd_float4x4
    }

    struct Result {
        /// Distance (far) pupils in the head frame.
        var left: Vec3
        var right: Vec3
        /// Near (as captured, converged on the camera) pupils.
        var nearLeft: Vec3
        var nearRight: Vec3
        var frames: Int
        /// Share of the PD on the subject's left, measured in the images (pupil-to-nose-bridge distances).
        /// Same-image ratio: immune to colour/depth registration and pose errors.
        var leftShare: Float?
        /// Median disagreement with ARKit's projected eye positions (pixels), for diagnostics.
        var arkitPixelError: Float
    }

    /// Why the last estimate returned nil, for logs.
    static var lastFailure = ""

    /// Distance from the eye's centre of rotation to the pupil plane (metres).
    var rotationRadius: Float = 0.0105

    /// - Parameters:
    ///   - arkitEyes: ARKit eye centres in ARKit's (uncorrected) face frame, to validate the image mapping.
    ///   - correction: ARKit-to-TrueDepth correction found during the scan.
    ///   - surface: scanned head surface (head frame).
    func estimate(_ samples: [Sample], arkitEyes: (left: Vec3, right: Vec3), correction: PoseCorrection,
                  surface: TriMesh) -> Result? {
        guard !samples.isEmpty else { return nil }
        // Vision → raw pixel mapping: the images are in sensor orientation; try all 8 axis flips/swaps
        // and keep the one that best matches where ARKit projects its eyes.
        struct Det { var a: CGPoint; var b: CGPoint; var bridge: [CGPoint]; var sample: Sample; var arkit: (SIMD2<Float>, SIMD2<Float>) }
        // ARKit stores front-camera frames in sensor orientation (portrait faces lie on their side):
        // pick the Vision orientation that finds a face, then use it for every frame.
        let orientations: [CGImagePropertyOrientation] = [.right, .left, .up, .down]
        guard let orientation = orientations.first(where: { o in samples.prefix(5).contains { detectPupils($0.image, orientation: o) != nil } })
        else { Self.lastFailure = "no pupils found by Vision"; return nil }
        var dets: [Det] = []
        for s in samples {
            guard let (p, q, bridge) = detectPupils(s.image, orientation: orientation) else { continue }
            let camFromFace = s.arPose.inverse
            func project(_ e: Vec3) -> SIMD2<Float> {
                let c = camFromFace.transformPoint(e)
                return SIMD2(s.intrinsics.columns.0.x * c.x / -c.z + s.intrinsics.columns.2.x,
                             s.intrinsics.columns.1.y * -c.y / -c.z + s.intrinsics.columns.2.y)
            }
            dets.append(Det(a: p, b: q, bridge: bridge, sample: s, arkit: (project(arkitEyes.left), project(arkitEyes.right))))
        }
        guard dets.count >= 3 else { Self.lastFailure = "pupils in only \(dets.count) frames"; return nil }

        func toPixel(_ p: CGPoint, _ t: Int, _ s: Sample) -> SIMD2<Float> {
            let a = Float(p.x), b = Float(1 - p.y) // top-left origin in the oriented image
            let w = Float(s.image.width), h = Float(s.image.height)
            let uv: (Float, Float)
            switch t {
            case 0: uv = (a, b); case 1: uv = (1 - a, b); case 2: uv = (a, 1 - b); case 3: uv = (1 - a, 1 - b)
            case 4: uv = (b, a); case 5: uv = (1 - b, a); case 6: uv = (b, 1 - a); default: uv = (1 - b, 1 - a)
            }
            return SIMD2(uv.0 * w, uv.1 * h)
        }
        var bestT = 0
        var bestErr = Float.infinity
        for t in 0..<8 {
            let errs = dets.map { d -> Float in
                let pa = toPixel(d.a, t, d.sample), pb = toPixel(d.b, t, d.sample)
                let e1 = simd_distance(pa, d.arkit.0) + simd_distance(pb, d.arkit.1)
                let e2 = simd_distance(pa, d.arkit.1) + simd_distance(pb, d.arkit.0)
                return min(e1, e2) / 2
            }
            if let m = errs.median(), m < bestErr { bestErr = m; bestT = t }
        }
        // Mapping sanity: pupils should land within ~40 px (≈ 12 mm at 30 cm) of ARKit's eyes.
        guard bestErr < 40 else {
            let d = dets[0]
            Self.lastFailure = String(format: "image mapping disagrees with ARKit by %.0f px; first frame: pupils (%.3f, %.3f) (%.3f, %.3f), ARKit (%.0f, %.0f)",
                                      bestErr, d.a.x, d.a.y, d.b.x, d.b.y, d.arkit.0.x, d.arkit.0.y)
            return nil
        }

        let front = HeightMap(mesh: surface, uAxis: Vec3(1, 0, 0), vAxis: Vec3(0, 1, 0), depthAxis: Vec3(0, 0, 1),
                              uRange: -0.1...0.1, vRange: -0.1...0.1, resolution: 0.0005)
        var nearL: [Vec3] = [], nearR: [Vec3] = [], farL: [Vec3] = [], farR: [Vec3] = []
        var shares: [Float] = []
        for d in dets {
            let pose = correction.apply(d.sample.arPose)
            var pa = toPixel(d.a, bestT, d.sample), pb = toPixel(d.b, bestT, d.sample)
            // assign to ARKit's left/right eye by proximity
            if simd_distance(pa, d.arkit.0) + simd_distance(pb, d.arkit.1) > simd_distance(pa, d.arkit.1) + simd_distance(pb, d.arkit.0) {
                swap(&pa, &pb)
            }
            // Nose bridge position along the pupil line, in the same image.
            if !d.bridge.isEmpty {
                let bridgePx = d.bridge.map { toPixel($0, bestT, d.sample) }
                let mid = bridgePx.reduce(SIMD2<Float>.zero, +) / Float(bridgePx.count)
                let axis = pb - pa
                let t = simd_dot(mid - pa, axis) / simd_length_squared(axis) // 0 at left pupil, 1 at right
                if t > 0.2 && t < 0.8 { shares.append(t) }
            }
            guard let l = intersect(pixel: pa, sample: d.sample, pose: pose, front: front),
                  let r = intersect(pixel: pb, sample: d.sample, pose: pose, front: front) else { continue }
            let cam = pose.translation
            nearL.append(l); nearR.append(r)
            farL.append(straightAhead(l, camera: cam)); farR.append(straightAhead(r, camera: cam))
        }
        if ProcessInfo.processInfo.environment["FF_DEBUG_SHARES"] == "1" {
            let sorted = shares.sorted()
            print("bridge shares:", sorted.count, "min", sorted.first ?? 0, "p25", sorted.isEmpty ? 0 : sorted[sorted.count / 4], "p75", sorted.isEmpty ? 0 : sorted[3 * sorted.count / 4], "max", sorted.last ?? 0)
        }
        guard nearL.count >= 3 else { Self.lastFailure = "pupil rays missed the eyes in \(dets.count - nearL.count) of \(dets.count) frames"; return nil }
        func med(_ v: [Vec3]) -> Vec3 {
            Vec3(v.map { $0.x }.median()!, v.map { $0.y }.median()!, v.map { $0.z }.median()!)
        }
        return Result(left: med(farL), right: med(farR), nearLeft: med(nearL), nearRight: med(nearR),
                      frames: nearL.count, leftShare: shares.count >= 3 ? shares.median() : nil, arkitPixelError: bestErr)
    }

    /// Pupil position if the eye looked straight ahead (+z) instead of at the camera.
    private func straightAhead(_ p: Vec3, camera: Vec3) -> Vec3 {
        let toCam = camera - p
        let sx = toCam.x / simd_length(toCam), sy = toCam.y / simd_length(toCam)
        return p - Vec3(sx, sy, 0) * rotationRadius
    }

    /// First crossing of the pupil ray with the scanned surface near the eye (front height map).
    private func intersect(pixel: SIMD2<Float>, sample s: Sample, pose: simd_float4x4, front: HeightMap) -> Vec3? {
        let k = s.intrinsics
        let dirCam = simd_normalize(Vec3((pixel.x - k.columns.2.x) / k.columns.0.x, -(pixel.y - k.columns.2.y) / k.columns.1.y, -1))
        let o = pose.translation, d = pose.transformDirection(dirCam)
        var prev: Vec3? = nil
        var t: Float = 0.12
        while t < 0.8 {
            let p = o + d * t
            if let z = front.value(u: p.x, v: p.y), p.z <= z {
                guard let q = prev else { return p }
                // refine by bisection
                var lo = t - 0.0005, hi = t
                for _ in 0..<12 {
                    let mid = (lo + hi) / 2, pm = o + d * mid
                    if let zm = front.value(u: pm.x, v: pm.y), pm.z <= zm { hi = mid } else { lo = mid }
                }
                _ = q
                return o + d * hi
            }
            prev = p
            t += 0.0005
        }
        return nil
    }

    /// Vision pupil landmarks in normalized coordinates of the oriented image (bottom-left origin).
    /// The mapping back to sensor pixels is chosen in `estimate` (8 axis flips/swaps checked against ARKit).
    private func detectPupils(_ image: CGImage, orientation: CGImagePropertyOrientation) -> (CGPoint, CGPoint, [CGPoint])? {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
        try? handler.perform([request])
        guard let face = request.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }),
              let lp = face.landmarks?.leftPupil?.normalizedPoints.first,
              let rp = face.landmarks?.rightPupil?.normalizedPoints.first else { return nil }
        let bb = face.boundingBox
        func img(_ p: CGPoint) -> CGPoint { CGPoint(x: bb.minX + p.x * bb.width, y: bb.minY + p.y * bb.height) }
        let bridge = (face.landmarks?.noseCrest?.normalizedPoints ?? []).map(img)
        return (img(lp), img(rp), bridge)
    }
}
