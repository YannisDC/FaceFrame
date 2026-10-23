import simd

/// How depth pixels map to the camera. ARKit does not document whether the TrueDepth depth map
/// is mirrored relative to the camera intrinsics, nor whether its calibration data is always
/// present, so the scanner tests every combination against the tracked face mesh at start-up and
/// keeps the one that agrees best.
struct DepthMapping: Hashable, CustomStringConvertible {
    var mirrored: Bool
    var useCalibrationData: Bool

    static let all: [DepthMapping] = [
        DepthMapping(mirrored: false, useCalibrationData: true),
        DepthMapping(mirrored: true, useCalibrationData: true),
        DepthMapping(mirrored: false, useCalibrationData: false),
        DepthMapping(mirrored: true, useCalibrationData: false),
    ]

    var description: String { "\(mirrored ? "mirrored" : "direct") / \(useCalibrationData ? "depth calibration" : "camera intrinsics")" }

    func image(_ f: FusionFrame) -> DepthImage {
        let w = f.width, h = f.height
        var fx, fy, cx, cy: Float
        if useCalibrationData, let k = f.calibrationK, let ref = f.calibrationReference, ref.x > 0 {
            let sx = Float(w) / ref.x, sy = Float(h) / ref.y
            fx = k.columns.0.x * sx; fy = k.columns.1.y * sy
            cx = k.columns.2.x * sx; cy = k.columns.2.y * sy
        } else {
            // Colour frames may be a 16:9 crop of the 4:3 sensor; depth covers the full 4:3 area.
            let s = Float(w) / f.imageResolution.x
            fx = f.intrinsics.columns.0.x * s; fy = f.intrinsics.columns.1.y * s
            cx = f.intrinsics.columns.2.x * s
            cy = f.intrinsics.columns.2.y * s + (Float(h) - f.imageResolution.y * s) / 2
        }
        var values = f.depth
        if mirrored {
            for y in 0..<h {
                let row = y * w
                for x in 0..<(w / 2) { values.swapAt(row + x, row + w - 1 - x) }
            }
        }
        return DepthImage(width: w, height: h, values: values, fx: fx, fy: fy, cx: cx, cy: cy)
    }

    /// Median absolute disagreement (metres) between the depth image and ARKit's face mesh.
    static func faceAgreement(_ img: DepthImage, cameraFromFace: simd_float4x4, faceVertices: [Vec3]) -> Float? {
        var errors: [Float] = []
        errors.reserveCapacity(faceVertices.count / 2)
        for i in stride(from: 0, to: faceVertices.count, by: 2) {
            let p = cameraFromFace.transformPoint(faceVertices[i])
            let z = -p.z
            if z < 0.1 { continue }
            let u = Int((img.fx * p.x / z + img.cx).rounded()), r = Int((img.fy * -p.y / z + img.cy).rounded())
            if u < 0 || r < 0 || u >= img.width || r >= img.height { continue }
            let d = img.values[r * img.width + u]
            if d <= 0 { continue }
            errors.append(abs(d - z))
        }
        return errors.count > 150 ? errors.median() : nil
    }
}
