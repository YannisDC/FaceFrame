import simd

/// Similarity transform (camera frame) that maps ARKit's face model onto the TrueDepth points:
/// depth ≈ s·R·q + t, where q is a face-mesh vertex in camera coordinates according to ARKit.
///
/// ARKit fits its face model from the colour image, where a larger face further away looks the same
/// as a smaller one closer by; on real scans its head pose was ~5 cm too deep relative to TrueDepth.
/// TrueDepth depth is metric, so the fused model uses a "corrected head frame" in which ARKit's
/// face model is scaled by `s` and every ARKit pose is corrected to agree with the depth.
struct PoseCorrection: Equatable {
    var scale: Float = 1
    var rotation = matrix_identity_float3x3
    var translation = Vec3.zero

    static let identity = PoseCorrection()

    /// Corrected head-from-camera pose for an ARKit head-from-camera pose.
    func apply(_ arPose: simd_float4x4) -> simd_float4x4 {
        let camFromFace = arPose.inverse
        let rc = simd_float3x3(columns: (Vec3(camFromFace.columns.0.x, camFromFace.columns.0.y, camFromFace.columns.0.z),
                                         Vec3(camFromFace.columns.1.x, camFromFace.columns.1.y, camFromFace.columns.1.z),
                                         Vec3(camFromFace.columns.2.x, camFromFace.columns.2.y, camFromFace.columns.2.z)))
        let tc = camFromFace.translation
        let r = rotation * rc
        let t = scale * (rotation * tc) + translation
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4(r.columns.0, 0)
        m.columns.1 = SIMD4(r.columns.1, 0)
        m.columns.2 = SIMD4(r.columns.2, 0)
        m.columns.3 = SIMD4(t, 1)
        return m.inverse
    }

    /// Converts a point of ARKit's face frame (eyes, face mesh) into the corrected head frame.
    func headPoint(_ p: Vec3) -> Vec3 { p * scale }

    struct Sample {
        var image: DepthImage
        var arPose: simd_float4x4
        var vertices: [Vec3]
    }

    struct Fit {
        var correction: PoseCorrection
        var rmsBefore: Float
        var rmsAfter: Float
        var pairs: Int
    }

    /// Trimmed ICP (point-to-point, similarity via Horn's quaternion method) between ARKit face
    /// vertices and the depth points of the same frames.
    static func estimate(_ samples: [Sample], iterations: Int = 30) -> Fit? {
        struct FrameData {
            var q: [Vec3]           // ARKit vertices in camera frame
            var grid: [SIMD3<Int32>: [Vec3]]
        }
        let cell: Float = 0.004
        func key(_ p: Vec3) -> SIMD3<Int32> { SIMD3<Int32>((p / cell).rounded(.down)) }

        var frames: [FrameData] = []
        var ratios: [Float] = []
        for s in samples {
            let camFromFace = s.arPose.inverse
            let q = s.vertices.map { camFromFace.transformPoint($0) }
            // Depth points in a generous window around the projected face.
            var minU = Int.max, maxU = Int.min, minV = Int.max, maxV = Int.min
            let img = s.image
            for p in q where -p.z > 0.1 {
                let u = Int(img.fx * p.x / -p.z + img.cx), v = Int(img.fy * -p.y / -p.z + img.cy)
                minU = min(minU, u); maxU = max(maxU, u); minV = min(minV, v); maxV = max(maxV, v)
                if u >= 0, v >= 0, u < img.width, v < img.height {
                    let d = img.values[v * img.width + u]
                    if d > 0 { ratios.append(d / -p.z) }
                }
            }
            guard minU < maxU else { continue }
            let pad = (maxU - minU) / 3
            var grid: [SIMD3<Int32>: [Vec3]] = [:]
            for v in max(0, minV - pad)..<min(img.height, maxV + pad) {
                for u in max(0, minU - pad)..<min(img.width, maxU + pad) where (u + v) & 1 == 0 {
                    let d = img.values[v * img.width + u]
                    if d <= 0 || d > 1 { continue }
                    let p = img.cameraPoint(x: u, y: v, depth: d)
                    grid[key(p), default: []].append(p)
                }
            }
            frames.append(FrameData(q: q, grid: grid))
        }
        // ARKit's ambiguity is a scale about the camera centre, which the depth measures directly
        // along each ray. Fixing the scale this way avoids trading scale against depth offset in
        // the ICP (a frontal face constrains that poorly); ICP then fits the small rigid remainder.
        guard !frames.isEmpty, let scale = ratios.median(), scale > 0.5, scale < 2 else { return nil }
        var c = PoseCorrection(scale: scale, rotation: matrix_identity_float3x3, translation: .zero)
        func nearest(_ p: Vec3, _ grid: [SIMD3<Int32>: [Vec3]]) -> (Vec3, Float)? {
            let k = key(p)
            var best: (Vec3, Float)? = nil
            for dz: Int32 in -1...1 { for dy: Int32 in -1...1 { for dx: Int32 in -1...1 {
                guard let pts = grid[k &+ SIMD3(dx, dy, dz)] else { continue }
                for x in pts {
                    let d2 = simd_distance_squared(x, p)
                    if best == nil || d2 < best!.1 { best = (x, d2) }
                }
            }}}
            return best
        }
        func pairs(_ c: PoseCorrection) -> [(src: Vec3, dst: Vec3, d2: Float)] {
            var out: [(Vec3, Vec3, Float)] = []
            for f in frames {
                for q in f.q {
                    let moved = c.scale * (c.rotation * q) + c.translation
                    if let (p, d2) = nearest(moved, f.grid) { out.append((q, p, d2)) }
                }
            }
            return out
        }

        let initialPairs = pairs(.identity)
        let rmsBefore = initialPairs.isEmpty ? Float.infinity
            : sqrt(initialPairs.map { min($0.d2, 0.0004) }.reduce(0, +) / Float(initialPairs.count))
        var used = 0
        var rmsAfter = Float.infinity
        for it in 0..<iterations {
            var ps = pairs(c)
            guard ps.count > 300 else { return nil }
            ps.sort { $0.d2 < $1.d2 }
            let keep = ps.prefix(Int(Float(ps.count) * (it < 5 ? 0.6 : 0.75)))
            let solved = similarity(keep.map { ($0.src, $0.dst) }, fixedScale: scale)
            guard let solved else { return nil }
            c = solved
            used = keep.count
            rmsAfter = sqrt(keep.map { $0.d2 }.reduce(0, +) / Float(keep.count))
        }
        return Fit(correction: c, rmsBefore: rmsBefore, rmsAfter: rmsAfter, pairs: used)
    }

    /// Least-squares similarity dst ≈ s·R·src + t (Horn 1987, quaternion form).
    static func similarity(_ pairs: [(Vec3, Vec3)], fixedScale: Float? = nil) -> PoseCorrection? {
        guard pairs.count >= 3 else { return nil }
        let n = Float(pairs.count)
        let ms = pairs.reduce(Vec3.zero) { $0 + $1.0 } / n
        let md = pairs.reduce(Vec3.zero) { $0 + $1.1 } / n
        var sxx: Float = 0, sxy: Float = 0, sxz: Float = 0, syx: Float = 0, syy: Float = 0, syz: Float = 0
        var szx: Float = 0, szy: Float = 0, szz: Float = 0
        var ss: Float = 0
        for (a, b) in pairs {
            let p = a - ms, q = b - md
            sxx += p.x * q.x; sxy += p.x * q.y; sxz += p.x * q.z
            syx += p.y * q.x; syy += p.y * q.y; syz += p.y * q.z
            szx += p.z * q.x; szy += p.z * q.y; szz += p.z * q.z
            ss += simd_length_squared(p)
        }
        guard ss > 0 else { return nil }
        var k: [[Float]] = [
            [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
            [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
            [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
            [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
        ]
        let vec = largestEigenvector4(&k)
        let q = simd_quatf(ix: vec[1], iy: vec[2], iz: vec[3], r: vec[0]).normalized
        let r = simd_float3x3(q)
        var num: Float = 0
        for (a, b) in pairs { num += simd_dot(b - md, r * (a - ms)) }
        let s = fixedScale ?? num / ss
        guard s > 0.5, s < 2 else { return nil }
        return PoseCorrection(scale: s, rotation: r, translation: md - s * (r * ms))
    }

    /// Jacobi eigen-decomposition of a symmetric 4x4 matrix; returns the eigenvector of the largest eigenvalue.
    private static func largestEigenvector4(_ a: inout [[Float]]) -> [Float] {
        var v: [[Float]] = (0..<4).map { i in (0..<4).map { $0 == i ? 1 : 0 } }
        for _ in 0..<50 {
            var off: Float = 0
            for i in 0..<4 { for j in (i + 1)..<4 { off += a[i][j] * a[i][j] } }
            if off < 1e-18 { break }
            for p in 0..<3 {
                for q in (p + 1)..<4 where abs(a[p][q]) > 1e-12 {
                    let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                    let t = (theta >= 0 ? 1 : -1) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot(), s = t * c
                    for k in 0..<4 {
                        let akp = a[k][p], akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for k in 0..<4 {
                        let apk = a[p][k], aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for k in 0..<4 {
                        let vkp = v[k][p], vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }
        let best = (0..<4).max { a[$0][$0] < a[$1][$1] }!
        return (0..<4).map { v[$0][best] }
    }
}
