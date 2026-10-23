import simd

/// A metric depth image with pinhole intrinsics (pixel x right, y down, z forward), already
/// corrected for any mirroring so that the intrinsics apply directly.
struct DepthImage {
    var width: Int
    var height: Int
    /// Metres, row-major; 0 = no measurement.
    var values: [Float]
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float

    /// Back-projects a pixel into ARKit camera space (x right, y up, looking down -z).
    @inline(__always) func cameraPoint(x: Int, y: Int, depth z: Float) -> Vec3 {
        Vec3((Float(x) - cx) / fx * z, -(Float(y) - cy) / fy * z, -z)
    }
}

extension DepthImage {
    /// Per-pixel fusion weight (0 = do not use). Pixels within 2 px of a depth jump are dropped
    /// (TrueDepth "flying pixels" at silhouettes blend foreground and background and would carve
    /// thin structures such as the nose); the rest are weighted 1–4 by how squarely the surface
    /// faces the camera, so grazing views count less than frontal ones.
    func qualityWeights(edgeJump: Float = 0.01, edgeRadius: Int = 2, minCosine: Float = 0.2) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        let r = edgeRadius
        guard width > 2 * r + 2, height > 2 * r + 2 else { return out }
        values.withUnsafeBufferPointer { d in
            for y in (r + 1)..<(height - r - 1) {
                for x in (r + 1)..<(width - r - 1) {
                    let i = y * width + x
                    let z = d[i]
                    if z <= 0 { continue }
                    var edge = false
                    for dy in -r...r where !edge {
                        let row = i + dy * width
                        for dx in -r...r {
                            let q = d[row + dx]
                            if q <= 0 || abs(q - z) > edgeJump { edge = true; break }
                        }
                    }
                    if edge { continue }
                    let p = cameraPoint(x: x, y: y, depth: z)
                    let px = cameraPoint(x: x + 1, y: y, depth: d[i + 1]) - cameraPoint(x: x - 1, y: y, depth: d[i - 1])
                    let py = cameraPoint(x: x, y: y + 1, depth: d[i + width]) - cameraPoint(x: x, y: y - 1, depth: d[i - width])
                    let n = simd_cross(px, py)
                    let nl = simd_length(n)
                    if nl == 0 { continue }
                    let cosine = abs(simd_dot(n / nl, simd_normalize(p)))
                    if cosine < minCosine { continue }
                    out[i] = UInt8(1 + (3 * cosine * cosine).rounded(.down))
                }
            }
        }
        return out
    }
}
