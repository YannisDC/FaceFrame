import simd

/// Analytic head (signed distance field) in the ARKit face frame. Used for the in-app demo and
/// for validating the reconstruction and fitting pipeline without a TrueDepth device.
struct SyntheticHead {
    var eyeCenterLeft = Vec3(0.032, 0.03, 0.028)
    var eyeCenterRight = Vec3(-0.032, 0.03, 0.028)
    /// Nose side angle from the sagittal plane (top view).
    var noseSplayDegrees: Float = 40
    var earTiltDegrees: Float = 22

    func distance(_ p: Vec3) -> Float {
        var d = ellipsoid(p - Vec3(0, 0, -0.035), Vec3(0.075, 0.105, 0.095))
        d = smoothUnion(d, ellipsoid(p - Vec3(0, 0.05, 0.035), Vec3(0.06, 0.014, 0.02)), 0.01) // brow
        for e in [eyeCenterLeft, eyeCenterRight] {
            d = max(d, -(simd_distance(p, e) - 0.02)) // eye socket
            d = min(d, simd_distance(p, e) - 0.0125)   // eyeball
        }
        // Nose: wedge whose ridge runs from the bridge (y=0.032) to the tip (y=-0.012).
        let a = noseSplayDegrees * .pi / 180
        let ridgeZ = 0.052 + (0.032 - p.y) * 0.75
        var nose = (p.z - (ridgeZ - abs(p.x) / tan(a))) * sin(a)
        nose = max(nose, p.y - 0.034, -0.012 - p.y, 0.03 - p.z)
        d = smoothUnion(d, nose, 0.004)
        // Ears: thin discs tilted away from the head so a groove forms above the attachment.
        for s: Float in [1, -1] {
            var q = p - Vec3(s * 0.073, 0.012, -0.058)
            q.x *= s
            let t = earTiltDegrees * .pi / 180
            let qx = q.x * cos(t) - q.y * sin(t)
            let qy = q.x * sin(t) + q.y * cos(t)
            let ear = ellipsoid(Vec3(qx, qy, q.z), Vec3(0.0045, 0.031, 0.018))
            d = smoothUnion(d, ear, 0.0015)
        }
        return d
    }

    private func ellipsoid(_ p: Vec3, _ r: Vec3) -> Float {
        let k0 = simd_length(p / r)
        let k1 = simd_length(p / (r * r))
        return k1 > 0 ? k0 * (k0 - 1) / k1 : -r.min()
    }

    private func smoothUnion(_ a: Float, _ b: Float, _ k: Float) -> Float {
        let h = max(0, min(1, 0.5 + 0.5 * (b - a) / k))
        return b * (1 - h) + a * h - k * h * (1 - h)
    }

    /// Sphere-traces a ray; returns the hit distance.
    func raycast(origin: Vec3, direction: Vec3, maxDistance: Float = 1) -> Float? {
        var t: Float = 0
        for _ in 0..<128 {
            let d = distance(origin + direction * t)
            if d < 0.00005 { return t }
            t += d * 0.9
            if t > maxDistance { return nil }
        }
        return nil
    }
}

extension TSDFVolume {
    /// Writes an analytic signed distance field directly into the volume (demo mode).
    func fill(sdf: (Vec3) -> Float, observedWeight: UInt8 = 10) {
        let blockSize = voxelSize * Float(1 << TSDFVolume.blockShift)
        let reach = blockSize * 0.9 + truncFront
        for bz in 0..<blockDims.z {
            for by in 0..<blockDims.y {
                for bx in 0..<blockDims.x {
                    let center = origin + (Vec3(Float(bx), Float(by), Float(bz)) + 0.5) * blockSize
                    if abs(sdf(center)) > reach { continue }
                    let base = SIMD3<Int32>(bx, by, bz) &<< TSDFVolume.blockShift
                    var values = [(Int16, UInt8)]()
                    values.reserveCapacity(TSDFVolume.voxelsPerBlock)
                    for li in 0..<TSDFVolume.voxelsPerBlock {
                        let v = base &+ SIMD3<Int32>(Int32(li & 7), Int32((li >> 3) & 7), Int32(li >> 6))
                        let d = sdf(voxelCenter(v))
                        if d < -truncBack || d > truncFront {
                            values.append((d > 0 ? Int16.max : Int16(-truncBack / truncFront * 32767), d > 0 ? observedWeight : 0))
                        } else {
                            values.append((Int16(d / truncFront * 32767), observedWeight))
                        }
                    }
                    setBlock(base, values: values)
                }
            }
        }
    }
}
