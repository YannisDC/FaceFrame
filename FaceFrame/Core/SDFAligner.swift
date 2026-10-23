import simd

/// Rigidly aligns a depth frame to the fused volume by minimising the signed distance of each
/// sample (Gauss-Newton on point-to-SDF residuals, Huber-weighted). This removes the millimetre
/// jitter of ARKit's head pose and keeps tracking when the head is turned past the point where
/// ARKit loses the face (needed to see the ears).
struct SDFAligner {
    struct Result {
        var pose: simd_float4x4
        var rmsMeters: Float
        var inlierRatio: Float
        var converged: Bool
    }

    var iterations = 8
    var huberDelta: Float = 0.001
    var minValidPoints = 250
    /// Weak prior keeping the correction close to the initial pose (use when the initial pose comes
    /// from ARKit). It only matters in directions the geometry does not constrain (smooth cheeks,
    /// the side of the head). nil disables the prior.
    var translationPriorSigma: Float? = nil
    var rotationPriorSigma: Float? = nil
    var residualSigma: Float = 0.001

    /// - Parameters:
    ///   - points: samples in camera space.
    ///   - initial: initial head-from-camera transform.
    func align(points: [Vec3], initial: simd_float4x4, volume: TSDFVolume) -> Result? {
        var pose = initial
        var rms: Float = .infinity
        var inliers: Float = 0
        let cutoff = volume.truncFront * 0.9
        var accW = SIMD3<Double>.zero, accV = SIMD3<Double>.zero
        let lambdaT = translationPriorSigma.map { Double(residualSigma * residualSigma / ($0 * $0)) } ?? 0
        let lambdaR = rotationPriorSigma.map { Double(residualSigma * residualSigma / ($0 * $0)) } ?? 0

        for _ in 0..<iterations {
            var h = [Double](repeating: 0, count: 36)
            var g = [Double](repeating: 0, count: 6)
            var sumSq: Double = 0
            var valid = 0
            var inlierCount = 0

            for pc in points {
                let q = pose.transformPoint(pc)
                guard let s = volume.sample(q, minWeight: 2) else { continue }
                let r = s.distance
                if abs(r) >= cutoff || r <= -volume.truncBack * 0.9 { continue }
                let gl = simd_length(s.gradient)
                if gl < 0.3 { continue }
                let n = s.gradient / gl
                valid += 1
                if abs(r) < 0.0015 { inlierCount += 1 }
                let w = abs(r) <= huberDelta ? 1 : huberDelta / abs(r)
                let qx = simd_cross(q, n)
                let j: [Double] = [Double(qx.x), Double(qx.y), Double(qx.z), Double(n.x), Double(n.y), Double(n.z)]
                let wr = Double(w * r)
                for a in 0..<6 {
                    g[a] += j[a] * wr
                    for b in a..<6 { h[a * 6 + b] += Double(w) * j[a] * j[b] }
                }
                sumSq += Double(r * r)
            }
            guard valid >= minValidPoints else { return nil }
            for a in 0..<6 {
                for b in 0..<a { h[a * 6 + b] = h[b * 6 + a] }
                h[a * 6 + a] *= 1.0001
                h[a * 6 + a] += 1e-9
            }
            for a in 0..<3 {
                h[a * 6 + a] += lambdaR
                g[a] += lambdaR * accW[a]
                h[(a + 3) * 6 + a + 3] += lambdaT
                g[a + 3] += lambdaT * accV[a]
            }
            rms = Float(sqrt(sumSq / Double(valid)))
            inliers = Float(inlierCount) / Float(valid)
            guard let x = RigidMath.solve(h, g.map { -$0 }, n: 6) else { return nil }
            let delta = RigidMath.twist(w: SIMD3(x[0], x[1], x[2]), v: SIMD3(x[3], x[4], x[5]))
            pose = delta * pose
            accW += SIMD3(x[0], x[1], x[2])
            accV += SIMD3(x[3], x[4], x[5])
            if simd_length(SIMD3(x[3], x[4], x[5])) < 2e-5 && simd_length(SIMD3(x[0], x[1], x[2])) < 2e-4 {
                break
            }
        }
        return Result(pose: pose, rmsMeters: rms, inlierRatio: inliers, converged: rms < 0.0015 && inliers > 0.45)
    }
}
