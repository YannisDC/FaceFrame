import simd

typealias Vec3 = SIMD3<Float>

extension simd_float4x4 {
    @inline(__always) func transformPoint(_ p: Vec3) -> Vec3 {
        let r = self * SIMD4<Float>(p, 1)
        return Vec3(r.x, r.y, r.z)
    }

    @inline(__always) func transformDirection(_ d: Vec3) -> Vec3 {
        let r = self * SIMD4<Float>(d, 0)
        return Vec3(r.x, r.y, r.z)
    }

    var translation: Vec3 { Vec3(columns.3.x, columns.3.y, columns.3.z) }

    /// Rotation angle (radians) of the upper-left 3x3 block.
    var rotationAngle: Float {
        let trace = columns.0.x + columns.1.y + columns.2.z
        return acos(max(-1, min(1, (trace - 1) / 2)))
    }
}

enum RigidMath {
    /// Exponential map of a small twist (rotation vector `w`, translation `v`), returned as a 4x4
    /// rigid transform meant to be left-multiplied onto an existing pose.
    static func twist(w: SIMD3<Double>, v: SIMD3<Double>) -> simd_float4x4 {
        let theta = simd_length(w)
        var r = matrix_identity_double3x3
        if theta > 1e-12 {
            let k = w / theta
            let kx = simd_double3x3(rows: [
                SIMD3(0, -k.z, k.y),
                SIMD3(k.z, 0, -k.x),
                SIMD3(-k.y, k.x, 0),
            ])
            r = matrix_identity_double3x3 + sin(theta) * kx + (1 - cos(theta)) * (kx * kx)
        }
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4<Float>(Float(r.columns.0.x), Float(r.columns.0.y), Float(r.columns.0.z), 0)
        m.columns.1 = SIMD4<Float>(Float(r.columns.1.x), Float(r.columns.1.y), Float(r.columns.1.z), 0)
        m.columns.2 = SIMD4<Float>(Float(r.columns.2.x), Float(r.columns.2.y), Float(r.columns.2.z), 0)
        m.columns.3 = SIMD4<Float>(Float(v.x), Float(v.y), Float(v.z), 1)
        return m
    }

    /// Rigid interpolation between two poses (slerp rotation, lerp translation).
    static func interpolate(_ a: simd_float4x4, _ b: simd_float4x4, _ t: Float) -> simd_float4x4 {
        let q = simd_slerp(simd_quatf(a), simd_quatf(b), t)
        var m = simd_float4x4(q)
        m.columns.3 = a.columns.3 + (b.columns.3 - a.columns.3) * t
        m.columns.3.w = 1
        return m
    }

    /// Continues the motion from `a` to `b` by `factor` more steps (b · (b·a⁻¹)^factor).
    static func extrapolate(_ a: simd_float4x4, _ b: simd_float4x4, _ factor: Float) -> simd_float4x4 {
        let rel = b * a.inverse
        let q = simd_quatf(rel)
        let angle = q.angle * factor
        var step = simd_float4x4(abs(q.angle) > 1e-6 ? simd_quatf(angle: angle, axis: q.axis) : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
        step.columns.3 = SIMD4(rel.translation * factor, 1)
        return step * b
    }

    /// Solves the dense linear system `a * x = b` (row-major `a`, n x n) with partial pivoting.
    static func solve(_ a: [Double], _ b: [Double], n: Int) -> [Double]? {
        var m = a
        var x = b
        for col in 0..<n {
            var pivot = col
            var best = abs(m[col * n + col])
            for row in (col + 1)..<n where abs(m[row * n + col]) > best {
                best = abs(m[row * n + col])
                pivot = row
            }
            if best < 1e-15 { return nil }
            if pivot != col {
                for k in 0..<n { m.swapAt(col * n + k, pivot * n + k) }
                x.swapAt(col, pivot)
            }
            let diag = m[col * n + col]
            for row in (col + 1)..<n {
                let f = m[row * n + col] / diag
                if f == 0 { continue }
                for k in col..<n { m[row * n + k] -= f * m[col * n + k] }
                x[row] -= f * x[col]
            }
        }
        for row in stride(from: n - 1, through: 0, by: -1) {
            var s = x[row]
            for k in (row + 1)..<n { s -= m[row * n + k] * x[k] }
            x[row] = s / m[row * n + row]
        }
        return x
    }
}

extension Array where Element == Float {
    func median() -> Float? {
        guard !isEmpty else { return nil }
        let s = sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
    }
}
