import Foundation
import simd

/// Ports Tools/gnm_align_to_scan.py: the head fitted to the photos keeps its identity and
/// expression; the metric scan gives it its pose and real-world size. Robust symmetric
/// point-to-plane ICP on the regions TrueDepth sees as skin solves pitch and translation (the scan's
/// frame is already level and centred), at each scale of a grid; the scale whose midline profile
/// (forehead, nose, lips, chin) matches the scan best wins. The profile is used rather than the
/// whole surface because surface distances reward shrinking the head onto the side of the skull.
struct GNMScaleAligner {
    struct Result {
        /// scan = scale · rotation · model + translation (metres).
        var scale: Float
        var rotation: simd_float3x3
        var translation: Vec3
        /// The model in the scan frame.
        var vertices: [Vec3]
        /// Median midline profile error at each scale tried (mm).
        var profileErrorMM: [(scale: Float, errorMM: Float)]
        /// RMS of the symmetric point-to-plane residuals of the last ICP round (mm).
        var rmsMM: Float
    }

    static let fitGroups = ["hockey_mask", "ears", "left_parotid_region", "right_parotid_region", "left_zygomatic_region",
                            "right_zygomatic_region", "left_temple_region", "right_temple_region"]
    static let thresholds: [Float] = [0.015, 0.010, 0.006, 0.004, 0.003, 0.003, 0.003, 0.003]
    static let maxNormalAngleDeg: Float = 50
    static let fitPoints = 80_000
    static let profileBelow: Float = 0.075, profileAbove: Float = 0.060
    static let scaleGrid: [Float] = stride(from: 0, through: 14, by: 1).map { 0.90 + 0.01 * Float($0) }

    let model: GNMPhotoModel
    /// The model at the photo fit's identity and expression (metres, GNM frame).
    let vertices: [Vec3]
    let normals: [Vec3]
    private let fitIndices: [Int]
    private let earMask: [Bool]
    private let skinMask: [Bool]

    init(model: GNMPhotoModel, identity: [Float], expression: [Float]) {
        self.model = model
        let flat = model.vertices(identity: identity, expression: expression)
        let v = (0..<model.vertexCount).map { Vec3(flat[3 * $0], flat[3 * $0 + 1], flat[3 * $0 + 2]) }
        vertices = v
        // Vertex normals over the triangles outside the eye exteriors (gnm.triangles_group('~eye_exteriors')).
        let exterior = model.groups["eye_exteriors"] ?? []
        var mesh = TriMesh()
        mesh.vertices = v
        mesh.triangles = model.triangles.filter { t in !(exterior[safe: Int(t.x)] ?? false) || !(exterior[safe: Int(t.y)] ?? false) || !(exterior[safe: Int(t.z)] ?? false) }
        mesh.computeNormals()
        normals = mesh.normals
        let skin = model.groups["skin"] ?? []
        skinMask = skin
        earMask = (0..<model.vertexCount).map { skin[$0] && (model.groups["ears"]?[$0] ?? false) }
        fitIndices = (0..<model.vertexCount).filter { i in skin[i] && Self.fitGroups.contains { model.groups[$0]?[i] ?? false } }
    }

    /// - Parameters: scan: the scan (head frame, metres, with normals); eyes: its eye centres (for the nose tip search).
    ///   - scale: a known metric scale (as gnm_align_to_scan.py's --scale), else the profile picks one from the grid.
    func align(scan: TriMesh, eyeLeft: Vec3, eyeRight: Vec3, scale known: Float? = nil, log: (String) -> Void = { _ in }) -> Result? {
        guard scan.normals.count == scan.vertices.count, !scan.vertices.isEmpty else { return nil }
        var rng = SplitMix64(seed: 0)
        let count = min(Self.fitPoints, scan.vertices.count)
        let sample = Array((0..<scan.vertices.count).shuffled(using: &rng).prefix(count))
        let p = sample.map { scan.vertices[$0] }, pn = sample.map { scan.normals[$0] }

        // Initial pose: scan and GNM share the ARKit face axes; match the nose tips.
        let face = fitIndices.filter { abs(vertices[$0].x) < 0.015 }
        guard let modelTip = face.max(by: { vertices[$0].z < vertices[$1].z }).map({ vertices[$0] }) else { return nil }
        let mid = (eyeLeft + eyeRight) / 2
        let near = p.filter { abs($0.x - mid.x) < 0.015 && $0.y < mid.y + 0.005 && $0.y > mid.y - 0.065 }
        guard let scanTip = near.max(by: { $0.z < $1.z }) else { return nil }
        var pitch: Float = 0
        var t = scanTip - modelTip
        let center = fitIndices.reduce(Vec3.zero) { $0 + vertices[$1] } / Float(fitIndices.count)

        var best: (error: Float, result: Result)?
        var curve: [(Float, Float)] = []
        var solved = solve(p, pn, pitch: &pitch, t: &t, scale: 1, center: center)
        for s in known.map({ [$0] }) ?? Self.scaleGrid {
            solved = solve(p, pn, pitch: &pitch, t: &t, scale: s, center: center)
            let r = transform(pitch: pitch, t: t, scale: s, center: center)
            let error = profileError(r, scale: s, scan: scan)
            curve.append((s, error))
            log(String(format: "scale %.2f: midline profile error %.2f mm", s, error))
            let result = Result(scale: s, rotation: r.rotation, translation: r.translation,
                                vertices: vertices.map { s * (r.rotation * $0) + r.translation }, profileErrorMM: [], rmsMM: solved)
            if error.isFinite, best == nil || error < best!.error { best = (error, result) }
        }
        guard var out = best?.result else { return nil }
        out.profileErrorMM = curve
        log(String(format: "metric scale %.2f", out.scale))
        return out
    }

    /// x_scan = s R x + T, with the rotation about the head's x axis (pitch) and the scale about `center`.
    private func transform(pitch: Float, t: Vec3, scale s: Float, center c: Vec3) -> (rotation: simd_float3x3, translation: Vec3) {
        let r = rotX(pitch)
        return (r, c - s * (r * c) + t)
    }

    private func rotX(_ a: Float) -> simd_float3x3 {
        simd_float3x3(rows: [Vec3(1, 0, 0), Vec3(0, cos(a), -sin(a)), Vec3(0, sin(a), cos(a))])
    }

    /// Robust symmetric point-to-plane ICP (Geman-McClure), coarse to fine; returns the last round's RMS (mm).
    @discardableResult
    private func solve(_ p: [Vec3], _ pn: [Vec3], pitch: inout Float, t: inout Vec3, scale s: Float, center c: Vec3) -> Float {
        let cosMax = cos(Self.maxNormalAngleDeg * .pi / 180)
        var rms: Float = 0
        for threshold in Self.thresholds {
            let r = rotX(pitch)
            let vw = fitIndices.map { s * (r * (vertices[$0] - c)) + c + t }
            let nw = fitIndices.map { r * normals[$0] }
            let grid = PointGrid(points: vw, normals: nw, cell: max(threshold, 0.004))
            var pairs: [(src: Int, dst: Vec3, dn: Vec3)] = []
            for (i, q) in p.enumerated() {
                guard let j = grid.nearestIndex(q, maxDistance: threshold), simd_dot(nw[j], pn[i]) > cosMax else { continue }
                pairs.append((fitIndices[j], q, pn[i]))
            }
            guard pairs.count > 100 else { continue }
            let sigma = threshold / 3
            // Gauss-Newton with Geman-McClure weights (in place of L-BFGS), on pitch and translation.
            for _ in 0..<10 {
                let r = rotX(pitch)
                let dr = simd_float3x3(rows: [Vec3(0, 0, 0), Vec3(0, -sin(pitch), -cos(pitch)), Vec3(0, cos(pitch), -sin(pitch))])
                var h = [Double](repeating: 0, count: 16), g = [Double](repeating: 0, count: 4)
                var sum: Double = 0
                for pr in pairs {
                    let local = vertices[pr.src] - c
                    let q = s * (r * local) + c + t
                    let qn = r * normals[pr.src]
                    let nsum = qn + pr.dn
                    let d = simd_dot(pr.dst - q, nsum) / 2
                    let w = Double(sigma * sigma) / pow(Double(d * d + sigma * sigma), 2)
                    let dq = s * (dr * local), dqn = dr * normals[pr.src]
                    let jp = (simd_dot(-dq, nsum) + simd_dot(pr.dst - q, dqn)) / 2
                    let jac = [Double(jp), Double(-nsum.x / 2), Double(-nsum.y / 2), Double(-nsum.z / 2)]
                    for a in 0..<4 {
                        g[a] += w * jac[a] * Double(d)
                        for b in 0..<4 { h[a * 4 + b] += w * jac[a] * jac[b] }
                    }
                    sum += Double(d * d)
                }
                for a in 0..<4 { h[a * 4 + a] += 1e-12 }
                guard let x = RigidMath.solve(h, g.map { -$0 }, n: 4) else { break }
                pitch += Float(x[0]); t += Vec3(Float(x[1]), Float(x[2]), Float(x[3]))
                rms = Float(sqrt(sum / Double(pairs.count))) * 1000
                if abs(x[0]) < 1e-6 && abs(x[1]) + abs(x[2]) + abs(x[3]) < 1e-7 { break }
            }
        }
        return rms
    }

    /// Median distance between the model's and the scan's midline profiles (mm): both cut by the head's
    /// midsagittal plane, anchored at their nose tips, compared as the frontmost depth at each height.
    private func profileError(_ tr: (rotation: simd_float3x3, translation: Vec3), scale: Float, scan: TriMesh) -> Float {
        let r = tr.rotation, tt = tr.translation
        let side = r.columns.0, up = r.columns.1, forward = r.columns.2
        // Section of the skin (without the ears) by the plane through T with normal R[:, 0].
        var model2: [SIMD2<Float>] = []
        for tri in model.triangles {
            let ids = [Int(tri.x), Int(tri.y), Int(tri.z)]
            guard ids.allSatisfy({ skinMask[$0] && !earMask[$0] }) else { continue }
            let w = ids.map { scale * (r * vertices[$0]) + tt }
            let d = w.map { simd_dot($0 - tt, side) }
            for k in 0..<3 {
                let a = k, b = (k + 1) % 3
                if (d[a] < 0) != (d[b] < 0) {
                    let f = d[a] / (d[a] - d[b])
                    let x = w[a] + f * (w[b] - w[a])
                    model2.append(SIMD2(simd_dot(x, up), simd_dot(x, forward)))
                }
            }
        }
        let scan2 = scan.vertices.filter { abs(simd_dot($0 - tt, side)) < 0.001 }.map { SIMD2(simd_dot($0, up), simd_dot($0, forward)) }
        func anchored(_ pr: [SIMD2<Float>]) -> [SIMD2<Float>] {
            guard let tip = pr.max(by: { $0.y < $1.y }) else { return [] }
            return pr.map { $0 - tip }
        }
        let heights = stride(from: -Self.profileBelow, to: Self.profileAbove, by: 0.001).map { $0 }
        func frontmost(_ pr: [SIMD2<Float>]) -> [Float] {
            var out = [Float](repeating: .nan, count: heights.count)
            for q in pr {
                let lo = max(0, Int(((q.x - 0.0008 + Self.profileBelow) / 0.001).rounded(.up)))
                let hi = min(heights.count - 1, Int(((q.x + 0.0008 + Self.profileBelow) / 0.001).rounded(.down)))
                guard lo <= hi else { continue }
                for kk in lo...hi where abs(q.x - heights[kk]) < 0.0008 {
                    if out[kk].isNaN || q.y > out[kk] { out[kk] = q.y }
                }
            }
            return out
        }
        let a = frontmost(anchored(model2)), b = frontmost(anchored(scan2))
        var diffs: [Float] = []
        for i in heights.indices where a[i].isFinite && b[i].isFinite { diffs.append(abs(a[i] - b[i])) }
        guard !diffs.isEmpty else { return .infinity }
        return diffs.sorted()[diffs.count / 2] * 1000
    }
}

/// Deterministic random numbers (for the scan sample).
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
