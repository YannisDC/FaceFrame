import Foundation
import simd

/// Google GNM head model (https://github.com/google/GNM, Apache 2.0), identity part only:
/// vertices = template + Σ c_k · basis_k. Units metres; axes match the ARKit face frame
/// (+x subject's left, +y up, +z forward).
///
/// Loaded from the files GNMFrames' tools/extract_gnm.py writes (bundled in Resources/GNMFrames,
/// copied there by Tools/sync_gnmframes.sh), the same files the in-app frame designer uses.
struct GNMModel {
    let template: [Vec3]
    /// Identity basis, component-major: basis[k * vertexCount + i].
    let basis: [Vec3]
    let components: Int
    let triangles: [SIMD3<UInt32>]
    /// Per-vertex membership of the named groups listed in gnm_groups.txt.
    let groups: [String: [Bool]]
    let leftEyeCenter: Vec3
    let rightEyeCenter: Vec3

    var vertexCount: Int { template.count }

    /// The model bundled with the app, or nil when the resources are missing.
    static func bundled(components: Int = 120) -> GNMModel? {
        if let dir = resourceDirectoryOverride { return try? GNMModel(directory: dir, components: components) }
        guard let url = Bundle.main.url(forResource: "identity_basis", withExtension: "bin") else { return nil }
        return try? GNMModel(directory: url.deletingLastPathComponent(), components: components)
    }

    /// Where to load the model from outside the app bundle (Mac replay tool).
    static var resourceDirectoryOverride: URL?

    /// The folder of the bundled model files (Resources/GNMFrames).
    static var resourceDirectory: URL? {
        resourceDirectoryOverride ?? Bundle.main.url(forResource: "identity_basis", withExtension: "bin")?.deletingLastPathComponent()
    }

    init(directory: URL, components: Int = 120) throws {
        func data(_ name: String) throws -> Data { try Data(contentsOf: directory.appendingPathComponent(name)) }
        struct Meta: Decodable { var vertices: Int; var groups: [String] }
        let meta = try JSONDecoder().decode(Meta.self, from: data("meta.json"))
        let n = meta.vertices
        let t = try data("template.bin").withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard t.count == n * 3 else { throw CocoaError(.fileReadCorruptFile) }
        template = (0..<n).map { Vec3(t[$0 * 3], t[$0 * 3 + 1], t[$0 * 3 + 2]) }

        let basisData = try data("identity_basis.bin")
        let available = basisData.count / (n * 6)
        let k = min(components, available)
        self.components = k
        basis = basisData.withUnsafeBytes { raw in
            let h = raw.bindMemory(to: Float16.self)
            return (0..<(k * n)).map { Vec3(Float(h[$0 * 3]), Float(h[$0 * 3 + 1]), Float(h[$0 * 3 + 2])) }
        }

        let tri = try data("triangles.bin").withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        triangles = (0..<(tri.count / 3)).map { SIMD3(tri[$0 * 3], tri[$0 * 3 + 1], tri[$0 * 3 + 2]) }

        let packed = [UInt8](try data("groups.bin"))
        var groups: [String: [Bool]] = [:]
        for (gi, name) in meta.groups.enumerated() where (gi + 1) * n <= packed.count {
            groups[name] = packed[(gi * n)..<((gi + 1) * n)].map { $0 != 0 }
        }
        self.groups = groups

        // Joints: neck, head, left eye, right eye.
        let j = try data("joint_template.bin").withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        leftEyeCenter = Vec3(j[6], j[7], j[8])
        rightEyeCenter = Vec3(j[9], j[10], j[11])
        sellionVertex = findSellionVertex()
    }

    func member(_ group: String, _ i: Int) -> Bool { groups[group]?[i] ?? false }

    /// Vertices of one pupil (GNM's `pupils` group on that eye), as GNMFrames measures the PD.
    func pupilVertices(_ side: HeadSide) -> [Int] {
        (0..<vertexCount).filter { member("pupils", $0) && member(side == .left ? "left_eye" : "right_eye", $0) }
    }

    /// The template vertex at the sellion (deepest point of the nasal bridge on the midline), found
    /// once on the average head. Every fitted head moves this same vertex, so it is a consistent
    /// anatomical landmark across scans.
    private(set) var sellionVertex: Int?

    private func findSellionVertex() -> Int? {
        var skin = TriMesh()
        skin.vertices = template
        skin.triangles = triangles.filter { member("skin", Int($0.x)) && member("skin", Int($0.y)) && member("skin", Int($0.z)) }
        skin.compact()
        skin.computeNormals()
        let a = FitAnalyzer(mesh: skin, eyeCenterLeft: leftEyeCenter + Vec3(0, 0, 0.0125),
                            eyeCenterRight: rightEyeCenter + Vec3(0, 0, 0.0125))
        let target = a.sellionPoint
        return (0..<vertexCount).filter { member("skin", $0) && abs(template[$0].x) < 0.002 }
            .min { simd_distance(template[$0], target) < simd_distance(template[$1], target) }
    }

    func vertices(_ c: [Float]) -> [Vec3] {
        var v = template
        let n = vertexCount
        for k in 0..<min(c.count, components) where c[k] != 0 {
            let ck = c[k]
            for i in 0..<n { v[i] += ck * basis[k * n + i] }
        }
        return v
    }
}

/// Fits GNM's identity and head pose to a scanned head mesh (point-to-plane Gauss-Newton with a
/// Gaussian prior on the identity coefficients). Regions the scan does not see — behind the
/// ears, under hair — are filled in by the learned shape prior.
struct GNMFitter {
    struct Result {
        var coefficients: [Float]
        /// Scan-frame transform applied to model vertices: w = scale · R · v + t.
        var scale: Float
        var rotation: simd_float3x3
        var translation: Vec3
        var vertices: [Vec3]
        /// Vertices of the statistical fit before the face refinement (smooth, plausible everywhere).
        var statisticalVertices: [Vec3]
        var rmsMM: Float
        /// Per-vertex distance to the scan (metres), nil where the scan has no surface nearby.
        var support: [Float?]
    }

    let model: GNMModel
    var identityComponents = 80
    /// Gaussian prior on the identity coefficients; nil is GNM's own standard normal.
    var prior: GNMIdentityPrior?

    /// A point measured on the person (head frame, metres) that the centroid of some model vertices
    /// must match: the photo part of the fit, e.g. the pupils found in the colour frames, which the
    /// depth scan cannot see (the eyes are left out of the surface fit).
    struct Landmark {
        var vertices: [Int]
        var target: Vec3
    }
    var landmarks: [Landmark] = []
    /// Weight of each landmark coordinate, as a share of all the surface residuals of an iteration.
    var landmarkShare: Float = 0.05
    /// Expected point-to-plane residual (scan noise + model error); sets the prior strength.
    var residualSigma: Float = 0.0015
    /// Groups fitted to the scan. Cranium/forehead are left out: TrueDepth sees hair there, not skin.
    var fitGroups = ["hockey_mask", "ears", "left_parotid_region", "right_parotid_region",
                     "left_zygomatic_region", "right_zygomatic_region", "left_temple_region", "right_temple_region"]
    var excludedGroups = ["eye_interiors", "eyes", "mouth_sock", "tongue", "teeth", "gums"]
    /// Also pull the model towards every scan point (not only model points towards the scan).
    var bidirectional = true
    /// Non-rigid refinement after the statistical fit: vertices move along their normals towards the
    /// scan (smoothed over the mesh), so the nose and ears follow the person's real geometry where the
    /// scan has data and fade back to the model elsewhere. 0 disables.
    var refinementIterations = 4
    var refinementMaxOffset: Float = 0.006

    private func refine(_ start: [Vec3], grid: PointGrid, scanSamples: [Vec3], skinTriangles: [SIMD3<UInt32>]) -> [Vec3] {
        let n = start.count
        var w = start
        // Mesh adjacency over skin triangles
        var nbrs = [[Int32]](repeating: [], count: n)
        for t in skinTriangles {
            let a = Int(t.x), b = Int(t.y), c = Int(t.z)
            nbrs[a].append(Int32(b)); nbrs[a].append(Int32(c))
            nbrs[b].append(Int32(a)); nbrs[b].append(Int32(c))
            nbrs[c].append(Int32(a)); nbrs[c].append(Int32(b))
        }
        // Only the face (GNM's "hockey mask"): elsewhere TrueDepth sees hair, clothing or only a noisy,
        // oblique view of the ears, and following it made the model lumpy and moved the ear seats.
        let isSkin = (0..<n).map {
            model.member("hockey_mask", $0) && model.member("skin", $0) && !model.member("eye_interiors", $0)
                && !model.member("eyes", $0) && !model.member("mouth_sock", $0) && !model.member("ears", $0)
        }
        for _ in 0..<refinementIterations {
            var mesh = TriMesh()
            mesh.vertices = w
            mesh.triangles = skinTriangles
            mesh.computeNormals()
            // Normal offsets from both directions, averaged per vertex.
            var sum = [Float](repeating: 0, count: n), cnt = [Float](repeating: 0, count: n)
            for i in 0..<n where isSkin[i] {
                if let (q, _) = grid.nearest(w[i], maxDistance: refinementMaxOffset) {
                    sum[i] += simd_dot(mesh.normals[i], q - w[i]); cnt[i] += 1
                }
            }
            let skinIdx = (0..<n).filter { isSkin[$0] }
            let mg = PointGrid(points: skinIdx.map { w[$0] }, normals: skinIdx.map { mesh.normals[$0] }, cell: 0.004)
            for sp in scanSamples {
                guard let li = mg.nearestIndex(sp, maxDistance: refinementMaxOffset) else { continue }
                let i = skinIdx[li]
                sum[i] += simd_dot(mesh.normals[i], sp - w[i]); cnt[i] += 1
            }
            var offset = (0..<n).map { cnt[$0] > 0 ? sum[$0] / cnt[$0] : 0 }
            var weight = cnt.map { min(1, $0 / 2) }
            // Smooth offsets (and confidence) over the surface: suppresses sensor noise, fades to zero
            // where the scan has no data.
            for _ in 0..<20 {
                var no = offset, nw = weight
                for i in 0..<n where isSkin[i] && !nbrs[i].isEmpty {
                    var so: Float = offset[i] * weight[i], sw: Float = weight[i]
                    for j in nbrs[i] { so += offset[Int(j)] * weight[Int(j)]; sw += weight[Int(j)] }
                    no[i] = sw > 0 ? so / sw : 0
                    nw[i] = (weight[i] + nbrs[i].reduce(0) { $0 + weight[Int($1)] }) / Float(nbrs[i].count + 1)
                }
                offset = no; weight = nw
            }
            for i in 0..<n where isSkin[i] {
                let o = max(-refinementMaxOffset, min(refinementMaxOffset, offset[i])) * min(1, weight[i] * 2)
                w[i] += mesh.normals[i] * o
            }
        }
        return w
    }

    func fit(scan: TriMesh, leftEye: Vec3, rightEye: Vec3, log: (String) -> Void = { _ in }) -> Result? {
        guard scan.normals.count == scan.vertices.count else { return nil }
        let n = model.vertexCount
        let k = min(identityComponents, model.components)
        let fitMask = (0..<n).map { i in
            fitGroups.contains { model.member($0, i) } && !excludedGroups.contains { model.member($0, i) } && model.member("skin", i)
        }
        let grid = PointGrid(points: scan.vertices, normals: scan.normals, cell: 0.004)
        // Scan samples for the scan→model term (each scan point pulls its nearest model point). Without
        // it, model vertices are satisfied by nearby scan points and a larger real nose or ear never
        // pulls the model outward (seen on device scans: model nose ~3 mm inside the scanned nose).
        let sampleStride = max(1, scan.vertices.count / 20000)
        let scanSamples = stride(from: 0, to: scan.vertices.count, by: sampleStride).map { scan.vertices[$0] }
        let maskedIndices = (0..<n).filter { fitMask[$0] }
        let skinTriangles = model.triangles.filter {
            model.member("skin", Int($0.x)) && model.member("skin", Int($0.y)) && model.member("skin", Int($0.z))
        }

        // 1. Initial placement from the eyes (ARKit's eye points sit near the front of the eye; GNM's
        //    are eyeball centres, so depth is refined by the rigid face stage below).
        let mModel = (model.leftEyeCenter + model.rightEyeCenter) / 2
        let mScan = (leftEye + rightEye) / 2
        let scale: Float = 1 // both GNM and TrueDepth are metric; identity handles head size
        var rotation = matrix_identity_float3x3
        var translation = mScan - scale * mModel + Vec3(0, 0, -0.012)
        let priorMean = (0..<k).map { prior?.mean[safe: $0] ?? 0 }
        let priorVariance = (0..<k).map { kk -> Float in let s = prior?.std[safe: kk] ?? 1; return max(s * s, 1e-4) }
        var c = priorMean

        func transformed(_ v: [Vec3]) -> [Vec3] { v.map { scale * (rotation * $0) + translation } }

        let stages: [(iterations: Int, maxDist: Float, identity: Bool)] = [(8, 0.015, false), (6, 0.008, true), (8, 0.005, true)]
        var rms: Float = .infinity
        for stage in stages {
            for _ in 0..<stage.iterations {
                let v = model.vertices(c)
                let w = transformed(v)
                let p = stage.identity ? k + 6 : 6
                var h = [Double](repeating: 0, count: p * p)
                var g = [Double](repeating: 0, count: p)
                var sumSq: Double = 0
                var count = 0
                var j = [Double](repeating: 0, count: p)
                for i in 0..<n where fitMask[i] {
                    guard let (q, nq) = grid.nearest(w[i], maxDistance: stage.maxDist) else { continue }
                    let r = simd_dot(nq, w[i] - q)
                    let hub: Float = abs(r) < 0.002 ? 1 : 0.002 / abs(r)
                    let wx = simd_cross(w[i], nq)
                    j[0] = Double(wx.x); j[1] = Double(wx.y); j[2] = Double(wx.z)
                    j[3] = Double(nq.x); j[4] = Double(nq.y); j[5] = Double(nq.z)
                    if stage.identity {
                        let sn = scale * (rotation.transpose * nq) // n · (s R b) = (s Rᵀ n) · b
                        for kk in 0..<k { j[6 + kk] = Double(simd_dot(sn, model.basis[kk * n + i])) }
                    }
                    let wr = Double(hub * r)
                    for a in 0..<p {
                        let ja = j[a]
                        if ja == 0 { continue }
                        g[a] += ja * wr
                        for b in a..<p { h[a * p + b] += Double(hub) * ja * j[b] }
                    }
                    sumSq += Double(r * r)
                    count += 1
                }
                if stage.identity && bidirectional {
                    // Scan → model: point-to-plane against the model's surface normal at its nearest vertex.
                    var mesh = TriMesh()
                    mesh.vertices = w
                    mesh.triangles = skinTriangles
                    mesh.computeNormals()
                    let mg = PointGrid(points: maskedIndices.map { w[$0] }, normals: maskedIndices.map { mesh.normals[$0] }, cell: 0.004)
                    for sp in scanSamples {
                        guard let li = mg.nearestIndex(sp, maxDistance: stage.maxDist) else { continue }
                        let i = maskedIndices[li]
                        let nq = mesh.normals[i]
                        let r = simd_dot(nq, w[i] - sp) // same sign convention as the model→scan term
                        let hub: Float = abs(r) < 0.002 ? 1 : 0.002 / abs(r)
                        let wx = simd_cross(w[i], nq)
                        j[0] = Double(wx.x); j[1] = Double(wx.y); j[2] = Double(wx.z)
                        j[3] = Double(nq.x); j[4] = Double(nq.y); j[5] = Double(nq.z)
                        let sn = scale * (rotation.transpose * nq)
                        for kk in 0..<k { j[6 + kk] = Double(simd_dot(sn, model.basis[kk * n + i])) }
                        let wr = Double(hub * r)
                        for a in 0..<p {
                            let ja = j[a]
                            if ja == 0 { continue }
                            g[a] += ja * wr
                            for b in a..<p { h[a * p + b] += Double(hub) * ja * j[b] }
                        }
                        sumSq += Double(r * r)
                        count += 1
                    }
                }
                guard count > 500 else { return nil }
                if stage.identity {
                    // Landmarks: three point-to-plane residuals each (one per axis) on the vertices' centroid.
                    let lw = Double(landmarkShare) * Double(count)
                    for lm in landmarks where !lm.vertices.isEmpty {
                        let inv = 1 / Float(lm.vertices.count)
                        let pt = lm.vertices.reduce(Vec3.zero) { $0 + w[$1] } * inv
                        for axis in 0..<3 {
                            var e = Vec3.zero
                            e[axis] = 1
                            let r = Double(pt[axis] - lm.target[axis])
                            let wx = simd_cross(pt, e)
                            j[0] = Double(wx.x); j[1] = Double(wx.y); j[2] = Double(wx.z)
                            j[3] = Double(e.x); j[4] = Double(e.y); j[5] = Double(e.z)
                            let sn = scale * (rotation.transpose * e)
                            for kk in 0..<k {
                                var d: Float = 0
                                for i in lm.vertices { d += simd_dot(sn, model.basis[kk * n + i]) }
                                j[6 + kk] = Double(d * inv)
                            }
                            for a in 0..<p where j[a] != 0 {
                                g[a] += lw * j[a] * r
                                for b in a..<p { h[a * p + b] += lw * j[a] * j[b] }
                            }
                        }
                    }
                }
                for a in 0..<p { for b in 0..<a { h[a * p + b] = h[b * p + a] } }
                if stage.identity {
                    let lambda = Double(residualSigma * residualSigma)
                    for kk in 0..<k {
                        let lk = lambda / Double(priorVariance[kk])
                        h[(6 + kk) * p + 6 + kk] += lk
                        g[6 + kk] += lk * Double(c[kk] - priorMean[kk])
                    }
                }
                for a in 0..<6 { h[a * p + a] += 1e-7 }
                rms = Float(sqrt(sumSq / Double(count)))
                guard let x = RigidMath.solve(h, g.map { -$0 }, n: p) else { return nil }
                let delta = RigidMath.twist(w: SIMD3(x[0], x[1], x[2]), v: SIMD3(x[3], x[4], x[5]))
                let dr = simd_float3x3(columns: (Vec3(delta.columns.0.x, delta.columns.0.y, delta.columns.0.z),
                                                 Vec3(delta.columns.1.x, delta.columns.1.y, delta.columns.1.z),
                                                 Vec3(delta.columns.2.x, delta.columns.2.y, delta.columns.2.z)))
                rotation = dr * rotation
                translation = dr * translation + delta.translation
                if stage.identity { for kk in 0..<k { c[kk] += Float(x[6 + kk]) } }
            }
            log(String(format: "stage maxDist %.0f mm identity %@: rms %.2f mm", stage.maxDist * 1000, stage.identity ? "yes" : "no", rms * 1000))
        }

        var w = transformed(model.vertices(c))
        let statistical = w
        if refinementIterations > 0 {
            w = refine(w, grid: grid, scanSamples: scanSamples, skinTriangles: skinTriangles)
        }
        let support: [Float?] = w.map { p in grid.nearest(p, maxDistance: 0.006).map { simd_distance($0.0, p) } }
        return Result(coefficients: c, scale: scale, rotation: rotation, translation: translation,
                      vertices: w, statisticalVertices: statistical, rmsMM: rms * 1000, support: support)
    }
}

extension GNMFitter.Result {
    /// How well the scan supports a side's ear: fraction of GNM ear vertices with scan surface within
    /// 6 mm, and their median distance (metres).
    func earSupport(_ model: GNMModel, side: HeadSide) -> (fraction: Float, median: Float) {
        let idx = (0..<model.vertexCount).filter { model.member("ears", $0) && side.sign * vertices[$0].x > 0 }
        let d = idx.compactMap { support[$0] }
        return (Float(d.count) / Float(max(1, idx.count)), d.median() ?? .infinity)
    }

    /// The fitted head as a skin-only mesh (eyeballs, teeth and tongue removed).
    func skinMesh(_ model: GNMModel) -> TriMesh {
        var m = TriMesh()
        m.vertices = vertices
        m.triangles = model.triangles.filter {
            model.member("skin", Int($0.x)) && model.member("skin", Int($0.y)) && model.member("skin", Int($0.z))
        }
        m.compact()
        m.computeNormals()
        return m
    }

    /// Ear seats from the fitted model: ear extent from GNM's `ears` group, skull surface from the
    /// neighbouring non-ear skin, groove found on the (noise-free) fitted surface.
    func earSeats(_ model: GNMModel, eyeLeft: Vec3, eyeRight: Vec3) -> [HeadSide: EarFit] {
        // Ears come from the statistical fit (the face refinement does not touch them anyway).
        var stat = self
        stat.vertices = statisticalVertices
        return stat.earSeatsOnVertices(model, eyeLeft: eyeLeft, eyeRight: eyeRight)
    }

    private func earSeatsOnVertices(_ model: GNMModel, eyeLeft: Vec3, eyeRight: Vec3) -> [HeadSide: EarFit] {
        let skin = skinMesh(model)
        let analyzer = FitAnalyzer(mesh: skin, eyeCenterLeft: eyeLeft, eyeCenterRight: eyeRight)
        var out: [HeadSide: EarFit] = [:]
        for side in HeadSide.allCases {
            let s = side.sign
            let ear = (0..<model.vertexCount).filter { model.member("ears", $0) && s * vertices[$0].x > 0 }.map { vertices[$0] }
            let head = (0..<model.vertexCount).filter {
                model.member("skin", $0) && !model.member("ears", $0) && s * vertices[$0].x > 0.03
            }.map { vertices[$0] }
            let skullX: (Float, Float) -> Float = { z, y in
                let near = head.filter { abs($0.z - z) < 0.006 && abs($0.y - y) < 0.006 }.map { s * $0.x }
                return near.median() ?? (ear.map { s * $0.x }.min() ?? 0.07)
            }
            if let e = analyzer.modelEarSeat(side: side, earVertices: ear, skullX: skullX) { out[side] = e }
        }
        return out
    }
}

/// Uniform hash grid for nearest-neighbour queries on scan points.
struct PointGrid {
    private var cells: [SIMD3<Int32>: [Int32]] = [:]
    private let points: [Vec3]
    private let normals: [Vec3]
    private let cell: Float

    init(points: [Vec3], normals: [Vec3], cell: Float) {
        self.points = points
        self.normals = normals
        self.cell = cell
        for (i, p) in points.enumerated() {
            cells[SIMD3<Int32>((p / cell).rounded(.down)), default: []].append(Int32(i))
        }
    }

    func nearestIndex(_ p: Vec3, maxDistance: Float) -> Int? {
        let c = SIMD3<Int32>((p / cell).rounded(.down))
        let reach = Int32((maxDistance / cell).rounded(.up))
        var best: Int32 = -1
        var bestD = maxDistance * maxDistance
        for dz in -reach...reach { for dy in -reach...reach { for dx in -reach...reach {
            guard let list = cells[c &+ SIMD3(dx, dy, dz)] else { continue }
            for i in list {
                let d = simd_distance_squared(points[Int(i)], p)
                if d < bestD { bestD = d; best = i }
            }
        }}}
        return best >= 0 ? Int(best) : nil
    }

    func nearest(_ p: Vec3, maxDistance: Float) -> (Vec3, Vec3)? {
        let c = SIMD3<Int32>((p / cell).rounded(.down))
        let reach = Int32((maxDistance / cell).rounded(.up))
        var best: Int32 = -1
        var bestD = maxDistance * maxDistance
        for dz in -reach...reach { for dy in -reach...reach { for dx in -reach...reach {
            guard let list = cells[c &+ SIMD3(dx, dy, dz)] else { continue }
            for i in list {
                let d = simd_distance_squared(points[Int(i)], p)
                if d < bestD { bestD = d; best = i }
            }
        }}}
        return best >= 0 ? (points[Int(best)], normals[Int(best)]) : nil
    }
}

/// GNM's identity prior for one sex and ethnicity class: a diagonal Gaussian estimated from GNM's
/// semantic sampler (Tools/gnm_export_priors.py), the prior GNM's photo fitting demo uses.
struct GNMIdentityPrior: Codable, Equatable {
    var sex: String
    var ethnicity: String
    var mean: [Float]
    var std: [Float]

    static let sexes = ["female", "male"]
    static let ethnicities = ["asian", "black", "middle_eastern", "white"]
    static let sexKey = "priorSex"
    static let ethnicityKey = "priorEthnicity"

    var label: String { "\(sex), \(ethnicity.replacingOccurrences(of: "_", with: " "))" }

    static func load(sex: String, ethnicity: String, from url: URL? = nil) -> GNMIdentityPrior? {
        struct File: Decodable { var classes: [GNMIdentityPrior] }
        let dir = GNMModel.resourceDirectoryOverride
        guard let url = url ?? dir.map({ $0.appendingPathComponent("gnm_identity_priors.json") })
                ?? Bundle.main.url(forResource: "gnm_identity_priors", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let file = try? JSONDecoder().decode(File.self, from: data) else { return nil }
        return file.classes.first { $0.sex == sex && $0.ethnicity == ethnicity }
    }

    /// The class chosen on the home screen, if both sex and ethnicity are set.
    static var selected: GNMIdentityPrior? {
        let d = UserDefaults.standard
        guard let s = d.string(forKey: sexKey), let e = d.string(forKey: ethnicityKey), !s.isEmpty, !e.isEmpty else { return nil }
        return load(sex: s, ethnicity: e)
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
