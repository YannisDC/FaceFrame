import Accelerate
import Foundation
import simd

/// GNM as the photo fit uses it: neutral pose, where the model is linear,
/// vertices = template + identity basis · identity + expression basis · expression (metres).
///
/// Loaded from Resources/GNMFrames: GNMFrames' model files plus gnm_expression_basis.bin and
/// gnm_photo_fit.json (Tools/gnm_export_photo_fit_data.py).
final class GNMPhotoModel {
    let vertexCount: Int
    /// Flat [x0, y0, z0, x1, …].
    let template: [Float]
    /// Component-major: identityBasis[k * 3V + 3i + axis].
    let identityBasis: [Float]
    let identityCount: Int
    /// Sparse expression blend shapes on the skin and landmark vertices: vertex indices and xyz offsets.
    let expression: [(vertices: [Int32], offsets: [Float])]
    /// GNM's 68 sparse face landmarks (iBUG order): three (vertex, weight) each.
    let landmarks: [[(vertex: Int, weight: Float)]]
    /// Points on the skin that may lie on an outline: the vertices and edge midpoints of the skin triangles.
    let outlinePoints: [(Int32, Int32)]
    /// Edges along the outer rim of each ear.
    let earRim: [HeadSide: [(Int32, Int32)]]
    let earVertices: [HeadSide: [Int]]
    let groups: [String: [Bool]]
    let triangles: [SIMD3<UInt32>]

    init(directory: URL) throws {
        func data(_ name: String) throws -> Data { try Data(contentsOf: directory.appendingPathComponent(name)) }
        struct Meta: Decodable { var vertices: Int; var groups: [String]; var landmarks68: [[Double]] }
        let meta = try JSONDecoder().decode(Meta.self, from: data("meta.json"))
        let n = meta.vertices
        vertexCount = n
        template = try data("template.bin").withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let basis = try data("identity_basis.bin")
        let count = basis.count / (n * 6)
        identityCount = count
        identityBasis = basis.withUnsafeBytes { raw in
            let h = raw.bindMemory(to: Float16.self)
            return (0..<(count * n * 3)).map { Float(h[$0]) }
        }
        let tri = try data("triangles.bin").withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        triangles = (0..<(tri.count / 3)).map { SIMD3(tri[$0 * 3], tri[$0 * 3 + 1], tri[$0 * 3 + 2]) }
        let packed = [UInt8](try data("groups.bin"))
        var groups: [String: [Bool]] = [:]
        for (gi, name) in meta.groups.enumerated() where (gi + 1) * n <= packed.count {
            groups[name] = packed[(gi * n)..<((gi + 1) * n)].map { $0 != 0 }
        }
        self.groups = groups
        landmarks = meta.landmarks68.map { row in (0..<3).map { (Int(row[2 * $0]), Float(row[2 * $0 + 1])) } }

        // Expression blend shapes.
        let blob = try data("gnm_expression_basis.bin")
        var expression: [(vertices: [Int32], offsets: [Float])] = []
        blob.withUnsafeBytes { raw in
            var o = 0
            func align() { o = (o + 3) & ~3 }
            while o + 4 <= raw.count {
                let count = Int(raw.loadUnaligned(fromByteOffset: o, as: UInt32.self)); o += 4
                let idx = (0..<count).map { Int32(raw.loadUnaligned(fromByteOffset: o + 2 * $0, as: UInt16.self)) }
                o += 2 * count; align()
                let val = (0..<(3 * count)).map { Float(raw.loadUnaligned(fromByteOffset: o + 2 * $0, as: Float16.self)) }
                o += 6 * count; align()
                expression.append((idx, val))
            }
        }
        self.expression = expression

        struct Extra: Decodable { var earRimEdges: [String: [[Int32]]] }
        let extra = try JSONDecoder().decode(Extra.self, from: data("gnm_photo_fit.json"))
        earRim = [.left: extra.earRimEdges["left"]!.map { ($0[0], $0[1]) }, .right: extra.earRimEdges["right"]!.map { ($0[0], $0[1]) }]

        let skin = groups["skin"] ?? []
        let ears = groups["ears"] ?? []
        let tmpl = template
        let tx = { (i: Int) in tmpl[3 * i] }
        earVertices = [.left: (0..<n).filter { ears[$0] && tx($0) > 0 }, .right: (0..<n).filter { ears[$0] && tx($0) < 0 }]
        // As gnm.triangles_group('skin'): triangles whose vertices are all skin.
        let skinTriangles = triangles.filter { skin[Int($0.x)] && skin[Int($0.y)] && skin[Int($0.z)] }
        var verts = Set<Int32>(), edges = Set<SIMD2<Int32>>()
        for t in skinTriangles {
            let v = [Int32(t.x), Int32(t.y), Int32(t.z)]
            for k in 0..<3 {
                verts.insert(v[k])
                let a = v[k], b = v[(k + 1) % 3]
                edges.insert(SIMD2(min(a, b), max(a, b)))
            }
        }
        outlinePoints = verts.sorted().map { ($0, $0) } + edges.sorted { ($0.x, $0.y) < ($1.x, $1.y) }.map { ($0.x, $0.y) }
    }

    func vertices(identity: [Float], expression e: [Float]) -> [Float] {
        var v = template
        let k = min(identity.count, identityCount)
        if k > 0 {
            identityBasis.withUnsafeBufferPointer { b in
                cblas_sgemv(CblasRowMajor, CblasTrans, Int32(k), Int32(3 * vertexCount), 1, b.baseAddress, Int32(3 * vertexCount),
                            identity, 1, 1, &v, 1)
            }
        }
        for (c, w) in e.enumerated() where w != 0 && c < expression.count {
            let comp = expression[c]
            for (j, vi) in comp.vertices.enumerated() {
                let o = 3 * Int(vi)
                v[o] += w * comp.offsets[3 * j]; v[o + 1] += w * comp.offsets[3 * j + 1]; v[o + 2] += w * comp.offsets[3 * j + 2]
            }
        }
        return v
    }

    func landmarkPositions(_ v: [Float]) -> [SIMD3<Double>] {
        landmarks.map { row in
            row.reduce(SIMD3<Double>.zero) { acc, e in
                acc + Double(e.weight) * SIMD3(Double(v[3 * e.vertex]), Double(v[3 * e.vertex + 1]), Double(v[3 * e.vertex + 2]))
            }
        }
    }
}

/// One photo of the person, as GNM's photo fit (demos/fit_head_to_images.py) observes it.
struct PhotoObservation {
    enum OutlineKind: String { case profile, hairBack, hairTop, hairSide }
    /// The outermost point of the head in a direction, per image row (or column if `alongColumns`), NaN where none.
    struct Outline {
        var kind: OutlineKind
        var values: [Double]
        /// +1 if the outline faces increasing coordinates.
        var direction: Double
        var alongColumns = false
    }

    /// Upright image size (pixels).
    var width: Int
    var height: Int
    /// OpenCV-convention intrinsics of the upright image: fx, fy, cx, cy.
    var fx: Double, fy: Double, cx: Double, cy: Double
    /// The 68 iBUG landmarks (pixels).
    var landmarks: [SIMD2<Double>]
    var outlines: [Outline] = []
    /// The outer helix of the visible ear (15 points, top to lobe), side views only.
    var earHelix: [SIMD2<Double>]?
    /// Initial head pose: GNM world (metres) to OpenCV camera (x right, y down, looking along +z).
    var initialRotation: simd_double3x3
    var initialTranslation: SIMD3<Double>
}

/// Ports `fit()` of GNM's demos/fit_head_to_images.py: one GNM identity and expression shared by all
/// photos, and one head pose per photo, fitted with Adam to the 68 landmarks, the profile and hair
/// outlines and the ear helix, under a Gaussian prior. Same losses, weights and schedule.
struct PhotoFitter {
    struct Result {
        var identity: [Float]
        var expression: [Float]
        var rotations: [simd_double3x3]
        var translations: [SIMD3<Double>]
        var losses: (landmarks: Double, outline: Double, ears: Double, prior: Double)
        /// Mean landmark error per view, in pixels.
        var landmarkErrorPixels: [Double]
        /// With anchors: the scale of the similarity that maps the head onto them, and each anchor's error (mm).
        var anchorScale: Double?
        var anchorErrorsMM: [Double] = []
    }

    /// A point measured on the scan (metres, scan frame) that the centroid of some model vertices must
    /// reach, through a similarity (rotation, translation, uniform scale) solved with the rest: the
    /// photos keep deciding the shape, and the anchors add the metric relations between them that
    /// photos alone get wrong (on a real capture: a 5 mm short PD and ears 10 mm too far in).
    struct Anchor {
        var name: String
        var vertices: [Int]
        var target: SIMD3<Double>
    }
    var anchors: [Anchor] = []
    /// Weight of the anchors: a 1 mm error costs as much as a landmark off by 1% of the face size.
    var anchorWeight = 1.0

    let model: GNMPhotoModel
    var prior: GNMIdentityPrior?
    var rigidSteps = 300
    var steps = 1000
    var identityRegularization = 3e-2
    var expressionRegularization = 1e-1
    var outlineWeight = 1.0
    var hairWeight = 1.0
    var hairThicknessMM = 10.0
    var earWeight = 1.0

    static let jawWeight = 0.3
    static let occludedWeight = 0.05
    static let occlusionYawDegrees = 35.0
    static let outlineBins = 60
    static let hairOutlineBins = 40
    static let outlierThreshold = 10.0

    struct OutlineTarget {
        var view: Int
        var weight: Double
        var direction: Double
        var alongColumns: Bool
        var start: Double
        var binSize: Double
        var targets: [Double]
        var valid: [Bool]
    }

    static func isSideView(_ r: simd_double3x3) -> Bool {
        // Depth of the world X axis (towards the subject's left) in the camera: r[2][0] in row-major.
        abs(r.columns.0.z) > sin(occlusionYawDegrees * .pi / 180)
    }

    static func makeOutlineTarget(view: Int, _ outline: PhotoObservation.Outline, weight: Double, bins: Int,
                                  offset: Double = 0) -> OutlineTarget? {
        let lines = outline.values.indices.filter { outline.values[$0].isFinite }
        guard lines.count >= bins, let first = lines.first, let lastLine = lines.last else { return nil }
        let last = lastLine + 1
        let binSize = Double(last - first) / Double(bins)
        var targets = [Double](repeating: 0, count: bins), valid = [Bool](repeating: false, count: bins)
        for i in 0..<bins {
            let s = Int(Double(first) + Double(i) * binSize), e = Int(Double(first) + Double(i + 1) * binSize)
            let values = Array(outline.values[s..<max(s, e)])
            let finite = values.filter { $0.isFinite }
            if !finite.isEmpty && Double(finite.count) >= 0.5 * Double(values.count) {
                targets[i] = outline.direction * (finite.map { outline.direction * $0 }.max()! - offset)
                valid[i] = true
            }
        }
        return OutlineTarget(view: view, weight: weight, direction: outline.direction, alongColumns: outline.alongColumns,
                             start: Double(first), binSize: binSize, targets: targets, valid: valid)
    }

    func fit(_ views: [PhotoObservation], log: (String) -> Void = { _ in }) -> Result {
        let n = model.vertexCount
        let k = model.identityCount
        let e = model.expression.count
        let nv = views.count

        let templateLandmarks = model.landmarkPositions(model.template)
        let sideSigns = templateLandmarks.map { abs($0.x) < 0.01 ? 0.0 : ($0.x > 0 ? 1.0 : -1.0) }
        var rot = views.map { axisAngle($0.initialRotation) }
        var trans = views.map { $0.initialTranslation }
        let mean = (0..<k).map { Double(prior?.mean[safe: $0] ?? 0) }
        let std = (0..<k).map { Double(prior?.std[safe: $0] ?? 1) }
        var identity = mean
        var expression = [Double](repeating: 0, count: e)

        let faceSizes = views.map { v -> Double in
            let xs = v.landmarks.map(\.x), ys = v.landmarks.map(\.y)
            return hypot(xs.max()! - xs.min()!, ys.max()! - ys.min()!)
        }
        let weights: [[Double]] = views.map { v in
            let r = v.initialRotation
            var w = [Double](repeating: 1, count: 68)
            let away = r.columns.0.z
            if abs(away) > sin(Self.occlusionYawDegrees * .pi / 180) {
                for j in 0..<68 where sideSigns[j] == (away > 0 ? 1 : -1) { w[j] = Self.occludedWeight }
                for j in 0..<17 { w[j] = 0 }
            } else {
                for j in 0..<17 { w[j] = Self.jawWeight }
            }
            return w
        }
        let weightSum = weights.joined().reduce(0, +)

        // Outline targets, decided with the initial poses.
        var outlineTargets: [OutlineTarget] = []
        for (vi, v) in views.enumerated() {
            let side = Self.isSideView(v.initialRotation)
            let hair = hairThicknessMM / 1000 * v.fx / v.initialTranslation.z
            for o in v.outlines {
                var t: OutlineTarget?
                switch o.kind {
                case .profile where side: t = Self.makeOutlineTarget(view: vi, o, weight: outlineWeight, bins: Self.outlineBins)
                case .hairBack where side: t = Self.makeOutlineTarget(view: vi, o, weight: hairWeight, bins: Self.hairOutlineBins, offset: hair)
                case .hairSide where !side: t = Self.makeOutlineTarget(view: vi, o, weight: hairWeight, bins: Self.hairOutlineBins, offset: hair)
                case .hairTop: t = Self.makeOutlineTarget(view: vi, o, weight: hairWeight, bins: Self.hairOutlineBins, offset: hair)
                default: t = nil
                }
                if let t, t.weight > 0 { outlineTargets.append(t) }
            }
        }
        // Ear helix targets: the rim of the visible ear in each side view.
        var earTargets: [(view: Int, edges: [(Int32, Int32)], helix: [SIMD2<Double>])] = []
        if earWeight > 0 {
            for (vi, v) in views.enumerated() {
                guard let helix = v.earHelix, Self.isSideView(v.initialRotation) else { continue }
                let side = visibleEar(model.template, v.initialRotation, v.initialTranslation)
                earTargets.append((vi, model.earRim[side]!, helix))
            }
        }
        log("photo fit: \(nv) views, \(outlineTargets.count) outlines, \(earTargets.count) ears")

        // MARK: Losses and gradients

        /// Accumulates d loss / d camera point into the pose gradient and returns d loss / d world point.
        struct PoseGrad { var g = simd_double3x3(0), t = SIMD3<Double>.zero }

        func project(_ p: SIMD3<Double>, _ r: simd_double3x3, _ t: SIMD3<Double>, _ v: PhotoObservation) -> (SIMD2<Double>, SIMD3<Double>) {
            let c = r * p + t
            return (SIMD2(v.fx * c.x / c.z + v.cx, v.fy * c.y / c.z + v.cy), c)
        }
        /// d loss / d camera point from d loss / d pixel.
        func backCamera(_ c: SIMD3<Double>, _ g: SIMD2<Double>, _ v: PhotoObservation) -> SIMD3<Double> {
            SIMD3(g.x * v.fx / c.z, g.y * v.fy / c.z, -(g.x * v.fx * c.x + g.y * v.fy * c.y) / (c.z * c.z))
        }
        func huber(_ x: Double, _ d: Double = 2) -> (Double, Double) {
            abs(x) < d ? (0.5 * x * x, x) : (d * (abs(x) - 0.5 * d), d * (x > 0 ? 1 : -1))
        }
        /// GNM's robust_mean: Huber mean of residuals (percent of face size) below the outlier threshold.
        func robustMean(_ r: [Double]) -> (Double, [Double]) {
            let kept = r.indices.filter { abs(r[$0]) < Self.outlierThreshold }
            guard !kept.isEmpty else { return (0, r.map { _ in 0 }) }
            var loss = 0.0, grad = [Double](repeating: 0, count: r.count)
            for i in kept {
                let (l, g) = huber(r[i])
                loss += l; grad[i] = g / Double(kept.count)
            }
            return (loss / Double(kept.count), grad)
        }

        // Similarity from the head (GNM frame) to the scan, for the anchors: scan = s R head + t.
        var simR = SIMD3<Double>.zero, simT = SIMD3<Double>.zero, simLogS = 0.0
        var simGrad = (r: simd_double3x3(0), t: SIMD3<Double>.zero, logS: 0.0)
        var anchorErrors: [Double] = []

        struct Evaluation {
            var loss = 0.0, landmarks = 0.0, outline = 0.0, ears = 0.0, prior = 0.0, anchors = 0.0
            var pose: [PoseGrad]
            var vertexGrad: [Double]
            var pixelErrors: [Double]
        }

        func evaluate(identity: [Double], expression: [Double], rigidOnly: Bool) -> Evaluation {
            var ev = Evaluation(pose: Array(repeating: PoseGrad(), count: nv), vertexGrad: [], pixelErrors: Array(repeating: 0, count: nv))
            let verts: [Float] = rigidOnly ? model.template
                : model.vertices(identity: identity.map(Float.init), expression: expression.map(Float.init))
            var vg = [Double](repeating: 0, count: rigidOnly ? 0 : 3 * n)
            let rs = rot.map(rotationMatrix)
            func point(_ i: Int) -> SIMD3<Double> { SIMD3(Double(verts[3 * i]), Double(verts[3 * i + 1]), Double(verts[3 * i + 2])) }
            func addVertex(_ i: Int, _ g: SIMD3<Double>, _ w: Double = 1) {
                guard !rigidOnly else { return }
                vg[3 * i] += w * g.x; vg[3 * i + 1] += w * g.y; vg[3 * i + 2] += w * g.z
            }
            /// Back-propagates d loss / d pixel of a world point p seen in view vi.
            func back(_ vi: Int, _ p: SIMD3<Double>, _ c: SIMD3<Double>, _ gPix: SIMD2<Double>) -> SIMD3<Double> {
                let gc = backCamera(c, gPix, views[vi])
                ev.pose[vi].t += gc
                ev.pose[vi].g += simd_double3x3(rows: [gc * p.x, gc * p.y, gc * p.z]).transpose // outer(gc, p)
                return rs[vi].transpose * gc
            }

            // Landmarks.
            let lms = rigidOnly ? templateLandmarks : model.landmarkPositions(verts)
            var landmarkLoss = 0.0
            for vi in 0..<nv {
                let v = views[vi]
                var errSum = 0.0, wSum = 0.0
                for j in 0..<68 {
                    let w = weights[vi][j]
                    let (px, c) = project(lms[j], rs[vi], trans[vi], v)
                    errSum += w * simd_length(px - v.landmarks[j]); wSum += w
                    guard w > 0 else { continue }
                    let res = (px - v.landmarks[j]) / faceSizes[vi] * 100
                    let hx = huber(res.x), hy = huber(res.y)
                    landmarkLoss += w * (hx.0 + hy.0)
                    let gPix = SIMD2(hx.1, hy.1) * (w * 100 / faceSizes[vi] / weightSum)
                    let gp = back(vi, lms[j], c, gPix)
                    if !rigidOnly { for (vtx, bw) in model.landmarks[j] { addVertex(vtx, gp, Double(bw)) } }
                }
                ev.pixelErrors[vi] = wSum > 0 ? errSum / wSum : 0
            }
            ev.landmarks = landmarkLoss / weightSum

            if !rigidOnly {
                // Outlines: the outermost projected skin point in each bin.
                let pts = model.outlinePoints.map { (point(Int($0.0)) + point(Int($0.1))) * 0.5 }
                var projected: [Int: [(SIMD2<Double>, SIMD3<Double>)]] = [:]
                for vi in Set(outlineTargets.map(\.view)) { projected[vi] = pts.map { project($0, rs[vi], trans[vi], views[vi]) } }
                var outlineLosses = 0.0
                for t in outlineTargets {
                    let proj = projected[t.view]!
                    let bins = t.targets.count
                    var best = [Double](repeating: -.infinity, count: bins), arg = [Int](repeating: -1, count: bins)
                    for (pi, (px, _)) in proj.enumerated() {
                        let along = t.alongColumns ? px.x : px.y, across = t.alongColumns ? px.y : px.x
                        let b = Int(floor((along - t.start) / t.binSize))
                        guard b >= 0, b < bins else { continue }
                        if t.direction * across > best[b] { best[b] = t.direction * across; arg[b] = pi }
                    }
                    let valid = (0..<bins).filter { t.valid[$0] && arg[$0] >= 0 }
                    let res = valid.map { 100 * (t.direction * best[$0] - t.targets[$0]) / faceSizes[t.view] }
                    let (l, g) = robustMean(res)
                    outlineLosses += t.weight * l
                    for (ri, b) in valid.enumerated() where g[ri] != 0 {
                        let scaleG = t.weight / Double(outlineTargets.count) * g[ri] * 100 / faceSizes[t.view]
                        let gPix = t.alongColumns ? SIMD2(0, scaleG) : SIMD2(scaleG, 0)
                        let (a, bIdx) = model.outlinePoints[arg[b]]
                        let gp = back(t.view, pts[arg[b]], proj[arg[b]].1, gPix)
                        addVertex(Int(a), gp, 0.5); addVertex(Int(bIdx), gp, 0.5)
                    }
                }
                ev.outline = outlineTargets.isEmpty ? 0 : outlineLosses / Double(outlineTargets.count)

                // Ears: each helix point to the closest segment of the projected rim.
                var earLosses = 0.0
                for et in earTargets {
                    let v = views[et.view]
                    let ends = et.edges.map { (project(point(Int($0.0)), rs[et.view], trans[et.view], v),
                                                project(point(Int($0.1)), rs[et.view], trans[et.view], v)) }
                    var dists: [Double] = [], hits: [(seg: Int, t: Double, u: SIMD2<Double>)] = []
                    for h in et.helix {
                        var bestD = Double.infinity, bestHit = (0, 0.0, SIMD2<Double>.zero)
                        for (si, (a, b)) in ends.enumerated() {
                            let d = b.0 - a.0
                            let tt = max(0, min(1, simd_dot(h - a.0, d) / max(simd_length_squared(d), 1e-12)))
                            let c = a.0 + tt * d
                            let dist = simd_length(h - c)
                            if dist < bestD { bestD = dist; bestHit = (si, tt, dist > 0 ? (h - c) / dist : .zero) }
                        }
                        dists.append(100 * bestD / faceSizes[et.view]); hits.append(bestHit)
                    }
                    let (l, g) = robustMean(dists)
                    earLosses += l
                    for (hi, hit) in hits.enumerated() where g[hi] != 0 {
                        // d dist / d closest = −u; closest = a + t (b − a).
                        let gc = -hit.u * (earWeight / Double(earTargets.count) * g[hi] * 100 / faceSizes[et.view])
                        let (ia, ib) = et.edges[hit.seg]
                        let (pa, pb) = ends[hit.seg]
                        let ga = back(et.view, point(Int(ia)), pa.1, gc * (1 - hit.t))
                        let gb = back(et.view, point(Int(ib)), pb.1, gc * hit.t)
                        addVertex(Int(ia), ga); addVertex(Int(ib), gb)
                    }
                }
                ev.ears = earTargets.isEmpty ? 0 : earWeight * earLosses / Double(earTargets.count)

                var priorLoss = 0.0
                for c in 0..<k { let z = (identity[c] - mean[c]) / std[c]; priorLoss += identityRegularization * z * z }
                for c in 0..<e { priorLoss += expressionRegularization * expression[c] * expression[c] }
                ev.prior = priorLoss

                // Anchors: Huber on each coordinate's error in mm, averaged.
                simGrad = (simd_double3x3(0), .zero, 0)
                anchorErrors = []
                if !anchors.isEmpty {
                    let sr = rotationMatrix(simR), sc = exp(simLogS)
                    let count = Double(3 * anchors.count)
                    for a in anchors {
                        let pa = a.vertices.reduce(SIMD3<Double>.zero) { $0 + point($1) } / Double(a.vertices.count)
                        let q = sc * (sr * pa) + simT
                        let res = (q - a.target) * 1000
                        anchorErrors.append(simd_length(res))
                        var gq = SIMD3<Double>.zero
                        for axis in 0..<3 {
                            let (l, g) = huber(res[axis])
                            ev.anchors += anchorWeight * l / count
                            gq[axis] = anchorWeight * g / count * 1000
                        }
                        simGrad.t += gq
                        simGrad.logS += simd_dot(gq, sc * (sr * pa))
                        simGrad.r += simd_double3x3(rows: [gq * (sc * pa.x), gq * (sc * pa.y), gq * (sc * pa.z)]).transpose
                        let gp = sc * (sr.transpose * gq)
                        for vtx in a.vertices { addVertex(vtx, gp, 1 / Double(a.vertices.count)) }
                    }
                }
            }
            ev.vertexGrad = vg
            ev.loss = ev.landmarks + ev.outline + ev.ears + ev.prior + ev.anchors
            return ev
        }

        /// d loss / d axis-angle from the accumulated outer product G = Σ gc ⊗ p: Σ G ⊙ dR/dr.
        func rotationGrad(_ r: SIMD3<Double>, _ g: simd_double3x3) -> SIMD3<Double> {
            var out = SIMD3<Double>.zero
            let h = 1e-6
            for a in 0..<3 {
                var rp = r, rm = r
                rp[a] += h; rm[a] -= h
                let d = (rotationMatrix(rp) - rotationMatrix(rm)) * (1 / (2 * h))
                var s = 0.0
                for c in 0..<3 { s += simd_dot(d[c], g[c]) }
                out[a] = s
            }
            return out
        }

        // MARK: Adam
        struct Adam {
            var m: [Double], v: [Double], t = 0
            let b1 = 0.9, b2 = 0.999, eps = 1e-8
            init(_ count: Int) { m = Array(repeating: 0, count: count); v = Array(repeating: 0, count: count) }
            mutating func step(_ x: inout [Double], _ g: [Double], lr: [Double]) {
                t += 1
                let c1 = 1 - pow(b1, Double(t)), c2 = 1 - pow(b2, Double(t))
                for i in x.indices {
                    m[i] = b1 * m[i] + (1 - b1) * g[i]
                    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
                    x[i] -= lr[i] * (m[i] / c1) / ((v[i] / c2).squareRoot() + eps)
                }
            }
        }
        func poseVector() -> [Double] { (0..<nv).flatMap { [rot[$0].x, rot[$0].y, rot[$0].z, trans[$0].x, trans[$0].y, trans[$0].z] } }
        func setPose(_ x: [Double]) {
            for i in 0..<nv {
                rot[i] = SIMD3(x[6 * i], x[6 * i + 1], x[6 * i + 2])
                trans[i] = SIMD3(x[6 * i + 3], x[6 * i + 4], x[6 * i + 5])
            }
        }
        func poseGradient(_ ev: Evaluation) -> [Double] {
            (0..<nv).flatMap { i -> [Double] in
                let gr = rotationGrad(rot[i], ev.pose[i].g)
                return [gr.x, gr.y, gr.z, ev.pose[i].t.x, ev.pose[i].t.y, ev.pose[i].t.z]
            }
        }

        // Stage 1: rigid alignment of the template.
        var adam = Adam(6 * nv)
        var x = poseVector()
        for _ in 0..<rigidSteps {
            let ev = evaluate(identity: identity, expression: expression, rigidOnly: true)
            adam.step(&x, poseGradient(ev), lr: Array(repeating: 1e-2, count: x.count))
            setPose(x)
        }
        log(String(format: "photo fit: rigid stage, landmarks %.4f", evaluate(identity: identity, expression: expression, rigidOnly: true).landmarks))

        // Stage 2: all parameters, cosine-annealed learning rates. The anchors' similarity starts from the
        // template: same axes (GNM's and the scan's head frames are both ARKit's face axes), matched spread and centroid.
        if !anchors.isEmpty {
            let pts = anchors.map { a in a.vertices.reduce(SIMD3<Double>.zero) { $0 + SIMD3(Double(model.template[3 * $1]), Double(model.template[3 * $1 + 1]),
                                                                                            Double(model.template[3 * $1 + 2])) } / Double(a.vertices.count) }
            let ma = pts.reduce(.zero, +) / Double(pts.count), mt = anchors.map(\.target).reduce(.zero, +) / Double(anchors.count)
            let sa = pts.map { simd_length_squared($0 - ma) }.reduce(0, +), st = anchors.map { simd_length_squared($0.target - mt) }.reduce(0, +)
            simLogS = 0.5 * Foundation.log(st / max(sa, 1e-12))
            simT = mt - exp(simLogS) * ma
        }
        adam = Adam(6 * nv + k + e + 7)
        x = poseVector() + identity + expression + [simR.x, simR.y, simR.z, simT.x, simT.y, simT.z, simLogS]
        var last: Evaluation?
        for step in 0..<steps {
            let ev = evaluate(identity: identity, expression: expression, rigidOnly: false)
            last = ev
            // d loss / d identity = B · d loss / d vertices (+ prior); same for the expression.
            var gi = [Float](repeating: 0, count: model.identityCount)
            let vgf = ev.vertexGrad.map(Float.init)
            model.identityBasis.withUnsafeBufferPointer { b in
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(model.identityCount), Int32(3 * n), 1, b.baseAddress, Int32(3 * n),
                            vgf, 1, 0, &gi, 1)
            }
            var g = poseGradient(ev)
            for c in 0..<k { g.append(Double(gi[c]) + 2 * identityRegularization * (identity[c] - mean[c]) / (std[c] * std[c])) }
            for c in 0..<e {
                let comp = model.expression[c]
                var s = 0.0
                for (j, vi) in comp.vertices.enumerated() {
                    let o = 3 * Int(vi)
                    s += Double(comp.offsets[3 * j]) * ev.vertexGrad[o] + Double(comp.offsets[3 * j + 1]) * ev.vertexGrad[o + 1]
                        + Double(comp.offsets[3 * j + 2]) * ev.vertexGrad[o + 2]
                }
                g.append(s + 2 * expressionRegularization * expression[c])
            }
            let srg = anchors.isEmpty ? .zero : rotationGrad(simR, simGrad.r)
            g += [srg.x, srg.y, srg.z, simGrad.t.x, simGrad.t.y, simGrad.t.z, simGrad.logS]
            let anneal = (1 + cos(Double.pi * Double(step) / Double(steps))) / 2
            let lr = Array(repeating: 2e-3 * anneal, count: 6 * nv) + Array(repeating: 2e-2 * anneal, count: k + e)
                + Array(repeating: 2e-3 * anneal, count: 7)
            adam.step(&x, g, lr: lr)
            setPose(Array(x[0..<(6 * nv)]))
            identity = Array(x[(6 * nv)..<(6 * nv + k)])
            expression = Array(x[(6 * nv + k)..<(6 * nv + k + e)])
            let o = 6 * nv + k + e
            simR = SIMD3(x[o], x[o + 1], x[o + 2]); simT = SIMD3(x[o + 3], x[o + 4], x[o + 5]); simLogS = x[o + 6]
            if step % 250 == 0 || step == steps - 1 {
                log(String(format: "photo fit: step %4d  landmarks %.4f  outline %.4f  ears %.4f  prior %.4f  anchors %.4f",
                           step, ev.landmarks, ev.outline, ev.ears, ev.prior, ev.anchors))
            }
        }
        let final = evaluate(identity: identity, expression: expression, rigidOnly: false)
        _ = last
        if !anchors.isEmpty {
            log("photo fit: anchors " + zip(anchors, anchorErrors).map { String(format: "%@ %.1f mm", $0.name, $1) }.joined(separator: ", ")
                + String(format: ", similarity scale %.3f", exp(simLogS)))
        }
        return Result(identity: identity.map(Float.init), expression: expression.map(Float.init),
                      rotations: rot.map(rotationMatrix), translations: trans,
                      losses: (final.landmarks, final.outline, final.ears, final.prior), landmarkErrorPixels: final.pixelErrors,
                      anchorScale: anchors.isEmpty ? nil : exp(simLogS), anchorErrorsMM: anchorErrors)
    }

    /// The ear closest to the camera.
    func visibleEar(_ v: [Float], _ r: simd_double3x3, _ t: SIMD3<Double>) -> HeadSide {
        func depth(_ side: HeadSide) -> Double {
            let idx = model.earVertices[side]!
            return idx.reduce(0.0) { acc, i in
                acc + (r * SIMD3(Double(v[3 * i]), Double(v[3 * i + 1]), Double(v[3 * i + 2])) + t).z
            } / Double(idx.count)
        }
        return depth(.left) < depth(.right) ? .left : .right
    }
}

func rotationMatrix(_ r: SIMD3<Double>) -> simd_double3x3 {
    let angle = max(simd_length(r), 1e-12)
    let a = r / angle
    let k = simd_double3x3(rows: [SIMD3(0, -a.z, a.y), SIMD3(a.z, 0, -a.x), SIMD3(-a.y, a.x, 0)])
    return matrix_identity_double3x3 + sin(angle) * k + (1 - cos(angle)) * (k * k)
}

func axisAngle(_ m: simd_double3x3) -> SIMD3<Double> {
    let q = simd_quatd(m)
    let angle = q.angle
    return angle < 1e-12 ? .zero : q.axis * angle
}
