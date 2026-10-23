import Foundation
import simd

enum HeadSide: String, Codable, CaseIterable {
    /// The subject's own left (+x in the ARKit face frame).
    case left
    case right
    var sign: Float { self == .left ? 1 : -1 }
}

/// Landmarks the user can re-place by tapping the model.
enum AdjustableLandmark: String, CaseIterable, Identifiable {
    case crest, earLeft, earRight
    var id: String { rawValue }
    var title: String {
        switch self {
        case .crest: return "Bridge crest"
        case .earLeft: return "Left ear seat"
        case .earRight: return "Right ear seat"
        }
    }
}

struct FitOverrides {
    var crest: Vec3?
    var earLeft: Vec3?
    var earRight: Vec3?

    subscript(_ l: AdjustableLandmark) -> Vec3? {
        get {
            switch l {
            case .crest: return crest
            case .earLeft: return earLeft
            case .earRight: return earRight
            }
        }
        set {
            switch l {
            case .crest: crest = newValue
            case .earLeft: earLeft = newValue
            case .earRight: earRight = newValue
            }
        }
    }
}

/// One horizontal cross-section of the nose, `levelBelowCrest` metres under the bridge crest.
struct NoseSlice {
    var levelBelowCrest: Float
    var ridge: Vec3
    /// Half-widths (left, right) at depths 1...15 mm behind the ridge; nil where the nose merges into the face.
    var halfWidths: [(depth: Float, left: Float?, right: Float?)]
    /// Width of the nose where it crosses the frame back plane (DBR at this level).
    var backPlaneWidth: Float?
    /// Nose side angle from the sagittal plane seen from above (pad splay angle), degrees.
    var splayLeft: Float?
    var splayRight: Float?
}

struct EarFit: Codable {
    var side: HeadSide
    var seat: Vec3
    /// How the seat was found: "crease" (valley found in the mesh), "outline" (side-view outline only) or "manual".
    var method: String
    var earTop: Vec3?
    var earBottom: Vec3?
    var earFront: Vec3?
    var earBack: Vec3?
    /// Ear protrusion height map outline for visualisation (head-frame points along the ear boundary).
    var outline: [Vec3]
}

struct FitResult {
    var pupilLeft: Vec3
    var pupilRight: Vec3
    var eyeCenterLeft: Vec3
    var eyeCenterRight: Vec3
    var corneaZ: Float
    var midlineX: Float

    var sellion: Vec3
    var noseTip: Vec3
    var crest: Vec3
    var crestIsManual: Bool
    /// The frame back plane is modelled vertical, perpendicular to z, resting on the crest.
    var backPlaneZ: Float
    var slices: [NoseSlice]
    var padContactLeft: [Vec3]
    var padContactRight: [Vec3]
    var frontalAngleLeft: Float?
    var frontalAngleRight: Float?

    var earLeft: EarFit?
    var earRight: EarFit?
    var warnings: [String]
    /// PD measured from the pupils in the colour images (distance PD), when available.
    var measuredPD: Float? = nil
    /// Share of the PD on the subject's left, measured in the images.
    var measuredLeftShare: Float? = nil
    var pupilsFromImages: Bool { measuredPD != nil }

    // Derived measurements (metres / degrees)
    var pd: Float { measuredPD ?? simd_distance(eyeCenterLeft, eyeCenterRight) }
    var monoPDLeft: Float { measuredLeftShare.map { $0 * pd } ?? abs(eyeCenterLeft.x - midlineX) }
    var monoPDRight: Float { measuredLeftShare.map { (1 - $0) * pd } ?? abs(eyeCenterRight.x - midlineX) }
    var vertexDistance: Float { backPlaneZ - corneaZ }
    var pupilHeightAboveCrest: Float { (pupilLeft.y + pupilRight.y) / 2 - crest.y }
    var sellionAbovePupils: Float { sellion.y - (pupilLeft.y + pupilRight.y) / 2 }

    func slice(at level: Float) -> NoseSlice? {
        slices.min(by: { abs($0.levelBelowCrest - level) < abs($1.levelBelowCrest - level) })
            .flatMap { abs($0.levelBelowCrest - level) < 0.0006 ? $0 : nil }
    }

    func dbr(at level: Float) -> Float? { slice(at: level)?.backPlaneWidth }

    func splay(at level: Float) -> Float? {
        guard let s = slice(at: level) else { return nil }
        let v = [s.splayLeft, s.splayRight].compactMap { $0 }
        return v.isEmpty ? nil : v.reduce(0, +) / Float(v.count)
    }

    /// Pad splay: median over the pad band (4–10 mm below the crest) of the mean left/right nose side
    /// angle. Single levels are noisy on TrueDepth data; the band median was stable across scans.
    var padSplay: Float? {
        let v = slices.filter { $0.levelBelowCrest >= 0.0035 && $0.levelBelowCrest <= 0.0105 }.compactMap { s -> Float? in
            let a = [s.splayLeft, s.splayRight].compactMap { $0 }
            return a.isEmpty ? nil : a.reduce(0, +) / Float(a.count)
        }
        return v.count >= 3 ? v.median() : nil
    }

    var frontalAngle: Float? {
        let v = [frontalAngleLeft, frontalAngleRight].compactMap { $0 }
        return v.isEmpty ? nil : v.reduce(0, +) / Float(v.count)
    }

    func ear(_ side: HeadSide) -> EarFit? { side == .left ? earLeft : earRight }

    /// Horizontal (z) distance from the frame back plane to the ear seat ("length to bend").
    func lengthToBend(_ side: HeadSide) -> Float? { ear(side).map { backPlaneZ - $0.seat.z } }

    var headWidthAtEars: Float? {
        guard let l = earLeft, let r = earRight else { return nil }
        return l.seat.x - r.seat.x
    }

    /// Temple drop: how far the ear seat sits below the pupil line (hinges are usually at pupil height).
    func templeDrop(_ side: HeadSide) -> Float? {
        ear(side).map { (pupilLeft.y + pupilRight.y) / 2 - $0.seat.y }
    }

    func templeDeclination(_ side: HeadSide) -> Float? {
        guard let drop = templeDrop(side), let len = lengthToBend(side), len > 0 else { return nil }
        return atan2(drop, len) * 180 / .pi
    }
}

/// Derives glasses-fitting landmarks and measurements from a head mesh in the ARKit face frame
/// (+x = subject's left, +y up, +z out of the face; metres).
final class FitAnalyzer {
    let mesh: TriMesh
    let eyeCenterLeft: Vec3
    let eyeCenterRight: Vec3
    var crestDropBelowSellion: Float = 0.001

    private let front: HeightMap
    private var ridge: [Int: (x: Float, z: Float)] = [:] // row -> ridge point
    private var baseWarnings: [String] = []
    private var sellion = Vec3.zero
    private var noseTip = Vec3.zero
    private var midlineX: Float = 0
    private var corneaZ: Float = 0
    private var pupilL = Vec3.zero
    private var pupilR = Vec3.zero
    private var autoEars: [HeadSide: EarFit] = [:]
    private var sideMaps: [HeadSide: (map: HeightMap, skull: (Float, Float) -> Float)] = [:]
    private var rejectedEars: [HeadSide: String] = [:]

    init(mesh: TriMesh, eyeCenterLeft: Vec3, eyeCenterRight: Vec3) {
        self.mesh = mesh
        self.eyeCenterLeft = eyeCenterLeft
        self.eyeCenterRight = eyeCenterRight
        let m = (eyeCenterLeft + eyeCenterRight) / 2
        front = HeightMap(mesh: mesh, uAxis: Vec3(1, 0, 0), vAxis: Vec3(0, 1, 0), depthAxis: Vec3(0, 0, 1),
                          uRange: (m.x - 0.075)...(m.x + 0.075), vRange: (m.y - 0.075)...(m.y + 0.045),
                          resolution: 0.0002,
                          include: { c, _ in c.z > m.z - 0.05 })
        analyzeFace()
        for side in HeadSide.allCases {
            if let ear = detectEar(side) { autoEars[side] = ear }
            else {
                let why = rejectedEars[side].map { " (detection rejected: \($0))" } ?? ""
                baseWarnings.append("The \(side.rawValue) ear was not captured clearly\(why). Rescan turning further to the \(side == .left ? "right" : "left") with hair tucked behind the ear, or place the seat manually.")
            }
        }
    }

    // MARK: - Face / nose

    private func analyzeFace() {
        let m = (eyeCenterLeft + eyeCenterRight) / 2
        let ipd = simd_distance(eyeCenterLeft, eyeCenterRight)

        // Nose ridge: most anterior point near the midline, per row.
        let searchHalf: Float = 0.012
        let rTop = front.row(forV: m.y + 0.03), rBottom = front.row(forV: m.y - 0.07)
        var raw: [Int: (x: Float, z: Float)] = [:]
        for r in max(0, rBottom)...min(front.height - 1, rTop) {
            var best: (Float, Float)? = nil
            for c in front.col(forU: m.x - searchHalf)...front.col(forU: m.x + searchHalf) {
                guard let z = front.value(col: c, row: r) else { continue }
                if best == nil || z > best!.1 { best = (front.u(col: c), z) }
            }
            if let best { raw[r] = best }
        }
        // Profile smoothing: 1 mm median (spikes), then Gaussian σ = 1 mm (sensor noise). The ridge x
        // gets the same treatment so cross-sections are centred consistently.
        var med: [Int: (x: Float, z: Float)] = [:]
        for (r, p) in raw {
            let zs = (-2...2).compactMap { raw[r + $0]?.z }
            let xs = (-2...2).compactMap { raw[r + $0]?.x }
            med[r] = (xs.count >= 3 ? xs.median()! : p.x, zs.count >= 3 ? zs.median()! : p.z)
        }
        ridge = Self.gaussian(med, sigmaRows: 0.001 / front.resolution)
        let heavy = Self.gaussian(med, sigmaRows: 0.0025 / front.resolution)

        func ridgePoint(row r: Int) -> Vec3? {
            guard let p = ridge[r] else { return nil }
            return Vec3(p.x, front.v(row: r), p.z)
        }

        // Nose tip: most anterior ridge point below the eyes.
        var tip: Vec3? = nil
        for r in front.row(forV: m.y - 0.065)...front.row(forV: m.y - 0.008) {
            if let p = ridgePoint(row: r), tip == nil || p.z > tip!.z { tip = p }
        }
        noseTip = tip ?? Vec3(m.x, m.y - 0.035, m.z + 0.035)
        if tip == nil { baseWarnings.append("Nose tip not found in scan.") }

        // Sellion: where the (heavily smoothed) profile bends most sharply from the nose slope into
        // the forehead (largest concave second difference over ±5 mm). Works with or without a brow
        // bump. The raw profile made this jump by up to 14 mm between scans of the same person.
        var sel: Vec3? = nil
        let h = Int(0.005 / front.resolution)
        // Anatomically the sellion lies ~10 mm below to ~25 mm above the pupil line.
        let lowRow = front.row(forV: max(noseTip.y + 0.012, m.y - 0.01)), highRow = front.row(forV: m.y + 0.025)
        if lowRow < highRow {
            var bestBend: Float = 0.0005
            for r in lowRow...highRow {
                guard let p = heavy[r], let up = heavy[r + h], let down = heavy[r - h] else { continue }
                let bend = up.z + down.z - 2 * p.z
                if bend > bestBend { bestBend = bend; sel = ridgePoint(row: r) }
            }
        }
        sellion = sel ?? Vec3(m.x, m.y + 0.005, m.z + 0.015)
        if sel == nil { baseWarnings.append("Nose bridge (sellion) not found in scan.") }
        else if sellion.y > m.y + 0.025 { baseWarnings.append("Sellion detected unusually high; check the bridge crest position.") }

        let ridgeXs = ridge.filter { front.v(row: $0.key) <= sellion.y && front.v(row: $0.key) >= noseTip.y }.map { $0.value.x }
        midlineX = ridgeXs.median() ?? m.x

        // Cornea apex: the scanned surface in front of each eye point. On devices ARKit's eye
        // transform sits near the front of the eye (the scanned cornea was level with it), so accept
        // the scan from 6 mm behind to 20 mm in front; only without scan data assume an eyeball centre.
        func apex(_ e: Vec3) -> Vec3 {
            let samples = [(0, 0), (-0.001, 0), (0.001, 0), (0, -0.001), (0, 0.001)].compactMap { d in
                front.value(u: e.x + Float(d.0), v: e.y + Float(d.1))
            }
            if let z = samples.median(), z - e.z > -0.006, z - e.z < 0.02 { return Vec3(e.x, e.y, z) }
            return Vec3(e.x, e.y, e.z + 0.0125)
        }
        pupilL = apex(eyeCenterLeft)
        pupilR = apex(eyeCenterRight)
        corneaZ = (pupilL.z + pupilR.z) / 2

        if ipd < 0.045 || ipd > 0.08 { baseWarnings.append(String(format: "Unusual eye distance (%.1f mm); eye tracking may have been poor.", ipd * 1000)) }
    }

    /// Gaussian smoothing of a row-indexed profile (missing rows are skipped, not zero-filled).
    private static func gaussian(_ p: [Int: (x: Float, z: Float)], sigmaRows: Float) -> [Int: (x: Float, z: Float)] {
        let reach = Int((sigmaRows * 3).rounded(.up))
        var out: [Int: (x: Float, z: Float)] = [:]
        for r in p.keys {
            var sx: Float = 0, sz: Float = 0, sw: Float = 0
            for k in -reach...reach {
                guard let q = p[r + k] else { continue }
                let w = exp(-Float(k * k) / (2 * sigmaRows * sigmaRows))
                sx += w * q.x; sz += w * q.z; sw += w
            }
            out[r] = (sx / sw, sz / sw)
        }
        return out
    }

    /// The sellion as found on this mesh (used to anchor the scan's crest from a fitted head model).
    var sellionPoint: Vec3 { sellion }

    /// Anchors the sellion height from a fitted head model (stable across scans); depth and x stay
    /// on the scanned ridge.
    /// What the head model and the photos added to the scan-only analysis, so a saved scan reopens as it was.
    struct Inputs: Codable {
        var sellionY: Float?
        var pupils: [Vec3]?
        var pupilLeftShare: Float?
        var modelEars: [EarFit] = []
        var modelEarWarnings: [String] = []
    }
    private(set) var inputs = Inputs()

    func apply(_ i: Inputs) {
        if let y = i.sellionY { anchorSellion(y: y) }
        if let p = i.pupils, p.count == 2 { setMeasuredPupils(left: p[0], right: p[1], leftShare: i.pupilLeftShare) }
        if !i.modelEars.isEmpty || !i.modelEarWarnings.isEmpty {
            setModelEars(Dictionary(i.modelEars.map { ($0.side, $0) }, uniquingKeysWith: { a, _ in a }), warnings: i.modelEarWarnings)
        }
    }

    func anchorSellion(y: Float) {
        inputs.sellionY = y
        guard let r = ridgeAt(y: y) else { return }
        sellion = r
        baseWarnings.removeAll { $0.hasPrefix("Nose bridge (sellion) not found") || $0.hasPrefix("Sellion detected unusually high") }
    }

    private func ridgeAt(y: Float) -> Vec3? {
        let r = front.row(forV: y)
        guard let p = ridge[r] else { return nil }
        return Vec3(p.x, front.v(row: r), p.z)
    }

    /// Smoothed nose cross-section at height `y`: for each side, the surface depth (z) at 0.2 mm steps
    /// outward from the ridge. Rows within ±0.6 mm are combined by median, then a 1 mm median runs
    /// along the row; short holes (≤ 1 mm) are bridged.
    private func crossSection(y: Float, ridge rp: Vec3) -> (left: [Float], right: [Float]) {
        let rc = front.row(forV: y), c0 = front.col(forU: rp.x)
        let rows = (rc - 3)...(rc + 3)
        let maxSteps = Int(0.03 / front.resolution)
        func side(_ step: Int) -> [Float] {
            var col: [Float?] = []
            for k in 0...maxSteps {
                let c = c0 + k * step
                let zs = rows.compactMap { front.value(col: c, row: $0) }
                col.append(zs.count >= 3 ? zs.median() : nil)
            }
            // bridge holes up to 1 mm, stop at larger ones
            var filled: [Float] = []
            var i = 0
            while i < col.count {
                if let z = col[i] { filled.append(z); i += 1; continue }
                guard let last = filled.last, let next = (i..<min(col.count, i + 6)).first(where: { col[$0] != nil }) else { break }
                let gap = next - i + 1, nz = col[next]!
                for g in 1..<gap { filled.append(last + (nz - last) * Float(g) / Float(gap)) }
                i = next
            }
            // 1 mm running median along the row
            return filled.indices.map { j in Array(filled[max(0, j - 2)...min(filled.count - 1, j + 2)]).median()! }
        }
        return (side(1), side(-1))
    }

    /// Distance from the ridge where the profile has dropped `depth` below the ridge and stays below
    /// for ≥ 0.6 mm (rejects single-sample dips).
    private func halfWidth(_ profile: [Float], ridgeZ: Float, depth: Float) -> Float? {
        let target = ridgeZ - depth
        guard profile.count > 4 else { return nil }
        for i in 1..<(profile.count - 3) where profile[i] <= target {
            if profile[(i + 1)...(i + 3)].allSatisfy({ $0 <= target + 0.0003 }) {
                let a = profile[i - 1], b = profile[i]
                let t = (a - target) / max(a - b, 1e-6)
                return (Float(i - 1) + t) * front.resolution
            }
            if profile[i] > ridgeZ + 0.002 { return nil }
        }
        return nil
    }

    /// Theil–Sen slope (median of pairwise slopes): robust to the odd outlier width.
    private static func robustSlope(_ pts: [(Float, Float)]) -> Float? {
        guard pts.count >= 3 else { return nil }
        var slopes: [Float] = []
        for i in 0..<pts.count { for j in (i + 1)..<pts.count where abs(pts[j].0 - pts[i].0) > 1e-6 {
            slopes.append((pts[j].1 - pts[i].1) / (pts[j].0 - pts[i].0))
        }}
        return slopes.median()
    }

    // MARK: - Ears

    private func detectEar(_ side: HeadSide) -> EarFit? {
        let s = side.sign
        let m = (eyeCenterLeft + eyeCenterRight) / 2
        let res: Float = 0.00025
        let uRange = (corneaZ - 0.14)...(corneaZ - 0.05)
        let vRange = (m.y - 0.07)...(m.y + 0.05)
        let map = HeightMap(mesh: mesh, uAxis: Vec3(0, 0, 1), vAxis: Vec3(0, 1, 0), depthAxis: Vec3(s, 0, 0),
                            uRange: uRange, vRange: vRange, resolution: res,
                            include: { c, n in s * c.x > 0.035 && s * n.x > -0.2 })

        // Robust cubic fit of the skull side, down-weighting the outward-protruding ear.
        var samples: [(Float, Float, Float)] = []
        for r in stride(from: 0, to: map.height, by: 4) {
            for c in stride(from: 0, to: map.width, by: 4) {
                if let d = map.value(col: c, row: r) { samples.append((map.u(col: c), map.v(row: r), d)) }
            }
        }
        guard samples.count > 400 else { return nil }
        let uc = (uRange.lowerBound + uRange.upperBound) / 2, us = (uRange.upperBound - uRange.lowerBound) / 2
        let vc = (vRange.lowerBound + vRange.upperBound) / 2, vs = (vRange.upperBound - vRange.lowerBound) / 2
        func basis(_ u: Float, _ v: Float) -> [Double] {
            let a = Double((u - uc) / us), b = Double((v - vc) / vs)
            return [1, a, b, a * a, a * b, b * b, a * a * a, a * a * b, a * b * b, b * b * b]
        }
        let nb = 10
        var coeffs = [Double](repeating: 0, count: nb)
        var weights = [Double](repeating: 1, count: samples.count)
        for _ in 0..<12 {
            var ata = [Double](repeating: 0, count: nb * nb)
            var atb = [Double](repeating: 0, count: nb)
            for (i, smp) in samples.enumerated() {
                let f = basis(smp.0, smp.1), w = weights[i]
                for a in 0..<nb {
                    atb[a] += w * f[a] * Double(smp.2)
                    for b in 0..<nb { ata[a * nb + b] += w * f[a] * f[b] }
                }
            }
            for a in 0..<nb { ata[a * nb + a] += 1e-6 }
            guard let sol = RigidMath.solve(ata, atb, n: nb) else { return nil }
            coeffs = sol
            for (i, smp) in samples.enumerated() {
                let f = basis(smp.0, smp.1)
                let fit = zip(f, coeffs).reduce(0) { $0 + $1.0 * $1.1 }
                let r = Double(smp.2) - fit
                if r > 0.0012 { weights[i] = pow(0.0012 / r, 3) }
                else if r < -0.004 { weights[i] = 0.3 }
                else { weights[i] = 1 }
            }
        }
        let skull: (Float, Float) -> Float = { u, v in
            Float(zip(basis(u, v), coeffs).reduce(0) { $0 + $1.0 * $1.1 })
        }
        sideMaps[side] = (map, skull)

        // Ear mask: protrusion beyond the skull fit.
        let w = map.width, h = map.height
        var mask = [Bool](repeating: false, count: w * h)
        for r in 0..<h {
            for c in 0..<w {
                if let d = map.value(col: c, row: r), d - skull(map.u(col: c), map.v(row: r)) > 0.003 { mask[r * w + c] = true }
            }
        }
        // Largest 4-connected component.
        var label = [Int32](repeating: -1, count: w * h)
        var best: [Int] = []
        var stack: [Int] = []
        for start in 0..<(w * h) where mask[start] && label[start] < 0 {
            var comp: [Int] = []
            stack.append(start)
            label[start] = 1
            while let i = stack.popLast() {
                comp.append(i)
                let r = i / w, c = i % w
                for (nr, nc) in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] where nr >= 0 && nr < h && nc >= 0 && nc < w {
                    let j = nr * w + nc
                    if mask[j] && label[j] < 0 { label[j] = 1; stack.append(j) }
                }
            }
            if comp.count > best.count { best = comp }
        }
        let minArea = 0.00015 / (res * res) // 150 mm²
        guard Float(best.count) >= minArea else { return nil }

        var topRow = Int.min, bottomRow = Int.max
        var colTop: [Int: Int] = [:]
        var inEar = [Bool](repeating: false, count: w * h)
        for i in best {
            inEar[i] = true
            let r = i / w, c = i % w
            topRow = max(topRow, r); bottomRow = min(bottomRow, r)
            colTop[c] = max(colTop[c] ?? Int.min, r)
        }
        // Upper 40 % of the ear: helix root, where the temple rests.
        let upperFrom = bottomRow + Int(Float(topRow - bottomRow) * 0.6)
        var frontCol = Int.min, backCol = Int.max
        for i in best where i / w >= upperFrom {
            frontCol = max(frontCol, i % w); backCol = min(backCol, i % w)
        }
        guard frontCol > backCol else { return nil }
        // Plausibility: an adult ear is ~45–75 mm tall. A protrusion outside that is noise or hair.
        let earHeight = Float(topRow - bottomRow) * res
        guard earHeight > 0.03, earHeight < 0.09 else {
            rejectedEars[side] = String(format: "outline %.0f mm tall", earHeight * 1000)
            return nil
        }

        func pt(_ c: Int, _ r: Int) -> Vec3 {
            let d = map.value(col: c, row: r) ?? skull(map.u(col: c), map.v(row: r))
            return map.point(col: c, row: r, depth: d)
        }
        // Visible junction estimate from the side outline: the temple rests over the ear root,
        // about the front third of the ear's upper edge.
        let seatCol = frontCol - Int(Float(frontCol - backCol) * 0.3)
        let nearest = colTop.keys.min(by: { abs($0 - seatCol) < abs($1 - seatCol) }) ?? seatCol
        let outlineTopV = map.v(row: colTop[nearest] ?? topRow)
        let seatU = map.u(col: nearest)
        var seat = map.point(u: seatU, v: outlineTopV - 0.002, depth: skull(seatU, outlineTopV - 0.002))
        var method = "outline"
        if let crease = findCrease(side: side, aroundU: seatU, topV: outlineTopV,
                                   earSpan: map.u(col: backCol)...map.u(col: frontCol), skull: skull) {
            seat = crease
            method = "crease"
        }

        // Boundary for display
        var outline: [Vec3] = []
        for i in best {
            let r = i / w, c = i % w
            if (r + c) % 3 != 0 { continue }
            let boundary = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)].contains { nr, nc in
                nr < 0 || nr >= h || nc < 0 || nc >= w || !inEar[nr * w + nc]
            }
            if boundary { outline.append(pt(c, r)) }
        }

        // The temple seat sits ~65–120 mm behind the cornea; anything else is not the ear root.
        let setback = corneaZ - seat.z
        guard setback > 0.06, setback < 0.125 else {
            rejectedEars[side] = String(format: "seat %.0f mm behind the eye", setback * 1000)
            return nil
        }

        let topCol = colTop.first(where: { $0.value == topRow })?.key ?? nearest
        let bottomCol = best.first(where: { $0 / w == bottomRow }).map { $0 % w } ?? nearest
        return EarFit(side: side, seat: seat, method: method,
                      earTop: pt(topCol, topRow), earBottom: pt(bottomCol, bottomRow),
                      earFront: pt(frontCol, upperFrom), earBack: pt(backCol, upperFrom),
                      outline: outline)
    }

    /// Ear seat for an ear whose extent is known (e.g. the `ears` vertices of a fitted GNM head):
    /// the groove search at the front third of the ear's upper edge. `skullX(z, y)` gives the
    /// lateral position (|x|) of the head surface next to the ear.
    func modelEarSeat(side: HeadSide, earVertices: [Vec3], skullX: @escaping (Float, Float) -> Float) -> EarFit? {
        guard earVertices.count > 50,
              let top = earVertices.max(by: { $0.y < $1.y }),
              let bottom = earVertices.min(by: { $0.y < $1.y }) else { return nil }
        let upper = earVertices.filter { $0.y > bottom.y + 0.6 * (top.y - bottom.y) }
        guard let front = upper.max(by: { $0.z < $1.z }), let back = upper.min(by: { $0.z < $1.z }) else { return nil }
        let seatZ = front.z - 0.3 * (front.z - back.z)
        // Top edge of the ear at that depth.
        let edge = earVertices.filter { abs($0.z - seatZ) < 0.003 }.map { $0.y }.max() ?? top.y
        let crease = findCrease(side: side, aroundU: seatZ, topV: edge, earSpan: back.z...front.z, skull: skullX)
        let seat = crease ?? Vec3(side.sign * skullX(seatZ, edge - 0.002), edge - 0.002, seatZ)
        return EarFit(side: side, seat: seat, method: crease != nil ? "model crease" : "model outline",
                      earTop: top, earBottom: bottom, earFront: front, earBack: back, outline: [])
    }

    /// Looks for the groove between the skull and the top of the ear in coronal slices of the mesh:
    /// the upper envelope of the slice dips between the skull wall and the helix. Only grooves at the
    /// top of the detected ear count (hair above the ear produces spurious valleys), and the result
    /// is a robust upper quantile of the groove heights rather than the single highest point.
    private func findCrease(side: HeadSide, aroundU: Float, topV: Float, earSpan: ClosedRange<Float>,
                            skull: (Float, Float) -> Float) -> Vec3? {
        let s = side.sign
        let yLo = topV - 0.014, yHi = topV + 0.01
        let skullX = skull(aroundU, topV)
        let xLo = skullX - 0.004, xHi = skullX + 0.02
        let zLo = aroundU - 0.005, zHi = aroundU + 0.005

        var tris: [(Vec3, Vec3, Vec3)] = []
        for t in mesh.triangles {
            let a = mesh.vertices[Int(t.x)], b = mesh.vertices[Int(t.y)], c = mesh.vertices[Int(t.z)]
            let cx = s * (a.x + b.x + c.x) / 3, cy = (a.y + b.y + c.y) / 3, cz = (a.z + b.z + c.z) / 3
            if cx < xLo || cx > xHi || cy < yLo || cy > yHi || cz < zLo - 0.001 || cz > zHi + 0.001 { continue }
            tris.append((a, b, c))
        }
        guard tris.count > 50 else { return nil }

        let binRes: Float = 0.00025
        let nBins = Int((xHi - xLo) / binRes) + 1
        var candidates: [Vec3] = []
        var z = zLo
        while z <= zHi {
            var env = [Float](repeating: -.infinity, count: nBins)
            for (a, b, c) in tris {
                let pts = [a, b, c]
                var hits: [Vec3] = []
                for (p, q) in [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[0])] {
                    if (p.z - z) * (q.z - z) < 0 {
                        let t = (z - p.z) / (q.z - p.z)
                        hits.append(p + t * (q - p))
                    }
                }
                guard hits.count == 2 else { continue }
                let n = max(1, Int(simd_distance(hits[0], hits[1]) / (binRes * 0.5)))
                for k in 0...n {
                    let p = hits[0] + Float(k) / Float(n) * (hits[1] - hits[0])
                    let bin = Int((s * p.x - xLo) / binRes)
                    if bin >= 0 && bin < nBins && p.y > env[bin] { env[bin] = p.y }
                }
            }
            // Valley in the upper envelope: low point with the skull wall rising medially and the helix laterally.
            let skullBin = Int((skull(z, topV) - xLo) / binRes)
            var bestBin: Int? = nil
            let lo = max(1, skullBin - 8), hi = min(nBins - 1, skullBin + 24)
            for i in (lo < hi ? lo..<hi : 0..<0) where env[i].isFinite {
                let y = env[i]
                let medial = env[max(0, i - 16)..<i].filter { $0.isFinite }.max() ?? -.infinity
                let lateral = env[(i + 1)..<min(nBins, i + 48)].filter { $0.isFinite }.max() ?? -.infinity
                if medial - y > 0.001 && lateral - y > 0.0015 {
                    if bestBin == nil || y < env[bestBin!] { bestBin = i }
                }
            }
            if let b = bestBin {
                candidates.append(Vec3(s * (xLo + (Float(b) + 0.5) * binRes), env[b], z))
            }
            z += 0.001
        }
        let valid = candidates.filter {
            $0.y > topV - 0.012 && $0.y < topV + 0.002 && $0.z > earSpan.lowerBound - 0.001 && $0.z < earSpan.upperBound + 0.002
        }
        guard valid.count >= 3 else { return nil }
        // A straight temple rests near the top of the groove; the 70th percentile resists noise spikes.
        let sorted = valid.sorted { $0.y < $1.y }
        return sorted[min(sorted.count - 1, Int(Float(sorted.count) * 0.7))]
    }

    private var measuredPupils: (left: Vec3, right: Vec3, pd: Float, leftShare: Float?)?

    /// Pupils measured in the colour images (distance-corrected, head frame).
    func setMeasuredPupils(left: Vec3, right: Vec3, leftShare: Float?) {
        measuredPupils = (left, right, simd_distance(left, right), leftShare)
        inputs.pupils = [left, right]
        inputs.pupilLeftShare = leftShare
    }

    /// Replaces the scan-detected ears with ears found on a fitted head model (GNM).
    func setModelEars(_ ears: [HeadSide: EarFit], warnings: [String] = []) {
        inputs.modelEars = Array(ears.values)
        inputs.modelEarWarnings = warnings
        for (side, ear) in ears {
            autoEars[side] = ear
            baseWarnings.removeAll { $0.hasPrefix("The \(side.rawValue) ear") }
        }
        baseWarnings.append(contentsOf: warnings)
    }

    // MARK: - Result

    func analyze(overrides: FitOverrides = FitOverrides()) -> FitResult {
        var warnings = baseWarnings
        var crest: Vec3
        let manualCrest = overrides.crest != nil
        if let c = overrides.crest {
            crest = c
        } else {
            crest = ridgeAt(y: sellion.y - crestDropBelowSellion) ?? sellion
        }
        // Snap the crest onto the ridge row it sits in (keeps depth consistent with the profile).
        if manualCrest, let r = ridgeAt(y: crest.y) { crest = Vec3(r.x, crest.y, r.z) }
        let backPlaneZ = crest.z

        // Raw cross-sections every 1 mm below the crest.
        struct Raw { var level: Float; var ridge: Vec3; var left: [Float?]; var right: [Float?]; var bpL: Float?; var bpR: Float? }
        var raws: [Raw] = []
        for mm in 0...25 {
            let level = Float(mm) / 1000
            let y = crest.y - level
            guard let rp = ridgeAt(y: y) else { continue }
            let cs = crossSection(y: y, ridge: rp)
            let depths = (1...15).map { Float($0) / 1000 }
            let bpDepth = rp.z - backPlaneZ
            raws.append(Raw(level: level, ridge: rp,
                            left: depths.map { halfWidth(cs.left, ridgeZ: rp.z, depth: $0) },
                            right: depths.map { halfWidth(cs.right, ridgeZ: rp.z, depth: $0) },
                            bpL: bpDepth <= 0 ? 0 : halfWidth(cs.left, ridgeZ: rp.z, depth: bpDepth),
                            bpR: bpDepth <= 0 ? 0 : halfWidth(cs.right, ridgeZ: rp.z, depth: bpDepth)))
        }
        // Smooth across levels: median over ±2 mm (a nose changes slowly from level to level; noise does not).
        func acrossLevels(_ i: Int, _ pick: (Raw) -> Float?) -> Float? {
            let vals = raws[max(0, i - 2)...min(raws.count - 1, i + 2)].compactMap(pick)
            return vals.count >= 2 ? vals.median() : pick(raws[i])
        }

        var slices: [NoseSlice] = []
        var padL: [Vec3] = [], padR: [Vec3] = []
        for (i, raw) in raws.enumerated() {
            let hw: [(Float, Float?, Float?)] = (0..<15).map { d in
                (Float(d + 1) / 1000, acrossLevels(i) { $0.left[d] }, acrossLevels(i) { $0.right[d] })
            }
            let l = acrossLevels(i) { $0.bpL }, r = acrossLevels(i) { $0.bpR }
            var bpWidth: Float? = nil
            if let l, let r {
                bpWidth = l + r
                if raw.level <= 0.020 && bpWidth! > 0 {
                    padL.append(Vec3(raw.ridge.x + l, raw.ridge.y, backPlaneZ))
                    padR.append(Vec3(raw.ridge.x - r, raw.ridge.y, backPlaneZ))
                }
            }
            // Splay: robust slope of half-width vs depth over 2–10 mm.
            func splay(_ pick: ((Float, Float?, Float?)) -> Float?) -> Float? {
                let pts = hw.filter { $0.0 >= 0.002 && $0.0 <= 0.010 }.compactMap { e in pick(e).map { (e.0, $0) } }
                guard pts.count >= 4, let slope = Self.robustSlope(pts) else { return nil }
                return atan(slope) * 180 / .pi
            }
            slices.append(NoseSlice(levelBelowCrest: raw.level, ridge: raw.ridge, halfWidths: hw, backPlaneWidth: bpWidth,
                                    splayLeft: splay { $0.1 }, splayRight: splay { $0.2 }))
        }

        // Frontal angle: robust inclination of the pad contact line (back plane) from vertical, 3–15 mm below crest.
        func frontal(_ pts: [Vec3], sign: Float) -> Float? {
            let sel = pts.filter { crest.y - $0.y >= 0.003 && crest.y - $0.y <= 0.015 }
            guard sel.count >= 5, let slope = Self.robustSlope(sel.map { ($0.y, $0.x) }) else { return nil }
            return atan(-sign * slope) * 180 / .pi
        }

        var earL = autoEars[.left], earR = autoEars[.right]
        if let o = overrides.earLeft { earL = manualEar(.left, seat: o, base: earL) }
        if let o = overrides.earRight { earR = manualEar(.right, seat: o, base: earR) }

        // Nose sanity: the two flanks of a nose are close to mirror images. A large left/right
        // difference means one flank is missing or carved in the scan (seen on real captures).
        let flank = slices.filter { $0.levelBelowCrest >= 0.004 && $0.levelBelowCrest <= 0.015 }.compactMap { s -> Float? in
            guard let e = s.halfWidths.first(where: { abs($0.depth - 0.006) < 0.0001 }), let l = e.left, let r = e.right else { return nil }
            return l - r
        }
        if flank.count >= 5, let asym = flank.median(), abs(asym) > 0.0025 {
            warnings.append(String(format: "The nose surface looks incomplete on the %@ side (flanks differ by %.1f mm). Rescan facing the phone more directly; bridge measurements may be off.",
                                   asym < 0 ? "left" : "right", abs(asym) * 1000))
        }

        let vd = backPlaneZ - corneaZ
        if vd < 0.008 || vd > 0.02 {
            warnings.append(String(format: "Frame back plane at the crest gives a vertex distance of %.1f mm (typical 10–15 mm). Consider moving the crest or using pads.", vd * 1000))
        }
        if let l = earL, let r = earR, abs(l.seat.y - r.seat.y) > 0.004 {
            warnings.append(String(format: "Ears differ in height by %.1f mm; temples need individual bends.", abs(l.seat.y - r.seat.y) * 1000))
        }

        var result = FitResult(pupilLeft: measuredPupils?.left ?? pupilL, pupilRight: measuredPupils?.right ?? pupilR,
                         eyeCenterLeft: eyeCenterLeft, eyeCenterRight: eyeCenterRight,
                         corneaZ: corneaZ, midlineX: midlineX,
                         sellion: sellion, noseTip: noseTip, crest: crest, crestIsManual: manualCrest,
                         backPlaneZ: backPlaneZ, slices: slices, padContactLeft: padL, padContactRight: padR,
                         frontalAngleLeft: frontal(padL, sign: 1), frontalAngleRight: frontal(padR, sign: -1),
                         earLeft: earL, earRight: earR, warnings: warnings)
        result.measuredPD = measuredPupils?.pd
        result.measuredLeftShare = measuredPupils?.leftShare
        return result
    }

    private func manualEar(_ side: HeadSide, seat: Vec3, base: EarFit?) -> EarFit {
        var e = base ?? EarFit(side: side, seat: seat, method: "manual", outline: [])
        e.seat = seat
        e.method = "manual"
        return e
    }
}
