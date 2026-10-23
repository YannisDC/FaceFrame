import Foundation
import simd

/// JSON written next to the mesh files. Everything is in millimetres in the "fitting frame":
/// origin at the bridge crest, axes of the ARKit face frame.
struct FitExport: Codable {
    struct Measurements: Codable {
        var pupillaryDistance: Float
        /// "images" (pupils detected in the colour frames) or "arkit" (ARKit's face model).
        var pupilSource: String
        var monocularPDLeft: Float
        var monocularPDRight: Float
        var vertexDistance: Float
        var pupilHeightAboveCrest: Float
        var sellionAbovePupilLine: Float
        var bridgeWidthDBR5: Float?
        var bridgeWidthDBR10: Float?
        var bridgeWidthDBR15: Float?
        var splayAngle5Deg: Float?
        var splayAngle10Deg: Float?
        /// Median nose side angle over the pad band 4–10 mm below the crest (most robust splay value).
        var padSplayDeg: Float?
        var frontalAngleDeg: Float?
        var lengthToBendLeft: Float?
        var lengthToBendRight: Float?
        var headWidthAtEarSeats: Float?
        var templeDropLeft: Float?
        var templeDropRight: Float?
        var templeDeclinationLeftDeg: Float?
        var templeDeclinationRightDeg: Float?
        var earSeatHeightDifference: Float?
        var earSeatSetbackDifference: Float?
    }

    struct NoseSliceExport: Codable {
        var levelBelowCrest: Float
        var ridge: [Float]
        var backPlaneWidth: Float?
        var splayLeftDeg: Float?
        var splayRightDeg: Float?
        /// [depth behind ridge, half-width left, half-width right]; -1 where not measurable.
        var halfWidths: [[Float]]
    }

    struct EarExport: Codable {
        var side: String
        var seat: [Float]
        var method: String
        var earTop: [Float]?
        var earBottom: [Float]?
        var earFront: [Float]?
        var earBack: [Float]?
    }

    var schema = "faceframe.fit/1"
    var units = "mm"
    var coordinateSystem = "Origin at the bridge crest. +x = subject's left, +y = up, +z = forward (out of the face). The exported meshes use the same frame."
    var frameModel = "Frame back plane is vertical (no pantoscopic tilt) and touches the nose at the crest. DBR = nose width in the back plane at 5/10/15 mm below the crest."
    var createdAt: Date
    var isDemo: Bool
    var landmarks: [String: [Float]]
    var measurements: Measurements
    var padContactLeft: [[Float]]
    var padContactRight: [[Float]]
    var noseSlices: [NoseSliceExport]
    var ears: [EarExport]
    var warnings: [String]
    var scan: ScanExport?
    var headModel: HeadModelExport?

    struct HeadModelExport: Codable {
        var model = "Google GNM v3 (Apache 2.0), identity fit"
        var fitRMSmm: Float
        /// Identity coefficients (GNM's head components, in order), at `headScale` (the metric size).
        /// GNMFrames reads this file directly to design frames on a laptop.
        var identity: [Float]?
        var headScale: Float? = 1
        /// "photos": GNM's photo fit on the colour frames, sized on the scan; "scan": fitted to the scan.
        var source: String?
        /// Sex/ethnicity class the identity prior was centred on, nil for GNM's default prior.
        var prior: String?
        var earCoverageLeft: Float?
        var earCoverageRight: Float?
        var earDistanceLeftMM: Float?
        var earDistanceRightMM: Float?
    }

    struct ScanExport: Codable {
        var depthMapping: String?
        var depthMatchMM: Float?
        var fusedFrames: Int
        var droppedTooFast: Int
        var droppedMisaligned: Int
        var droppedOther: Int
        var meanAlignmentRMSmm: Float?
        var arkitScaleCorrection: Float?
        var arkitOffsetMM: [Float]?
        var faceFitBeforeMM: Float?
        var faceFitAfterMM: Float?
        var depthMappingSwitches: Int
    }

    init(result r: FitResult, createdAt: Date, isDemo: Bool, diagnostics: ScanDiagnostics? = nil,
         modelFit: ModelFitSummary? = nil) {
        let o = r.crest
        func p(_ v: Vec3) -> [Float] { let q = (v - o) * 1000; return [q.x, q.y, q.z].map { ($0 * 100).rounded() / 100 } }
        func mm(_ v: Float?) -> Float? { v.map { ($0 * 10000).rounded() / 10 } }
        func deg(_ v: Float?) -> Float? { v.map { ($0 * 10).rounded() / 10 } }

        self.createdAt = createdAt
        self.isDemo = isDemo
        var lm: [String: [Float]] = [
            "crest": p(r.crest), "sellion": p(r.sellion), "noseTip": p(r.noseTip),
            "pupilLeft": p(r.pupilLeft), "pupilRight": p(r.pupilRight),
            "eyeCenterLeft": p(r.eyeCenterLeft), "eyeCenterRight": p(r.eyeCenterRight),
        ]
        if let e = r.earLeft { lm["earSeatLeft"] = p(e.seat) }
        if let e = r.earRight { lm["earSeatRight"] = p(e.seat) }
        landmarks = lm

        measurements = Measurements(
            pupillaryDistance: mm(r.pd)!, pupilSource: r.pupilsFromImages ? "images" : "arkit", monocularPDLeft: mm(r.monoPDLeft)!, monocularPDRight: mm(r.monoPDRight)!,
            vertexDistance: mm(r.vertexDistance)!, pupilHeightAboveCrest: mm(r.pupilHeightAboveCrest)!,
            sellionAbovePupilLine: mm(r.sellionAbovePupils)!,
            bridgeWidthDBR5: mm(r.dbr(at: 0.005)), bridgeWidthDBR10: mm(r.dbr(at: 0.010)), bridgeWidthDBR15: mm(r.dbr(at: 0.015)),
            splayAngle5Deg: deg(r.splay(at: 0.005)), splayAngle10Deg: deg(r.splay(at: 0.010)), padSplayDeg: deg(r.padSplay), frontalAngleDeg: deg(r.frontalAngle),
            lengthToBendLeft: mm(r.lengthToBend(.left)), lengthToBendRight: mm(r.lengthToBend(.right)),
            headWidthAtEarSeats: mm(r.headWidthAtEars),
            templeDropLeft: mm(r.templeDrop(.left)), templeDropRight: mm(r.templeDrop(.right)),
            templeDeclinationLeftDeg: deg(r.templeDeclination(.left)), templeDeclinationRightDeg: deg(r.templeDeclination(.right)),
            earSeatHeightDifference: r.earLeft.flatMap { l in r.earRight.map { mm(l.seat.y - $0.seat.y)! } },
            earSeatSetbackDifference: r.earLeft.flatMap { l in r.earRight.map { mm(l.seat.z - $0.seat.z)! } })

        padContactLeft = r.padContactLeft.map(p)
        padContactRight = r.padContactRight.map(p)
        noseSlices = r.slices.map { s in
            NoseSliceExport(levelBelowCrest: mm(s.levelBelowCrest)!, ridge: p(s.ridge), backPlaneWidth: mm(s.backPlaneWidth),
                            splayLeftDeg: deg(s.splayLeft), splayRightDeg: deg(s.splayRight),
                            halfWidths: s.halfWidths.map { [mm($0.depth)!, mm($0.left) ?? -1, mm($0.right) ?? -1] })
        }
        ears = [r.earLeft, r.earRight].compactMap { $0 }.map { e in
            EarExport(side: e.side.rawValue, seat: p(e.seat), method: e.method,
                      earTop: e.earTop.map(p), earBottom: e.earBottom.map(p), earFront: e.earFront.map(p), earBack: e.earBack.map(p))
        }
        warnings = r.warnings
        headModel = modelFit.map {
            HeadModelExport(fitRMSmm: $0.rmsMM, identity: $0.identity.map { ($0 * 10000).rounded() / 10000 },
                            headScale: ($0.headScale * 10000).rounded() / 10000, source: $0.source, prior: $0.prior,
                            earCoverageLeft: $0.earCoverage[.left], earCoverageRight: $0.earCoverage[.right],
                            earDistanceLeftMM: $0.earDistanceMM[.left], earDistanceRightMM: $0.earDistanceMM[.right])
        }
        scan = diagnostics.map {
            ScanExport(depthMapping: $0.mapping, depthMatchMM: $0.mappingAgreementMM, fusedFrames: $0.fusedFrames,
                       droppedTooFast: $0.rejectedFast, droppedMisaligned: $0.rejectedAlignment,
                       droppedOther: $0.rejectedOther, meanAlignmentRMSmm: $0.meanAlignmentRMSmm,
                       arkitScaleCorrection: $0.arkitScale, arkitOffsetMM: $0.arkitOffsetMM.map { [$0.x, $0.y, $0.z] },
                       faceFitBeforeMM: $0.faceFitBeforeMM, faceFitAfterMM: $0.faceFitAfterMM,
                       depthMappingSwitches: $0.mappingSwitches)
        }
    }
}

enum ScanExporter {
    /// Writes STL + PLY meshes and the fit JSON to Documents/Scans/<timestamp>/ and returns the file URLs.
    /// "scan_<timestamp>" (or "demo_…"): the export folder's and files' name.
    static func name(_ session: FitSession) -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return (session.isDemo ? "demo_" : "scan_") + fmt.string(from: session.createdAt)
    }

    /// Documents/Scans/<name>/, created if needed.
    static func directory(_ session: FitSession) throws -> URL {
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent("Scans", isDirectory: true).appendingPathComponent(name(session), isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func export(_ session: FitSession) throws -> [URL] {
        let name = name(session)
        let dir = try directory(session)

        let result = session.result
        // Mesh in the fitting frame (origin at the crest), millimetres.
        let toFitting = simd_float4x4(translation: -result.crest)
        let mesh = session.mesh.transformed(toFitting)

        let stl = dir.appendingPathComponent("\(name)_head.stl")
        let ply = dir.appendingPathComponent("\(name)_head.ply")
        let json = dir.appendingPathComponent("\(name)_fit.json")
        try mesh.stlData().write(to: stl)
        try mesh.plyData().write(to: ply)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(FitExport(result: result, createdAt: session.createdAt, isDemo: session.isDemo,
                                     diagnostics: session.diagnostics, modelFit: session.modelFit)).write(to: json)
        var files = [json, stl, ply]
        if let model = session.modelFit {
            // Complete head from the fitted GNM model (fills the back of the ears, skull, neck).
            let completed = dir.appendingPathComponent("\(name)_head_completed.stl")
            try model.mesh.transformed(toFitting).stlData().write(to: completed)
            files.append(completed)
        }
        return files
    }

    /// Zips a raw capture folder (via the file coordinator) for sharing.
    static func zipCapture(_ directory: URL) throws -> URL {
        var coordinatorError: NSError?
        var copyError: Error?
        var result: URL?
        NSFileCoordinator().coordinate(readingItemAt: directory, options: .forUploading, error: &coordinatorError) { zip in
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent(directory.lastPathComponent + ".zip")
            try? FileManager.default.removeItem(at: dest)
            do {
                try FileManager.default.copyItem(at: zip, to: dest)
                result = dest
            } catch {
                copyError = error
            }
        }
        if let e = coordinatorError ?? copyError { throw e }
        guard let result else { throw CocoaError(.fileWriteUnknown) }
        return result
    }
}

extension simd_float4x4 {
    init(translation t: Vec3) {
        self = matrix_identity_float4x4
        columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1)
    }
}
