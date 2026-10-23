import Foundation
import simd

/// Finished scans kept on the phone: the scanned surface (point cloud and mesh), the fitted head
/// model with its identity and scale, and everything the review screen shows, so a scan can be
/// reopened, adjusted and given frames later without scanning again. Each lives in its scan's folder
/// (Files › FaceFrame › Scans › <name>), next to its exports and frames:
///
///     session.json   identity, scale, source and prior of the head model, the measurements' inputs,
///                    landmark adjustments and scan diagnostics
///     scan.mesh      the cleaned scan: vertices, normals and triangles (metres, head frame), LZFSE-compressed
///     head.mesh      the head model's skin surface, same format
enum SavedScans {
    struct Session: Codable {
        var version = 1
        var name: String
        var createdAt: Date
        var savedAt: Date
        var isDemo: Bool
        var eyeLeft: Vec3
        var eyeRight: Vec3
        var analysis: FitAnalyzer.Inputs
        var overrides: [String: Vec3]
        var diagnostics: ScanDiagnostics?
        var headModel: HeadModel?
        /// The raw capture's folder name in Captures, when the scan was recorded.
        var capture: String?

        struct HeadModel: Codable {
            var identity: [Float]
            var headScale: Float
            var source: String
            var prior: String?
            var rmsMM: Float
            var earCoverage: [String: Float]
            var earDistanceMM: [String: Float]
        }
    }

    /// A saved scan in the list.
    struct Entry: Identifiable {
        var url: URL
        var session: Session
        var id: URL { url }
    }

    /// Posted after a scan is saved or deleted.
    static let didChange = Notification.Name("SavedScansDidChange")

    static func folder(_ session: FitSession) throws -> URL { try ScanExporter.directory(session) }

    static func isSaved(_ session: FitSession) -> Bool {
        (try? folder(session)).map { FileManager.default.fileExists(atPath: $0.appendingPathComponent("session.json").path) } ?? false
    }

    /// Saves (or re-saves, with the current adjustments) a finished scan.
    static func save(_ fit: FitSession) throws {
        let dir = try folder(fit)
        let a = fit.analyzer
        var overrides: [String: Vec3] = [:]
        for l in AdjustableLandmark.allCases { if let p = fit.overrides[l] { overrides[l.rawValue] = p } }
        let session = Session(
            name: ScanExporter.name(fit), createdAt: fit.createdAt, savedAt: Date(), isDemo: fit.isDemo,
            eyeLeft: a.eyeCenterLeft, eyeRight: a.eyeCenterRight, analysis: a.inputs, overrides: overrides,
            diagnostics: fit.diagnostics,
            headModel: fit.modelFit.map { m in
                Session.HeadModel(identity: m.identity, headScale: m.headScale, source: m.source, prior: m.prior, rmsMM: m.rmsMM,
                                  earCoverage: Dictionary(uniqueKeysWithValues: m.earCoverage.map { ($0.key.rawValue, $0.value) }),
                                  earDistanceMM: Dictionary(uniqueKeysWithValues: m.earDistanceMM.map { ($0.key.rawValue, $0.value) }))
            },
            capture: fit.captureDirectory?.lastPathComponent)
        try write(fit.mesh, to: dir.appendingPathComponent("scan.mesh"))
        if let m = fit.modelFit { try write(m.mesh, to: dir.appendingPathComponent("head.mesh")) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(session).write(to: dir.appendingPathComponent("session.json"), options: .atomic)
        DispatchQueue.main.async { NotificationCenter.default.post(name: didChange, object: nil) }
    }

    /// Saved scans, newest first.
    static func list() -> [Entry] {
        guard let docs = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) else { return [] }
        let root = docs.appendingPathComponent("Scans", isDirectory: true)
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return dirs.compactMap { dir in
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("session.json")),
                  let s = try? decoder.decode(Session.self, from: data) else { return nil }
            return Entry(url: dir, session: s)
        }
        .sorted { $0.session.createdAt > $1.session.createdAt }
    }

    /// Reopens a saved scan as it was saved: same measurements, head model and adjustments. The scan-only
    /// analysis is recomputed from the saved surface (a few seconds); the fits are not run again.
    static func load(_ entry: Entry) throws -> FitSession {
        let s = entry.session
        let mesh = try read(entry.url.appendingPathComponent("scan.mesh"))
        let analyzer = FitAnalyzer(mesh: mesh, eyeCenterLeft: s.eyeLeft, eyeCenterRight: s.eyeRight)
        analyzer.apply(s.analysis)
        var modelFit: ModelFitSummary?
        if let h = s.headModel, let head = try? read(entry.url.appendingPathComponent("head.mesh")) {
            func sides(_ d: [String: Float]) -> [HeadSide: Float] {
                Dictionary(uniqueKeysWithValues: d.compactMap { k, v in HeadSide(rawValue: k).map { ($0, v) } })
            }
            modelFit = ModelFitSummary(mesh: head, rmsMM: h.rmsMM, identity: h.identity, prior: h.prior, source: h.source,
                                       headScale: h.headScale, earCoverage: sides(h.earCoverage), earDistanceMM: sides(h.earDistanceMM))
        }
        let session = FitSession(mesh: mesh, analyzer: analyzer, isDemo: s.isDemo, diagnostics: s.diagnostics,
                                 modelFit: modelFit, createdAt: s.createdAt)
        var overrides = FitOverrides()
        for l in AdjustableLandmark.allCases { overrides[l] = s.overrides[l.rawValue] }
        session.overrides = overrides
        if let c = s.capture {
            let dir = CaptureReplay.directory.appendingPathComponent(c, isDirectory: true)
            if FileManager.default.fileExists(atPath: dir.path) { session.captureDirectory = dir }
        }
        return session
    }

    /// Removes a saved scan with its exports and frames.
    static func delete(_ entry: Entry) throws {
        try FileManager.default.removeItem(at: entry.url)
        NotificationCenter.default.post(name: didChange, object: nil)
    }

    // MARK: - Mesh files

    private static let magic: UInt32 = 0x4D_46_46_31 // "1FFM"

    private static func write(_ m: TriMesh, to url: URL) throws {
        var data = Data()
        func append<T>(_ values: [T]) { values.withUnsafeBytes { data.append(contentsOf: $0) } }
        append([magic, UInt32(m.vertices.count), UInt32(m.triangles.count), UInt32(m.normals.count == m.vertices.count ? 1 : 0)])
        append(m.vertices.flatMap { [$0.x, $0.y, $0.z] })
        if m.normals.count == m.vertices.count { append(m.normals.flatMap { [$0.x, $0.y, $0.z] }) }
        append(m.triangles.flatMap { [$0.x, $0.y, $0.z] })
        try (data as NSData).compressed(using: .lzfse).write(to: url, options: .atomic)
    }

    private static func read(_ url: URL) throws -> TriMesh {
        let data = try (NSData(contentsOf: url).decompressed(using: .lzfse)) as Data
        return try data.withUnsafeBytes { raw -> TriMesh in
            func u32(_ i: Int) -> UInt32 { raw.loadUnaligned(fromByteOffset: 4 * i, as: UInt32.self) }
            guard raw.count >= 16, u32(0) == magic else { throw CocoaError(.fileReadCorruptFile) }
            let nv = Int(u32(1)), nt = Int(u32(2)), hasNormals = u32(3) == 1
            guard raw.count == 16 + 12 * nv * (hasNormals ? 2 : 1) + 12 * nt else { throw CocoaError(.fileReadCorruptFile) }
            var o = 16
            func vecs(_ n: Int) -> [Vec3] {
                let f = raw.baseAddress!.advanced(by: o).assumingMemoryBound(to: Float.self)
                defer { o += 12 * n }
                return (0..<n).map { Vec3(f[3 * $0], f[3 * $0 + 1], f[3 * $0 + 2]) }
            }
            var m = TriMesh()
            m.vertices = vecs(nv)
            if hasNormals { m.normals = vecs(nv) }
            let t = raw.baseAddress!.advanced(by: o).assumingMemoryBound(to: UInt32.self)
            m.triangles = (0..<nt).map { SIMD3(t[3 * $0], t[3 * $0 + 1], t[3 * $0 + 2]) }
            if !hasNormals { m.computeNormals() }
            return m
        }
    }
}
