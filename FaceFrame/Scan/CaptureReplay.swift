import CoreGraphics
import Foundation
import ImageIO
import simd

/// Runs a recorded raw capture (Documents/Captures/<name>, see ScanController) through the same fusion,
/// meshing, head model fits and measurements as a live scan: on the phone ("Replay a capture" on the
/// home screen) and on a Mac (Tools/replay.sh).
enum CaptureReplay {
    /// Recorded captures: Documents/Captures (Files › FaceFrame › Captures).
    static var directory: URL {
        let docs = (try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return docs.appendingPathComponent("Captures", isDirectory: true)
    }

    /// The folder for a new recording.
    static func newCapture() -> URL {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return directory.appendingPathComponent("capture_\(fmt.string(from: Date()))", isDirectory: true)
    }

    /// Recorded captures, newest first.
    static func captures() -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        return items.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    // MARK: - Replay

    /// Writes the reconstruction's detailed log next to the scans (Files › FaceFrame › Scans › replay_<capture>.log).
    static func run(_ dir: URL, progress: @escaping (String) -> Void = { _ in }) throws -> FitSession? {
        var logText = ""
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let logURL = docs.appendingPathComponent("Scans", isDirectory: true).appendingPathComponent("replay_\(dir.lastPathComponent).log")
        try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let start = Date()
        HeadReconstruction.log = { line in
            logText += String(format: "%6.1f s  ", Date().timeIntervalSince(start)) + line + "\n"
            try? logText.write(to: logURL, atomically: true, encoding: .utf8)
        }
        defer { HeadReconstruction.log = { _ in } }
        let progress: (String) -> Void = { message in HeadReconstruction.log("· " + message); progress(message) }
        progress("Loading capture…")
        let frames = try CaptureRecorder.loadFrames(from: dir)
        let engine = FusionEngine()
        for (i, f) in frames.enumerated() {
            if i % 25 == 0 { progress("Fusing depth frames… \(i)/\(frames.count)") }
            _ = engine.process(f)
        }
        guard let volume = engine.volume, let eyes = engine.eyes else { return nil }
        let pupils = engine.rawEyes.map { PupilInputs(dir, arkitEyes: $0, correction: engine.correction) }
        let photos = engine.rawEyes.map { HeadReconstruction.PhotoInput(frames: loadPhotoFrames(dir), arkitEyes: $0) }
        let session = HeadReconstruction.build(volume: volume, eyeLeft: eyes.left, eyeRight: eyes.right, diagnostics: engine.diagnostics,
                                               pupils: pupils, photos: photos, progress: progress)
        session?.captureDirectory = dir
        if let s = session {
            let r = s.result
            HeadReconstruction.log(String(format: "PD %.1f mm (%@), ear seats L %@ / R %@", r.pd * 1000, r.pupilsFromImages ? "images" : "ARKit",
                                          r.earLeft.map { String(format: "%.1f, %.1f, %.1f", $0.seat.x * 1000, $0.seat.y * 1000, $0.seat.z * 1000) } ?? "-",
                                          r.earRight.map { String(format: "%.1f, %.1f, %.1f", $0.seat.x * 1000, $0.seat.y * 1000, $0.seat.z * 1000) } ?? "-"))
            if let m = s.modelFit {
                HeadReconstruction.log(String(format: "head model: fitted to the %@, scale %.3f", m.source, m.headScale))
            }
        } else {
            HeadReconstruction.log("no usable surface")
        }
        return session
    }

    private static func PupilInputs(_ dir: URL, arkitEyes: (left: Vec3, right: Vec3), correction: PoseCorrection) -> HeadReconstruction.PupilInput {
        HeadReconstruction.PupilInput(samples: loadPupilSamples(dir, arkitEyes: arkitEyes), arkitEyes: arkitEyes, correction: correction)
    }

    private static func entries(_ dir: URL) -> [ColorRecorder.Entry] {
        guard let text = try? String(contentsOf: dir.appendingPathComponent("color_frames.jsonl"), encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap { try? JSONDecoder().decode(ColorRecorder.Entry.self, from: Data($0.utf8)) }
    }

    private static func frame(_ dir: URL, _ e: ColorRecorder.Entry) -> (CGImage, simd_float3x3, simd_float4x4)? {
        guard let a = e.arPose, let src = CGImageSourceCreateWithURL(dir.appendingPathComponent(e.file) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let pose = simd_float4x4(SIMD4(a[0], a[1], a[2], a[3]), SIMD4(a[4], a[5], a[6], a[7]), SIMD4(a[8], a[9], a[10], a[11]), SIMD4(a[12], a[13], a[14], a[15]))
        let k = simd_float3x3(SIMD3(e.intrinsics[0], e.intrinsics[1], e.intrinsics[2]), SIMD3(e.intrinsics[3], e.intrinsics[4], e.intrinsics[5]),
                              SIMD3(e.intrinsics[6], e.intrinsics[7], e.intrinsics[8]))
        return (img, k, pose)
    }

    /// Near-frontal colour frames for measuring the pupils, as a live scan keeps them.
    static func loadPupilSamples(_ dir: URL, arkitEyes: (left: Vec3, right: Vec3)) -> [PupilEstimator.Sample] {
        entries(dir).compactMap { e in
            guard let a = e.arPose else { return nil }
            let t = Vec3(a[12], a[13], a[14])
            let toCam = simd_normalize(t - (arkitEyes.left + arkitEyes.right) / 2)
            let yaw = atan2(toCam.x, toCam.z) * 180 / .pi, pitch = asin(toCam.y) * 180 / .pi
            guard abs(yaw) < 12, abs(pitch) < 12, let (img, k, pose) = frame(dir, e) else { return nil }
            return PupilEstimator.Sample(image: img, intrinsics: k, arPose: pose)
        }
    }

    /// Every colour frame with an ARKit pose, for the photo fit to choose its views from.
    static func loadPhotoFrames(_ dir: URL) -> [PhotoObserver.Frame] {
        entries(dir).compactMap { e in frame(dir, e).map { PhotoObserver.Frame(image: $0.0, intrinsics: $0.1, arPose: $0.2) } }
    }
}
