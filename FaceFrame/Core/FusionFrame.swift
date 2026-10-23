import Foundation
import simd

/// Everything the fusion engine needs from one TrueDepth frame. Platform independent so captures
/// recorded on the phone can be replayed on a Mac with exactly the same code.
struct FusionFrame {
    /// Depth capture time (seconds, ARKit clock).
    var time: Double
    var width: Int
    var height: Int
    /// Metres, raw sensor orientation (no mirroring applied); 0 = no measurement.
    var depth: [Float]
    var calibrationK: simd_float3x3?
    var calibrationReference: SIMD2<Float>?
    /// Colour camera intrinsics and resolution (fallback when the depth has no calibration data).
    var intrinsics: simd_float3x3
    var imageResolution: SIMD2<Float>
    /// ARKit head-from-camera pose interpolated at `time`; nil when ARKit is not tracking the face.
    var arPose: simd_float4x4?
    /// Head angular speed relative to the camera around `time` (deg/s, translation folded in).
    var motion: Float
    /// Eye centres in the face frame.
    var leftEye: Vec3?
    var rightEye: Vec3?
    /// ARKit face mesh vertices in the face frame (only sent while calibrating).
    var faceVertices: [Vec3]?
}

// MARK: - Binary capture format (LZFSE-compressed: header length, JSON header, Float16 depth, Float32 vertices)

extension FusionFrame {
    private struct Header: Codable {
        var time: Double
        var width: Int
        var height: Int
        var calibrationK: [Float]?
        var calibrationReference: [Float]?
        var intrinsics: [Float]
        var imageResolution: [Float]
        var arPose: [Float]?
        var motion: Float
        var leftEye: [Float]?
        var rightEye: [Float]?
        var vertexCount: Int
    }

    enum CaptureError: Error { case corrupt }

    func encoded() throws -> Data {
        let header = Header(time: time, width: width, height: height,
                            calibrationK: calibrationK.map(Self.flat), calibrationReference: calibrationReference.map { [$0.x, $0.y] },
                            intrinsics: Self.flat(intrinsics), imageResolution: [imageResolution.x, imageResolution.y],
                            arPose: arPose.map(Self.flat), motion: motion,
                            leftEye: leftEye.map { [$0.x, $0.y, $0.z] }, rightEye: rightEye.map { [$0.x, $0.y, $0.z] },
                            vertexCount: faceVertices?.count ?? 0)
        let json = try JSONEncoder().encode(header)
        var data = Data()
        var n = UInt32(json.count).littleEndian
        withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }
        data.append(json)
        let half = depth.map { Float16($0) }
        half.withUnsafeBytes { data.append(contentsOf: $0) }
        let verts = (faceVertices ?? []).flatMap { [$0.x, $0.y, $0.z] }
        verts.withUnsafeBytes { data.append(contentsOf: $0) }
        return try (data as NSData).compressed(using: .lzfse) as Data
    }

    static func decode(_ compressed: Data) throws -> FusionFrame {
        let data = try (compressed as NSData).decompressed(using: .lzfse) as Data
        guard data.count >= 4 else { throw CaptureError.corrupt }
        let n = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
        guard data.count >= 4 + n else { throw CaptureError.corrupt }
        let h = try JSONDecoder().decode(Header.self, from: data.subdata(in: 4..<(4 + n)))
        let depthBytes = h.width * h.height * 2
        guard data.count >= 4 + n + depthBytes + h.vertexCount * 12 else { throw CaptureError.corrupt }
        var depth = [Float](repeating: 0, count: h.width * h.height)
        data.subdata(in: (4 + n)..<(4 + n + depthBytes)).withUnsafeBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for i in 0..<depth.count { depth[i] = Float(p[i]) }
        }
        var verts: [Vec3]? = nil
        if h.vertexCount > 0 {
            let start = 4 + n + depthBytes
            verts = data.subdata(in: start..<(start + h.vertexCount * 12)).withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Float.self)
                return (0..<h.vertexCount).map { Vec3(p[$0 * 3], p[$0 * 3 + 1], p[$0 * 3 + 2]) }
            }
        }
        func v3(_ a: [Float]?) -> Vec3? { a.map { Vec3($0[0], $0[1], $0[2]) } }
        return FusionFrame(time: h.time, width: h.width, height: h.height, depth: depth,
                           calibrationK: h.calibrationK.map(Self.mat3), calibrationReference: h.calibrationReference.map { SIMD2($0[0], $0[1]) },
                           intrinsics: Self.mat3(h.intrinsics), imageResolution: SIMD2(h.imageResolution[0], h.imageResolution[1]),
                           arPose: h.arPose.map(Self.mat4), motion: h.motion,
                           leftEye: v3(h.leftEye), rightEye: v3(h.rightEye), faceVertices: verts)
    }

    private static func flat(_ m: simd_float3x3) -> [Float] {
        [m.columns.0, m.columns.1, m.columns.2].flatMap { [$0.x, $0.y, $0.z] }
    }
    private static func flat(_ m: simd_float4x4) -> [Float] {
        [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }
    private static func mat3(_ a: [Float]) -> simd_float3x3 {
        simd_float3x3(SIMD3(a[0], a[1], a[2]), SIMD3(a[3], a[4], a[5]), SIMD3(a[6], a[7], a[8]))
    }
    private static func mat4(_ a: [Float]) -> simd_float4x4 {
        simd_float4x4(SIMD4(a[0], a[1], a[2], a[3]), SIMD4(a[4], a[5], a[6], a[7]),
                      SIMD4(a[8], a[9], a[10], a[11]), SIMD4(a[12], a[13], a[14], a[15]))
    }
}

/// Writes frames to Documents/Captures/<name>/ on a background queue so a scan can be replayed
/// offline (see Tools/Replay).
final class CaptureRecorder {
    let directory: URL
    private let queue = DispatchQueue(label: "faceframe.recorder", qos: .utility)
    private var count = 0
    private let maxFrames: Int

    init(directory: URL, maxFrames: Int = 450) throws {
        self.directory = directory
        self.maxFrames = maxFrames
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func append(_ frame: FusionFrame) {
        guard count < maxFrames else { return }
        let index = count
        count += 1
        let url = directory.appendingPathComponent(String(format: "frame_%05d.ffd", index))
        queue.async {
            try? frame.encoded().write(to: url)
        }
    }

    /// Blocks until every queued frame is on disk.
    func flush() { queue.sync {} }

    static func loadFrames(from directory: URL) throws -> [FusionFrame] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "ffd" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return try files.map { try FusionFrame.decode(Data(contentsOf: $0)) }
    }
}
