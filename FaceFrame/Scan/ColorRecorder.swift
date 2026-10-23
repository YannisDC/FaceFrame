import CoreImage
import CoreVideo
import Foundation
import simd

/// Saves the colour camera frames of a scan next to the depth capture (Captures/<name>/):
/// `color_00012.jpg` plus one line per image in `color_frames.jsonl` with the capture time, the
/// colour intrinsics and resolution and ARKit's raw head-from-camera pose at that time. Intended for
/// fitting the head model to image landmarks / silhouettes offline.
///
/// ARKit reuses its pixel buffers, so frames are copied on the caller's queue and encoded later.
final class ColorRecorder {
    struct Entry: Codable {
        var index: Int
        var file: String
        var time: Double
        /// Column-major 3x3 intrinsics of the colour image (pixels).
        var intrinsics: [Float]
        var width: Int
        var height: Int
        /// ARKit head-from-camera pose (column-major 4x4) at `time`; nil when the face was not tracked.
        /// Camera convention: x right, y up, looking down -z. Apply the scan's ARKit correction for metric use.
        var arPose: [Float]?
    }

    let directory: URL
    private let queue = DispatchQueue(label: "faceframe.color-recorder", qos: .utility)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let maxFrames: Int
    private var count = 0
    private var log: FileHandle?

    init(directory: URL, maxFrames: Int = 300) throws {
        self.directory = directory
        self.maxFrames = maxFrames
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let logURL = directory.appendingPathComponent("color_frames.jsonl")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        log = try FileHandle(forWritingTo: logURL)
    }

    deinit { try? log?.close() }

    func append(_ buffer: CVPixelBuffer, time: Double, intrinsics: simd_float3x3, arPose: simd_float4x4?) {
        guard count < maxFrames, let copy = Self.copy(buffer) else { return }
        let index = count
        count += 1
        let name = String(format: "color_%05d.jpg", index)
        let entry = Entry(index: index, file: name, time: time,
                          intrinsics: [intrinsics.columns.0, intrinsics.columns.1, intrinsics.columns.2].flatMap { [$0.x, $0.y, $0.z] },
                          width: CVPixelBufferGetWidth(copy), height: CVPixelBufferGetHeight(copy),
                          arPose: arPose.map { [$0.columns.0, $0.columns.1, $0.columns.2, $0.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] } })
        let url = directory.appendingPathComponent(name)
        queue.async { [context, log] in
            let image = CIImage(cvPixelBuffer: copy)
            if let space = CGColorSpace(name: CGColorSpace.sRGB),
               let jpeg = context.jpegRepresentation(of: image, colorSpace: space,
                                                     options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.85]) {
                try? jpeg.write(to: url)
            }
            if var line = try? JSONEncoder().encode(entry) {
                line.append(0x0A)
                log?.write(line)
            }
        }
    }

    /// Deep copy of a (possibly planar) pixel buffer.
    private static func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        var dst: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(nil, CVPixelBufferGetWidth(src), CVPixelBufferGetHeight(src),
                                  CVPixelBufferGetPixelFormatType(src), attrs, &dst) == kCVReturnSuccess, let dst else { return nil }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        if CVPixelBufferIsPlanar(src) {
            for p in 0..<CVPixelBufferGetPlaneCount(src) {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, p), let d = CVPixelBufferGetBaseAddressOfPlane(dst, p) else { return nil }
                let rows = CVPixelBufferGetHeightOfPlane(src, p)
                let sb = CVPixelBufferGetBytesPerRowOfPlane(src, p), db = CVPixelBufferGetBytesPerRowOfPlane(dst, p)
                for r in 0..<rows { memcpy(d + r * db, s + r * sb, min(sb, db)) }
            }
        } else {
            guard let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            let sb = CVPixelBufferGetBytesPerRow(src), db = CVPixelBufferGetBytesPerRow(dst)
            for r in 0..<CVPixelBufferGetHeight(src) { memcpy(d + r * db, s + r * sb, min(sb, db)) }
        }
        return dst
    }
}
