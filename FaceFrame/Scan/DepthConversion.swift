import AVFoundation
import CoreVideo
import simd

enum DepthConversion {
    /// Copies metric depth out of an `AVDepthData` (raw sensor orientation, 0 = no measurement).
    static func fill(_ frame: inout FusionFrame, from depthData: AVDepthData) {
        let depth = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let buffer = depth.depthDataMap
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var values = [Float](repeating: 0, count: w * h)
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            for y in 0..<h {
                let row = (base + y * rowBytes).assumingMemoryBound(to: Float32.self)
                for x in 0..<w {
                    let d = row[x]
                    values[y * w + x] = d.isFinite && d > 0.1 && d < 1.2 ? d : 0
                }
            }
        }
        frame.width = w
        frame.height = h
        frame.depth = values
        if let c = depth.cameraCalibrationData {
            frame.calibrationK = c.intrinsicMatrix
            frame.calibrationReference = SIMD2(Float(c.intrinsicMatrixReferenceDimensions.width),
                                               Float(c.intrinsicMatrixReferenceDimensions.height))
        }
    }
}
