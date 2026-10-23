import Foundation
import simd

/// Sparse, block-allocated truncated signed distance volume expressed in the head (face anchor)
/// coordinate frame. Each depth frame is fused with the point-based "ray band" integrator: every
/// depth sample updates the voxels along its viewing ray within the truncation band. Positive
/// distances are in front of the surface (free space), negative distances are inside the head.
///
/// All methods must be called from a single serial queue.
final class TSDFVolume {
    static let blockShift: Int32 = 3
    static let blockMask: Int32 = 7
    static let voxelsPerBlock = 512

    let voxelSize: Float
    let origin: Vec3
    let dims: SIMD3<Int32>
    let blockDims: SIMD3<Int32>
    /// Free-space side of the truncation band (meters). Values are normalized by this.
    let truncFront: Float
    /// Occupied side of the truncation band (meters). Slightly shorter than the free-space side so
    /// thin parts (ear helix) survive; much shorter biases the alignment (measured on synthetic data).
    let truncBack: Float
    let maxWeight: UInt8

    private(set) var blockLookup: [Int32]
    private(set) var blockCoords: [SIMD3<Int32>] = []
    private(set) var tsdf: [Int16] = []
    private(set) var weight: [UInt8] = []
    private(set) var integratedFrames = 0

    init(origin: Vec3, extent: Vec3, voxelSize: Float, truncFront: Float = 0.003, truncBack: Float = 0.0025, maxWeight: UInt8 = 250) {
        self.origin = origin
        self.voxelSize = voxelSize
        self.truncFront = truncFront
        self.truncBack = truncBack
        self.maxWeight = maxWeight
        let d = SIMD3<Int32>((extent / voxelSize).rounded(.up))
        let bd = (d &+ TSDFVolume.blockMask) &>> TSDFVolume.blockShift
        self.blockDims = bd
        self.dims = bd &<< TSDFVolume.blockShift
        self.blockLookup = [Int32](repeating: -1, count: Int(bd.x) * Int(bd.y) * Int(bd.z))
    }

    var allocatedBlocks: Int { blockCoords.count }

    // MARK: - Indexing

    @inline(__always) private func blockSlot(_ b: SIMD3<Int32>) -> Int {
        Int((b.z * blockDims.y + b.y) * blockDims.x + b.x)
    }

    @inline(__always) func voxelCoord(of p: Vec3) -> SIMD3<Int32> {
        SIMD3<Int32>(((p - origin) / voxelSize).rounded(.down))
    }

    @inline(__always) func voxelCenter(_ v: SIMD3<Int32>) -> Vec3 {
        origin + (Vec3(v) + 0.5) * voxelSize
    }

    @inline(__always) func inBounds(_ v: SIMD3<Int32>) -> Bool {
        v.x >= 0 && v.y >= 0 && v.z >= 0 && v.x < dims.x && v.y < dims.y && v.z < dims.z
    }

    /// Linear storage index of a voxel, or -1 when its block is not allocated / out of bounds.
    @inline(__always) func storageIndex(_ v: SIMD3<Int32>) -> Int {
        guard inBounds(v) else { return -1 }
        let b = blockLookup[blockSlot(v &>> TSDFVolume.blockShift)]
        if b < 0 { return -1 }
        let l = v & TSDFVolume.blockMask
        return Int(b) << 9 | Int((l.z << 6) | (l.y << 3) | l.x)
    }

    private func allocateBlock(containing v: SIMD3<Int32>) {
        guard inBounds(v) else { return }
        let b = v &>> TSDFVolume.blockShift
        let slot = blockSlot(b)
        if blockLookup[slot] >= 0 { return }
        blockLookup[slot] = Int32(blockCoords.count)
        blockCoords.append(b)
        tsdf.append(contentsOf: repeatElement(Int16.max, count: TSDFVolume.voxelsPerBlock))
        weight.append(contentsOf: repeatElement(0, count: TSDFVolume.voxelsPerBlock))
    }

    /// Overwrites a whole block (`base` = its first voxel) with (normalized value, weight) pairs.
    func setBlock(_ base: SIMD3<Int32>, values: [(Int16, UInt8)]) {
        allocateBlock(containing: base)
        let start = storageIndex(base)
        guard start >= 0, values.count == TSDFVolume.voxelsPerBlock else { return }
        for (i, v) in values.enumerated() {
            tsdf[start + i] = v.0
            weight[start + i] = v.1
        }
    }

    /// Normalized stored value -> meters.
    @inline(__always) func distance(fromStored s: Int16) -> Float {
        Float(s) / 32767 * truncFront
    }

    // MARK: - Integration

    private var blockStamp: [Int32] = []
    /// Free-space carving: minimum pixel quality (1–4) and whether carving uses the pixel's full weight.
    static var carveMinQuality: Int32 = 3
    static var carveFullWeight = false
    /// How far in front of the observed surface (metres) carving looks for allocated noise blocks.
    /// Off by default: on a real capture it removed floaters but thinned the face surface.
    static var carveReach: Float = 0
    /// Robust fusion: established voxels (weight ≥ robustMinWeight) ignore observations deviating by
    /// more than robustMaxDeviation (normalized by truncFront; 0.75 ≈ 3 mm). Off by default: on a real
    /// capture it locked in early errors and damaged the nose (see Tools/Replay experiments).
    static var robustMinWeight: Int32 = .max
    static var robustMaxDeviation: Float = 0.75

    /// Fuses one depth frame with projective TSDF integration.
    /// - Parameters:
    ///   - depth: the depth image (metres, pinhole model, x right / y down / z forward).
    ///   - headFromCamera: pose of the camera in the volume frame, ARKit camera convention
    ///     (x right, y up, looking down -z).
    ///   - surfacePoints: depth samples already in the volume frame; they decide which blocks are
    ///     allocated and updated (keep only points that belong to the head).
    func integrate(depth: DepthImage, headFromCamera: simd_float4x4, surfacePoints: [Vec3], pixelWeights: [UInt8]? = nil) {
        let o = headFromCamera.translation
        let frameId = Int32(integratedFrames + 1)
        var touched: [Int32] = []
        let blockStep = voxelSize * Float(1 << TSDFVolume.blockShift) * 0.5
        for p in surfacePoints {
            let d = simd_normalize(p - o)
            var t = -truncFront
            while true {
                let v = voxelCoord(of: p + d * min(t, truncBack))
                if inBounds(v) {
                    allocateBlock(containing: v)
                    let id = blockLookup[blockSlot(v &>> TSDFVolume.blockShift)]
                    if blockStamp.count <= Int(id) {
                        blockStamp.append(contentsOf: repeatElement(0, count: Int(id) + 1 - blockStamp.count))
                    }
                    if blockStamp[Int(id)] != frameId {
                        blockStamp[Int(id)] = frameId
                        touched.append(id)
                    }
                }
                if t >= truncBack { break }
                t += blockStep
            }
        }

        // Blocks already allocated in front of the surface (noise floating in free space) are not
        // next to this frame's surface; walk back along the rays so carving can reach and clear them.
        if TSDFVolume.carveReach > 0 {
            var k = 0
            while k < surfacePoints.count {
                let p = surfacePoints[k]
                k += 3
                let d = simd_normalize(p - o)
                var t = truncFront + blockStep
                while t < TSDFVolume.carveReach {
                    let v = voxelCoord(of: p - d * t)
                    t += blockStep
                    guard inBounds(v) else { continue }
                    let id = blockLookup[blockSlot(v &>> TSDFVolume.blockShift)]
                    if id < 0 { continue }
                    if blockStamp.count <= Int(id) {
                        blockStamp.append(contentsOf: repeatElement(0, count: Int(id) + 1 - blockStamp.count))
                    }
                    if blockStamp[Int(id)] != frameId {
                        blockStamp[Int(id)] = frameId
                        touched.append(id)
                    }
                }
            }
        }

        let camFromHead = headFromCamera.inverse
        let fx = depth.fx, fy = depth.fy, cx = depth.cx, cy = depth.cy
        let w = depth.width, h = depth.height
        let invTrunc = 1 / truncFront
        let maxW = Int32(maxWeight)
        let edgeJump: Float = 0.008
        let blockCoords = self.blockCoords
        let tf = truncFront
        let weights = pixelWeights ?? []
        let hasWeights = !weights.isEmpty
        // Camera-space position of a voxel = p0 + lx*ax + ly*ay + lz*az (affine in voxel index).
        let ax = camFromHead.transformDirection(Vec3(voxelSize, 0, 0))
        let ay = camFromHead.transformDirection(Vec3(0, voxelSize, 0))
        let az = camFromHead.transformDirection(Vec3(0, 0, voxelSize))
        let origin = self.origin, vs = self.voxelSize
        let tb = truncBack
        weights.withUnsafeBufferPointer { qBuf in
        depth.values.withUnsafeBufferPointer { dBuf in
            tsdf.withUnsafeMutableBufferPointer { tsdfBuf in
                weight.withUnsafeMutableBufferPointer { wBuf in
                    let dPtr = dBuf.baseAddress!, tsdfPtr = tsdfBuf.baseAddress!, wPtr = wBuf.baseAddress!
                    let chunks = max(1, min(touched.count, 64))
                    touched.withUnsafeBufferPointer { touchedBuf in
                        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                            var k = chunk
                            while k < touchedBuf.count {
                                let id = touchedBuf[k]
                                k += chunks
                                let base = blockCoords[Int(id)] &<< TSDFVolume.blockShift
                                let p0 = camFromHead.transformPoint(origin + (Vec3(base) + 0.5) * vs)
                                for li in 0..<TSDFVolume.voxelsPerBlock {
                                    let pc = p0 + Float(li & 7) * ax + Float((li >> 3) & 7) * ay + Float(li >> 6) * az
                                    let z = -pc.z
                                    if z < 0.05 { continue }
                                    let xn = pc.x / z, yn = -pc.y / z
                                    let u = Int((fx * xn + cx).rounded()), r = Int((fy * yn + cy).rounded())
                                    if u < 1 || r < 1 || u >= w - 1 || r >= h - 1 { continue }
                                    let i = r * w + u
                                    let d = dPtr[i]
                                    if d <= 0 { continue }
                                    var q: Int32 = 1
                                    if hasWeights {
                                        q = Int32(qBuf[i])
                                        if q == 0 { continue }
                                    } else if abs(dPtr[i - 1] - d) > edgeJump || abs(dPtr[i + 1] - d) > edgeJump
                                        || abs(dPtr[i - w] - d) > edgeJump || abs(dPtr[i + w] - d) > edgeJump {
                                        continue // depth discontinuity: the nearest pixel is unreliable
                                    }
                                    let sdf = (d - z) * (1 + xn * xn + yn * yn).squareRoot()
                                    if sdf < -tb { continue }
                                    var val = sdf * invTrunc
                                    if sdf > tf {
                                        // Free space in front of the surface: carving clears noise, but only from
                                        // trustworthy pixels (not near a depth edge, surface seen squarely) and with
                                        // weight 1, so one bad view cannot erase a well-observed surface such as
                                        // the nose. Unrestricted carving from silhouette pixels erased nose flanks.
                                        if q < TSDFVolume.carveMinQuality && hasWeights { continue }
                                        val = 1
                                        if !TSDFVolume.carveFullWeight { q = 1 }
                                    }
                                    let idx = Int(id) << 9 | li
                                    let wOld = Int32(wPtr[idx])
                                    let old = wOld == 0 ? 0 : Float(tsdfPtr[idx]) / 32767
                                    // Robust update: once a voxel is well established, an observation that
                                    // disagrees by more than ~3 mm is an outlier (depth noise, residual pose
                                    // error) and is ignored. Averaging such outliers in created a second,
                                    // buried "skin" a few millimetres under the real one on device scans.
                                    if wOld >= TSDFVolume.robustMinWeight && abs(val - old) > TSDFVolume.robustMaxDeviation { continue }
                                    let fused = (old * Float(wOld) + val * Float(q)) / Float(wOld + q)
                                    tsdfPtr[idx] = Int16(max(-1, min(1, fused)) * 32767)
                                    wPtr[idx] = UInt8(min(wOld + q, maxW))
                                }
                            }
                        }
                    }
                }
            }
        }
        }
        integratedFrames += 1
    }

    // MARK: - Sampling

    /// Trilinearly interpolated signed distance (meters) and its gradient at `p`, or nil when any of
    /// the 8 neighbouring voxels has not been observed.
    func sample(_ p: Vec3, minWeight: UInt8 = 1) -> (distance: Float, gradient: Vec3)? {
        let g = (p - origin) / voxelSize - 0.5
        let base = g.rounded(.down)
        let f = g - base
        let b = SIMD3<Int32>(base)
        var c = SIMD8<Float>()
        for i in 0..<8 {
            let v = b &+ SIMD3<Int32>(Int32(i & 1), Int32((i >> 1) & 1), Int32((i >> 2) & 1))
            let idx = storageIndex(v)
            if idx < 0 || weight[idx] < minWeight { return nil }
            c[i] = distance(fromStored: tsdf[idx])
        }
        let fx = f.x, fy = f.y, fz = f.z
        let c00 = c[0] * (1 - fx) + c[1] * fx
        let c10 = c[2] * (1 - fx) + c[3] * fx
        let c01 = c[4] * (1 - fx) + c[5] * fx
        let c11 = c[6] * (1 - fx) + c[7] * fx
        let c0 = c00 * (1 - fy) + c10 * fy
        let c1 = c01 * (1 - fy) + c11 * fy
        let value = c0 * (1 - fz) + c1 * fz

        let gx = ((c[1] - c[0]) * (1 - fy) * (1 - fz) + (c[3] - c[2]) * fy * (1 - fz)
                + (c[5] - c[4]) * (1 - fy) * fz + (c[7] - c[6]) * fy * fz)
        let gy = ((c10 - c00) * (1 - fz) + (c11 - c01) * fz)
        let gz = c1 - c0
        return (value, Vec3(gx, gy, gz) / voxelSize)
    }
}
