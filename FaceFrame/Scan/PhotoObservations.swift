import Accelerate
import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import simd
import Vision

/// Turns colour frames of a scan into what GNM's photo fit observes (`PhotoObservation`), as
/// demos/fit_head_to_images.py does on photos: the 68 iBUG landmarks (face-alignment's FAN), the
/// head's profile and hair outlines (MediaPipe's multiclass segmenter) and the ear helix
/// (Ear_Landmarker), all run as Core ML models (Tools/convert_photo_fit_models.py). Where the script
/// solves an initial head pose with PnP, the scan already knows it: ARKit's pose of each frame.
final class PhotoObserver {
    /// A colour frame as ARKit delivers it: sensor orientation, with its intrinsics and head pose.
    struct Frame {
        var image: CGImage
        /// Colour intrinsics in raw sensor pixels.
        var intrinsics: simd_float3x3
        /// Raw ARKit head-from-camera pose (camera: x right, y up, looking down −z).
        var arPose: simd_float4x4
    }

    let model: GNMPhotoModel
    private let landmarker: MLModel
    private let segmenter: MLModel
    private let earLandmarker: MLModel
    var log: (String) -> Void = { _ in }

    /// Where to load the models outside the app bundle (Mac replay tool): a folder of .mlpackage files.
    static var modelDirectoryOverride: URL?

    init(model: GNMPhotoModel) throws {
        self.model = model
        func load(_ name: String) throws -> MLModel {
            let config = MLModelConfiguration()
            #if targetEnvironment(simulator)
            config.computeUnits = .cpuOnly // the simulator's GPU path returned NaN for these models
            #endif
            if let dir = Self.modelDirectoryOverride {
                let compiled = try MLModel.compileModel(at: dir.appendingPathComponent("\(name).mlpackage"))
                return try MLModel(contentsOf: compiled, configuration: config)
            }
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { throw CocoaError(.fileNoSuchFile) }
            return try MLModel(contentsOf: url, configuration: config)
        }
        landmarker = try load("FaceLandmarks68")
        segmenter = try load("HeadSegmenter")
        earLandmarker = try load("EarLandmarks55")
    }

    // MARK: - Views

    /// The views GNM's photo fit wants: one frontal and the most turned frame to each side.
    static func selectViews(_ frames: [Frame], minSideYaw: Float = 45) -> [Frame] {
        func yaw(_ f: Frame) -> (Float, Float) {
            let t = f.arPose.columns.3
            return (atan2(t.x, t.z) * 180 / .pi, atan2(t.y, hypot(t.x, t.z)) * 180 / .pi)
        }
        guard let front = frames.min(by: { abs(yaw($0).0) + abs(yaw($0).1) < abs(yaw($1).0) + abs(yaw($1).1) }) else { return [] }
        var views = [front]
        if let left = frames.max(by: { yaw($0).0 < yaw($1).0 }), yaw(left).0 >= minSideYaw { views.append(left) }
        if let right = frames.min(by: { yaw($0).0 < yaw($1).0 }), yaw(right).0 <= -minSideYaw { views.append(right) }
        return views
    }

    /// An upright RGBA copy of a frame, with intrinsics and the pose in that image.
    struct Upright {
        var pixels: [UInt8]
        var width: Int
        var height: Int
        var image: CGImage
        var fx: Double, fy: Double, cx: Double, cy: Double
        /// Raw OpenCV camera (x right, y down, z forward) to upright OpenCV camera.
        var rotation: simd_double3x3
    }

    static func upright(_ frame: Frame) -> Upright? {
        let w = frame.image.width, h = frame.image.height
        guard let raw = rgba(frame.image) else { return nil }
        // OpenCV camera from ARKit's head frame.
        let camFromHead = frame.arPose.inverse
        let flip = simd_double3x3(diagonal: SIMD3(1, -1, -1))
        let up = flip * SIMD3<Double>(Double(camFromHead.columns.1.x), Double(camFromHead.columns.1.y), Double(camFromHead.columns.1.z))
        // Quarter turns clockwise that put the head's up axis towards the top of the image.
        let turns = [
            simd_double3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]),
            simd_double3x3(rows: [SIMD3(0, -1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)]),
            simd_double3x3(rows: [SIMD3(-1, 0, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1)]),
            simd_double3x3(rows: [SIMD3(0, 1, 0), SIMD3(-1, 0, 0), SIMD3(0, 0, 1)]),
        ]
        let k = (0..<4).min { (turns[$0] * up).y < (turns[$1] * up).y }!
        let K = frame.intrinsics
        let fx = Double(K.columns.0.x), fy = Double(K.columns.1.y), cx = Double(K.columns.2.x), cy = Double(K.columns.2.y)
        let W = Double(w), H = Double(h)
        let (ow, oh) = k % 2 == 1 ? (h, w) : (w, h)
        let (nfx, nfy, ncx, ncy): (Double, Double, Double, Double)
        switch k {
        case 1: (nfx, nfy, ncx, ncy) = (fy, fx, H - cy, cx)
        case 2: (nfx, nfy, ncx, ncy) = (fx, fy, W - cx, H - cy)
        case 3: (nfx, nfy, ncx, ncy) = (fy, fx, cy, W - cx)
        default: (nfx, nfy, ncx, ncy) = (fx, fy, cx, cy)
        }
        var out = [UInt8](repeating: 0, count: ow * oh * 4)
        var src = raw
        let constant: [UInt8] = [UInt8(kRotate0DegreesClockwise), UInt8(kRotate90DegreesClockwise),
                                 UInt8(kRotate180DegreesClockwise), UInt8(kRotate270DegreesClockwise)]
        src.withUnsafeMutableBytes { s in
            out.withUnsafeMutableBytes { d in
                var sb = vImage_Buffer(data: s.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                var db = vImage_Buffer(data: d.baseAddress, height: vImagePixelCount(oh), width: vImagePixelCount(ow), rowBytes: ow * 4)
                var bg: [UInt8] = [0, 0, 0, 0]
                vImageRotate90_ARGB8888(&sb, &db, constant[k], &bg, vImage_Flags(kvImageNoFlags))
            }
        }
        guard let image = cgImage(out, ow, oh) else { return nil }
        return Upright(pixels: out, width: ow, height: oh, image: image, fx: nfx, fy: nfy, cx: ncx, cy: ncy, rotation: turns[k])
    }

    // MARK: - Observation

    /// The photo fit's observation of one frame. `arkitEyes`: ARKit's eye centres in its face frame,
    /// to place GNM's eyes for the initial pose.
    func observe(_ frame: Frame, arkitEyes: (left: Vec3, right: Vec3)) -> PhotoObservation? {
        guard let up = Self.upright(frame) else { return nil }
        // Initial pose: ARKit's, GNM's eye midpoint on ARKit's.
        let camFromHead = frame.arPose.inverse
        let flip = simd_double3x3(diagonal: SIMD3(1, -1, -1))
        let rc = simd_double3x3(columns: (SIMD3<Double>(Double(camFromHead.columns.0.x), Double(camFromHead.columns.0.y), Double(camFromHead.columns.0.z)),
                                          SIMD3<Double>(Double(camFromHead.columns.1.x), Double(camFromHead.columns.1.y), Double(camFromHead.columns.1.z)),
                                          SIMD3<Double>(Double(camFromHead.columns.2.x), Double(camFromHead.columns.2.y), Double(camFromHead.columns.2.z))))
        let r0 = up.rotation * flip * rc
        let eyeA = SIMD3<Double>(Double((arkitEyes.left.x + arkitEyes.right.x) / 2), Double((arkitEyes.left.y + arkitEyes.right.y) / 2),
                                 Double((arkitEyes.left.z + arkitEyes.right.z) / 2))
        let eyeCam = up.rotation * flip * SIMD3<Double>(camFromHead.transformPointD(eyeA))
        let eyeG = model.eyeMidpoint
        let t0 = eyeCam - r0 * eyeG

        let guess = model.landmarkPositions(model.template).map { p -> SIMD2<Double> in
            let c = r0 * p + t0
            return SIMD2(up.fx * c.x / c.z + up.cx, up.fy * c.y / c.z + up.cy)
        }
        guard let box = faceBox(up, guess: guess), let landmarks = detectLandmarks(up, box: box),
              landmarks.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { log("no landmarks in a view"); return nil }
        var obs = PhotoObservation(width: up.width, height: up.height, fx: up.fx, fy: up.fy, cx: up.cx, cy: up.cy,
                                   landmarks: landmarks, initialRotation: r0, initialTranslation: t0)
        if let masks = segment(up, landmarks: landmarks) {
            obs.outlines = Self.extractOutlines(person: masks.person, skin: masks.skin, hair: masks.hair, width: up.width, height: up.height,
                                                landmarks: landmarks)
        }
        if PhotoFitter.isSideView(r0) {
            obs.earHelix = detectEar(up, rotation: r0, translation: t0)
        }
        return obs
    }

    // MARK: Face box and landmarks (face_alignment)

    /// The face box face_alignment's S3FD detector would give: Vision's face rectangle, mapped to S3FD's
    /// framing; if Vision finds no face (strongly turned heads), the box of GNM's landmarks under the
    /// initial pose. As [x1, y1, x2, y2], pixels.
    private func faceBox(_ up: Upright, guess: [SIMD2<Double>]) -> [Double]? {
        let request = VNDetectFaceRectanglesRequest()
        #if targetEnvironment(simulator)
        request.usesCPUOnly = true // the simulator has no Neural Engine for Vision's face models
        #endif
        let handler = VNImageRequestHandler(cgImage: up.image, options: [:])
        try? handler.perform([request])
        let gx = guess.map(\.x), gy = guess.map(\.y)
        let guessBox = [gx.min()!, gy.min()!, gx.max()!, gy.max()!]
        let gc = SIMD2((guessBox[0] + guessBox[2]) / 2, (guessBox[1] + guessBox[3]) / 2)
        let faces = (request.results ?? []).map { r -> [Double] in
            let b = r.boundingBox
            return [b.minX * Double(up.width), (1 - b.maxY) * Double(up.height), b.maxX * Double(up.width), (1 - b.minY) * Double(up.height)]
        }
        // The face nearest to where the head should be.
        if let f = faces.min(by: { simd_distance(SIMD2(($0[0] + $0[2]) / 2, ($0[1] + $0[3]) / 2), gc)
                                    < simd_distance(SIMD2(($1[0] + $1[2]) / 2, ($1[1] + $1[3]) / 2), gc) }),
           simd_distance(SIMD2((f[0] + f[2]) / 2, (f[1] + f[3]) / 2), gc) < 0.5 * (guessBox[2] - guessBox[0] + guessBox[3] - guessBox[1]) {
            log(String(format: "face box from Vision %.0f %.0f %.0f %.0f", f[0], f[1], f[2], f[3]))
            return Self.visionToS3FD(f)
        }
        log("no Vision face: box from the initial pose")
        return Self.landmarksToS3FD(guessBox)
    }

    /// Calibrated on the frames of a capture against face_alignment's S3FD boxes (Tools/Replay --calibrate-face-box).
    static var visionBoxScale = SIMD4<Double>(0, 0, 1, 1)
    static func visionToS3FD(_ b: [Double]) -> [Double] {
        let w = b[2] - b[0], h = b[3] - b[1]
        let cx = (b[0] + b[2]) / 2 + visionBoxScale.x * w, cy = (b[1] + b[3]) / 2 + visionBoxScale.y * h
        let nw = w * visionBoxScale.z, nh = h * visionBoxScale.w
        return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]
    }
    static func landmarksToS3FD(_ b: [Double]) -> [Double] {
        // The 68 landmarks span the jaw to the brows; S3FD's box is about as wide and reaches the forehead.
        let w = b[2] - b[0], h = b[3] - b[1]
        return [b[0] - 0.02 * w, b[1] - 0.15 * h, b[2] + 0.02 * w, b[3] + 0.02 * h]
    }

    /// face_alignment's get_landmarks_from_image for one box: crop, FAN (flip-averaged), heatmap decoding.
    private func detectLandmarks(_ up: Upright, box d: [Double]) -> [SIMD2<Double>]? {
        let center = SIMD2((d[2] + d[0]) / 2, (d[3] + d[1]) / 2 - (d[3] - d[1]) * 0.12)
        let scale = (d[2] - d[0] + d[3] - d[1]) / 195
        let h = 200 * scale
        // utils.crop: the image region [ul, br) resized to 256 x 256, zero outside.
        let ul = SIMD2((1 * h / 256 + center.x - h / 2).rounded(.towardZero), (1 * h / 256 + center.y - h / 2).rounded(.towardZero))
        let br = SIMD2((256 * h / 256 + center.x - h / 2).rounded(.towardZero), (256 * h / 256 + center.y - h / 2).rounded(.towardZero))
        guard let input = pixelBuffer(up, region: CGRect(x: ul.x, y: ul.y, width: br.x - ul.x, height: br.y - ul.y), size: 256, fill: 0),
              let out = try? landmarker.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])),
              let hm = out.featureValue(for: "heatmaps")?.multiArrayValue else { return nil }
        let res = 64
        let ptr = hm.dataPointer
        func at(_ j: Int, _ y: Int, _ x: Int) -> Float {
            let i = j * res * res + y * res + x
            switch hm.dataType {
            case .float16: return Float(ptr.assumingMemoryBound(to: Float16.self)[i])
            case .float32: return ptr.assumingMemoryBound(to: Float.self)[i]
            default: return Float(ptr.assumingMemoryBound(to: Double.self)[i])
            }
        }
        return (0..<68).map { j in
            var best = -Float.infinity, bx = 0, by = 0
            for y in 0..<res { for x in 0..<res { let v = at(j, y, x); if v > best { best = v; bx = x; by = y } } }
            var p = SIMD2(Double(bx + 1), Double(by + 1))
            if bx > 0, bx < 63, by > 0, by < 63 {
                let dx = at(j, by, bx + 1) - at(j, by, bx - 1), dy = at(j, by + 1, bx) - at(j, by - 1, bx)
                p += SIMD2(Double(dx.sign == .minus ? -1 : (dx == 0 ? 0 : 1)), Double(dy.sign == .minus ? -1 : (dy == 0 ? 0 : 1))) * 0.25
            }
            p -= 0.5
            // transform(..., invert=True) at resolution 64.
            return p * (h / Double(res)) + center - h / 2
        }
    }

    // MARK: Segmentation and outlines (segment_head, extract_outlines)

    private func segment(_ up: Upright, landmarks: [SIMD2<Double>]) -> (person: [Bool], skin: [Bool], hair: [Bool])? {
        let xs = landmarks.map(\.x), ys = landmarks.map(\.y)
        let center = SIMD2(xs.reduce(0, +) / 68, ys.reduce(0, +) / 68)
        let half = max(xs.max()! - xs.min()!, ys.max()! - ys.min()!) * 1.6
        let x0 = Int(max(center.x - half, 0)), y0 = Int(max(center.y - half, 0))
        let x1 = Int(min(center.x + half, Double(up.width))), y1 = Int(min(center.y + half, Double(up.height)))
        guard x1 > x0, y1 > y0,
              let input = pixelBuffer(up, region: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0), size: 256, fill: 0),
              let out = try? segmenter.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])),
              let conf = out.featureValue(for: "confidences")?.multiArrayValue else { return nil }
        let ptr = conf.dataPointer
        func at(_ i: Int) -> Float {
            conf.dataType == .float16 ? Float(ptr.assumingMemoryBound(to: Float16.self)[i]) : ptr.assumingMemoryBound(to: Float.self)[i]
        }
        var labels = [UInt8](repeating: 0, count: 256 * 256)
        for i in 0..<(256 * 256) {
            var best = 0
            for c in 1..<6 where at(i * 6 + c) > at(i * 6 + best) { best = c }
            labels[i] = UInt8(best)
        }
        let n = up.width * up.height
        var person = [Bool](repeating: false, count: n), skin = person, hair = person
        let cw = Double(x1 - x0), ch = Double(y1 - y0)
        for y in y0..<y1 {
            let sy = min(255, Int((Double(y - y0) + 0.5) * 256 / ch))
            for x in x0..<x1 {
                let sx = min(255, Int((Double(x - x0) + 0.5) * 256 / cw))
                let l = labels[sy * 256 + sx]
                let i = y * up.width + x
                person[i] = l != 0; skin[i] = l == 2 || l == 3; hair[i] = l == 1
            }
        }
        return (person, skin, hair)
    }

    static let outlineForeheadExtent = 0.6
    static let hairSearchExtent = 1.3

    /// The outermost person pixel in rows (or columns), NaN where none or not made of `surface`.
    static func outermost(person: [Bool], surface: [Bool], width: Int, height: Int, lines: Range<Int>, direction: Double,
                          inset: Int, alongColumns: Bool = false) -> [Double] {
        let count = alongColumns ? width : height, length = alongColumns ? height : width
        func idx(_ line: Int, _ pos: Int) -> Int { alongColumns ? pos * width + line : line * width + pos }
        var outline = [Double](repeating: .nan, count: count)
        for line in lines where line >= 0 && line < count {
            var position: Int?
            if direction > 0 { position = (0..<length).last { person[idx(line, $0)] } } else { position = (0..<length).first { person[idx(line, $0)] } }
            guard let p = position else { continue }
            let inner = max(0, min(length - 1, Int(Double(p) - direction * Double(inset))))
            if surface[idx(line, inner)] { outline[line] = Double(p) + 0.5 * direction }
        }
        return outline
    }

    static func extractOutlines(person: [Bool], skin: [Bool], hair: [Bool], width: Int, height: Int,
                                landmarks: [SIMD2<Double>]) -> [PhotoObservation.Outline] {
        let meanX = landmarks.map(\.x).reduce(0, +) / Double(landmarks.count)
        let direction: Double = landmarks[30].x > meanX ? 1 : -1
        let brows = landmarks[17..<27].map(\.y).min()!
        let chin = landmarks[8].y
        let faceHeight = chin - brows
        let inset = max(Int(0.01 * faceHeight), 1)
        let profileTop = Int(brows - outlineForeheadExtent * faceHeight)
        var outlines = [PhotoObservation.Outline(kind: .profile,
                                                 values: outermost(person: person, surface: skin, width: width, height: height,
                                                                   lines: profileTop..<(Int(chin) + 1), direction: direction, inset: inset),
                                                 direction: direction)]
        let search = hairSearchExtent * faceHeight
        let cols = max(Int(meanX - search), 0)..<(min(Int(meanX + search), width - 1) + 1)
        var headTop: Int?
        for y in 0..<max(Int(brows), 1) where headTop == nil {
            if cols.contains(where: { person[y * width + $0] }) { headTop = y }
        }
        guard let top = headTop else { return outlines }
        var topOutline = outermost(person: person, surface: hair, width: width, height: height, lines: cols, direction: -1, inset: inset, alongColumns: true)
        for i in topOutline.indices where topOutline[i] > brows { topOutline[i] = .nan }
        outlines.append(.init(kind: .hairTop, values: topOutline, direction: -1, alongColumns: true))
        outlines.append(.init(kind: .hairBack,
                              values: outermost(person: person, surface: hair, width: width, height: height,
                                                lines: top..<(Int(landmarks[33].y) + 1), direction: -direction, inset: inset),
                              direction: -direction))
        for side in [-1.0, 1.0] {
            outlines.append(.init(kind: .hairSide,
                                  values: outermost(person: person, surface: hair, width: width, height: height,
                                                    lines: top..<(Int(brows) + 1), direction: side, inset: inset),
                                  direction: side))
        }
        return outlines
    }

    // MARK: Ears (EarDetector)

    static let earCropOccupancy = 0.777
    static let earRefinements = 2
    static let earMinConfidence: Float = 0.2

    private func detectEar(_ up: Upright, rotation r: simd_double3x3, translation t: SIMD3<Double>) -> [SIMD2<Double>]? {
        let side = PhotoFitter(model: model).visibleEar(model.template, r, t)
        let pts = model.earVertices[side]!.map { i -> SIMD2<Double> in
            let c = r * SIMD3(Double(model.template[3 * i]), Double(model.template[3 * i + 1]), Double(model.template[3 * i + 2])) + t
            return SIMD2(up.fx * c.x / c.z + up.cx, up.fy * c.y / c.z + up.cy)
        }
        var center = (SIMD2(pts.map(\.x).min()!, pts.map(\.y).min()!) + SIMD2(pts.map(\.x).max()!, pts.map(\.y).max()!)) / 2
        var extent = max(pts.map(\.x).max()! - pts.map(\.x).min()!, pts.map(\.y).max()! - pts.map(\.y).min()!)
        var result: (landmarks: [SIMD2<Double>], confidence: Float)?
        for _ in 0...Self.earRefinements {
            let side = extent / Self.earCropOccupancy
            let size = (side).rounded()
            let x0 = (center.x - side / 2).rounded(), y0 = (center.y - side / 2).rounded()
            guard let input = pixelBuffer(up, region: CGRect(x: x0, y: y0, width: size, height: size), size: 192, fill: 128),
                  let out = try? earLandmarker.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])),
                  let lm = out.featureValue(for: "landmarks")?.multiArrayValue,
                  let cf = out.featureValue(for: "confidence")?.multiArrayValue else { return nil }
            let landmarks = (0..<55).map { SIMD2(lm[[0, $0, 0] as [NSNumber]].doubleValue, lm[[0, $0, 1] as [NSNumber]].doubleValue) * size + SIMD2(x0, y0) }
            let confidence = (0..<55).map { cf[[0, $0] as [NSNumber]].floatValue }.reduce(0, +) / 55
            result = (landmarks, confidence)
            let lx = landmarks.map(\.x), ly = landmarks.map(\.y)
            center = (SIMD2(lx.min()!, ly.min()!) + SIMD2(lx.max()!, ly.max()!)) / 2
            extent = max(lx.max()! - lx.min()!, ly.max()! - ly.min()!)
        }
        guard let r = result else { return nil }
        log(String(format: "%@ ear detected with confidence %.2f", side.rawValue, r.confidence))
        guard r.confidence >= Self.earMinConfidence else { return nil }
        return Array(r.landmarks[3..<18])
    }

    // MARK: - Pixels

    static func rgba(_ image: CGImage) -> [UInt8]? {
        let w = image.width, h = image.height
        var px = [UInt8](repeating: 0, count: w * h * 4)
        let ok = px.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? px : nil
    }

    static func cgImage(_ px: [UInt8], _ w: Int, _ h: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(px) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    /// The image region (top-left pixel coordinates) resized to size x size, `fill` grey outside the image.
    private func pixelBuffer(_ up: Upright, region: CGRect, size: Int, fill: UInt8) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess, let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        let g = CGFloat(fill) / 255
        ctx.setFillColor(red: g, green: g, blue: g, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.interpolationQuality = .high
        let s = CGFloat(size) / region.width, sy = CGFloat(size) / region.height
        ctx.concatenate(CGAffineTransform(a: s, b: 0, c: 0, d: sy, tx: -s * region.minX,
                                          ty: CGFloat(size) - sy * (CGFloat(up.height) - region.minY)))
        ctx.draw(up.image, in: CGRect(x: 0, y: 0, width: up.width, height: up.height))
        return pb
    }
}

extension GNMPhotoModel {
    /// Midpoint of the eyeballs' centres (the `left_eye` and `right_eye` groups) on the template.
    var eyeMidpoint: SIMD3<Double> {
        let idx = (0..<vertexCount).filter { (groups["left_eye"]?[$0] ?? false) || (groups["right_eye"]?[$0] ?? false) }
        return idx.reduce(SIMD3<Double>.zero) { $0 + SIMD3(Double(template[3 * $1]), Double(template[3 * $1 + 1]), Double(template[3 * $1 + 2])) }
            / Double(max(idx.count, 1))
    }
}

extension simd_float4x4 {
    func transformPointD(_ p: SIMD3<Double>) -> SIMD3<Double> {
        let q = self * SIMD4<Float>(Float(p.x), Float(p.y), Float(p.z), 1)
        return SIMD3(Double(q.x), Double(q.y), Double(q.z))
    }
}
