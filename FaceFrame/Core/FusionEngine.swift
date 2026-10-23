import Foundation
import simd

/// Recent ARKit head poses, used to look up the pose at the depth capture time. Depth arrives at
/// ~15 Hz and can be up to ~66 ms older than the 60 Hz frame that delivers it; using the frame's
/// own pose misplaces the depth by several millimetres whenever the head is turning.
struct PoseHistory {
    private var samples: [(time: Double, pose: simd_float4x4)] = []

    mutating func append(time: Double, pose: simd_float4x4) {
        samples.append((time, pose))
        if let last = samples.last?.time { samples.removeAll { last - $0.time > 1.0 } }
    }

    mutating func reset() { samples.removeAll() }

    /// Interpolated pose at `t`, nil when there is no tracked pose close enough on both sides.
    func pose(at t: Double) -> simd_float4x4? {
        guard let first = samples.first, let last = samples.last else { return nil }
        if t >= last.time { return t - last.time < 0.02 ? last.pose : nil }
        if t < first.time { return nil }
        for i in 1..<samples.count where samples[i].time >= t {
            let a = samples[i - 1], b = samples[i]
            if b.time - a.time > 0.1 { return nil } // tracking gap
            let u = Float((t - a.time) / max(b.time - a.time, 1e-6))
            return RigidMath.interpolate(a.pose, b.pose, u)
        }
        return nil
    }

    /// Head speed relative to the camera around `t`: degrees per second, with head translation (in
    /// camera coordinates) folded in as 1° per 3 mm.
    func motion(at t: Double) -> Float {
        guard let first = samples.first?.time, let last = samples.last?.time else { return 0 }
        let t0 = max(first, t - 0.04), t1 = min(last, t + 0.04)
        guard t1 - t0 > 0.01, let a = pose(at: t0), let b = pose(at: t1) else { return 0 }
        let rel = b * a.inverse
        let deg = rel.rotationAngle * 180 / .pi
        // Translation of the head (face origin) as seen by the camera; orbiting due to rotation cancels.
        let mm = simd_distance(a.inverse.translation, b.inverse.translation) * 1000
        return max(deg, mm / 3) / Float(t1 - t0)
    }
}

/// Coverage of the head, as fractions of the frames each view needs.
struct ScanCoverage: Equatable {
    var front: Float = 0
    var leftEar: Float = 0
    var rightEar: Float = 0
    var chinUp: Float = 0
    var chinDown: Float = 0
}

/// Guided sequence for a self-scan. Instructions are spoken because the user cannot read the
/// screen with the head turned 60°+ (which is what it takes for TrueDepth to see an ear).
enum ScanStep: Int, CaseIterable {
    case lookStraight, turnLeft, centerAfterLeft, turnRight, centerAfterRight, chinUp, chinDown, done

    var instruction: String {
        switch self {
        case .lookStraight: return "Look straight at the phone"
        case .turnLeft: return "Slowly turn your head to the LEFT until your right ear faces the phone"
        case .centerAfterLeft, .centerAfterRight: return "Slowly come back to the center"
        case .turnRight: return "Slowly turn your head to the RIGHT until your left ear faces the phone"
        case .chinUp: return "Tilt your chin up"
        case .chinDown: return "Tilt your chin down"
        case .done: return "Scan complete"
        }
    }

    var spoken: String {
        switch self {
        case .lookStraight: return "Look straight at the phone and hold still."
        case .turnLeft: return "Now slowly turn your head to the left, as far as you comfortably can."
        case .centerAfterLeft, .centerAfterRight: return "Good. Come back to the center."
        case .turnRight: return "Now slowly turn your head to the right, as far as you comfortably can."
        case .chinUp: return "Tilt your chin up."
        case .chinDown: return "And tilt your chin down."
        case .done: return "Scan complete."
        }
    }
}

struct ScanDiagnostics: Equatable, Codable {
    var mapping: String?
    var mappingAgreementMM: Float?
    var fusedFrames = 0
    var rejectedFast = 0
    var rejectedAlignment = 0
    var rejectedOther = 0
    var meanAlignmentRMSmm: Float?
    /// Times the depth mirroring was found wrong mid-scan (fusion restarted each time).
    var mappingSwitches = 0
    /// ARKit-to-TrueDepth correction found at calibration (see PoseCorrection).
    var arkitScale: Float?
    var arkitOffsetMM: Vec3?
    var faceFitBeforeMM: Float?
    var faceFitAfterMM: Float?
}

struct ScanStatus: Equatable {
    enum Tracking: Equatable { case searching, arkit, geometry, lost, tooFast }
    var tracking: Tracking = .searching
    var coverage = ScanCoverage()
    var step: ScanStep = .lookStraight
    var yawDegrees: Float = 0
    var pitchDegrees: Float = 0
    var distance: Float = 0
    var alignmentRMSmm: Float?
    var calibrating = true
    var diagnostics = ScanDiagnostics()
}

/// Fuses `FusionFrame`s into a head-fixed TSDF volume: depth mapping calibration, pose prediction,
/// alignment to the model, frame rejection, integration and coverage guidance. No ARKit
/// dependency, so recorded captures replay identically on a Mac. Not thread safe; use one queue.
final class FusionEngine {
    static let voxelSize: Float = 0.0005
    /// Frames faster than this are blurred and badly timed; they are not fused.
    static let maxMotion: Float = 60
    static let bootstrapFrames = 12

    private(set) var volume: TSDFVolume?
    private(set) var mapping: DepthMapping?
    private(set) var eyeMid = Vec3.zero
    private var mappingErrors: [DepthMapping: [Float]] = [:]
    private(set) var correction = PoseCorrection.identity
    /// Recent tracked frames with face vertices (calibration and re-estimation after a mirror switch).
    private var faceFrames: [FusionFrame] = []
    private var calibrationEyes: [(Vec3, Vec3)] = []
    private var eyeSum = (Vec3.zero, Vec3.zero)
    private var eyeCount = 0
    private var lastPose: simd_float4x4?
    private var lastPoseTime: Double = 0
    private var previousPose: (pose: simd_float4x4, time: Double)?
    private var counts = (front: 0, left: 0, right: 0, up: 0, down: 0)
    private let targets = (front: 15, left: 22, right: 22, up: 10, down: 8)
    private(set) var step: ScanStep = .lookStraight
    private var aligner = SDFAligner()
    private var diag = ScanDiagnostics()
    private var rmsSum: Float = 0

    var eyes: (left: Vec3, right: Vec3)? {
        eyeCount > 0 ? (correction.headPoint(eyeSum.0 / Float(eyeCount)), correction.headPoint(eyeSum.1 / Float(eyeCount))) : nil
    }
    var diagnostics: ScanDiagnostics { diag }
    /// ARKit eye centres in ARKit's own (uncorrected) face frame.
    var rawEyes: (left: Vec3, right: Vec3)? {
        eyeCount > 0 ? (eyeSum.0 / Float(eyeCount), eyeSum.1 / Float(eyeCount)) : nil
    }

    func process(_ input: FusionFrame) -> ScanStatus {
        if input.arPose != nil, input.faceVertices != nil {
            faceFrames.append(input)
            if faceFrames.count > 8 { faceFrames.removeFirst() }
        }
        guard let mapping else { return calibrate(input) }
        var f = input
        f.arPose = input.arPose.map(correction.apply)
        guard let volume else { return status(.searching) }

        // 1. Reject motion blur / mistimed frames outright.
        if f.motion > Self.maxMotion {
            diag.rejectedFast += 1
            return status(.tooFast, pose: f.arPose ?? lastPose)
        }

        // 2. Predict the head-from-camera pose: ARKit (corrected) and/or constant-velocity motion.
        let bootstrapping = volume.integratedFrames < Self.bootstrapFrames
        let motionPrediction = bootstrapping ? nil : predictFromMotion(at: f.time)
        let predicted: simd_float4x4
        var tracking: ScanStatus.Tracking
        if let arPose = f.arPose {
            predicted = arPose
            tracking = .arkit
        } else if let m = motionPrediction {
            predicted = m
            tracking = .geometry
        } else {
            diag.rejectedOther += 1
            return status(.lost)
        }
        if let arPose = f.arPose, let verts = f.faceVertices,
           checkMirroring(f, arPose: arPose, vertices: verts.map(correction.headPoint)) {
            return status(.searching)
        }
        // The first frames define the model: only slow, near-frontal, ARKit-tracked ones.
        if bootstrapping {
            let v = viewAngles(predicted)
            if f.motion > 20 || abs(v.yaw) > 25 || abs(v.pitch) > 20 {
                diag.rejectedOther += 1
                return status(tracking, pose: predicted)
            }
        }

        let img = mapping.image(f)
        let quality = img.qualityWeights()

        // 3. Collect head points.
        let box = (min: volume.origin, max: volume.origin + Vec3(volume.dims) * volume.voxelSize)
        let rigidMinY = eyeMid.y - 0.045
        var cameraPoints: [Vec3] = []
        var alignPoints: [Vec3] = []
        cameraPoints.reserveCapacity(img.width * img.height / 3)
        for y in 0..<img.height {
            for x in 0..<img.width {
                let d = img.values[y * img.width + x]
                if d <= 0 || d > 0.8 || quality[y * img.width + x] == 0 { continue }
                let pc = img.cameraPoint(x: x, y: y, depth: d)
                let ph = predicted.transformPoint(pc)
                if ph.x < box.min.x || ph.y < box.min.y || ph.z < box.min.z
                    || ph.x > box.max.x || ph.y > box.max.y || ph.z > box.max.z { continue }
                cameraPoints.append(pc)
                if ph.y > rigidMinY && (x ^ y) & 3 == 0 { alignPoints.append(pc) }
            }
        }
        guard cameraPoints.count > 2000 else {
            diag.rejectedOther += 1
            return status(.lost)
        }

        // 4. Align to the fused model. Once the model exists, a frame that does not align is dropped:
        //    fusing it would blur the surface.
        var pose = predicted
        var rms: Float? = nil
        if volume.integratedFrames >= 6 {
            if alignPoints.count > 4000 {
                alignPoints = stride(from: 0, to: alignPoints.count, by: alignPoints.count / 4000 + 1).map { alignPoints[$0] }
            }
            // Try ARKit's pose first, then the motion prediction. ARKit drifts at large turns
            // (before losing the face), so a geometric alignment that disagrees with it is accepted
            // when it fits the model clearly well.
            let largeTurn = abs(viewAngles(predicted).yaw) > 35
            var candidates: [(pose: simd_float4x4, fromARKit: Bool)] = []
            if let a = f.arPose { candidates.append((a, true)) }
            if let m = motionPrediction {
                let close = f.arPose.map { a in
                    simd_length((m * a.inverse).translation) < 0.002 && (m * a.inverse).rotationAngle < 0.02
                } ?? false
                if !close { candidates.append((m, false)) }
            }
            var accepted: (SDFAligner.Result, Bool)? = nil
            for c in candidates {
                aligner.translationPriorSigma = c.fromARKit ? (largeTurn ? 0.006 : 0.002) : nil
                aligner.rotationPriorSigma = c.fromARKit ? (largeTurn ? 0.05 : 0.0175) : nil
                guard let r = aligner.align(points: alignPoints, initial: c.pose, volume: volume), r.converged else { continue }
                let excellent = r.rmsMeters < 0.0013 && r.inlierRatio > 0.55
                if c.fromARKit {
                    let delta = r.pose * c.pose.inverse
                    let plausible = simd_length(delta.translation) < 0.012 && delta.rotationAngle < 0.12
                    if plausible || excellent { accepted = (r, true); break }
                } else if excellent {
                    accepted = (r, false)
                    break
                }
            }
            if let (r, fromARKit) = accepted {
                pose = r.pose
                rms = r.rmsMeters
                if !fromARKit || f.arPose == nil { tracking = .geometry }
            } else if !bootstrapping {
                diag.rejectedAlignment += 1
                return status(f.arPose != nil ? tracking : .lost, pose: predicted)
            }
        }

        // 5. Fuse.
        let headPoints = cameraPoints.map { pose.transformPoint($0) }
        volume.integrate(depth: img, headFromCamera: pose, surfacePoints: headPoints, pixelWeights: quality)
        previousPose = lastPose.map { ($0, lastPoseTime) }
        lastPose = pose
        lastPoseTime = f.time
        diag.fusedFrames = volume.integratedFrames
        if let rms {
            rmsSum += rms
            diag.meanAlignmentRMSmm = rmsSum / Float(diag.fusedFrames) * 1000
        }

        // 6. Eyes (only near-frontal, slow, ARKit-tracked frames).
        let view = viewAngles(pose)
        if let l = f.leftEye, let r = f.rightEye, f.arPose != nil, abs(view.yaw) < 15, abs(view.pitch) < 15, f.motion < 20 {
            eyeSum.0 += l
            eyeSum.1 += r
            eyeCount += 1
        }
        updateCoverage(view)
        return status(tracking, pose: pose, rms: rms)
    }

    private var mirrorEvidence: [Float] = [] // error(current) / error(mirrored) on turned frames

    /// On frames where the face is turned (asymmetric in the image) compare the current mapping with
    /// its mirror. If the mirror is consistently better, switch and restart fusion. Returns true when
    /// fusion was restarted.
    private func checkMirroring(_ f: FusionFrame, arPose: simd_float4x4, vertices: [Vec3]) -> Bool {
        guard let mapping, mappingVerified == false else { return false }
        let v = viewAngles(arPose)
        guard abs(v.yaw) > 12, abs(v.yaw) < 50, f.motion < Self.maxMotion else { return false }
        var mirror = mapping
        mirror.mirrored.toggle()
        let cameraFromFace = arPose.inverse
        guard let e0 = DepthMapping.faceAgreement(mapping.image(f), cameraFromFace: cameraFromFace, faceVertices: vertices),
              let e1 = DepthMapping.faceAgreement(mirror.image(f), cameraFromFace: cameraFromFace, faceVertices: vertices)
        else { return false }
        mirrorEvidence.append(e0 / max(e1, 1e-5))
        guard mirrorEvidence.count >= 6, let ratio = mirrorEvidence.median() else { return false }
        if ratio > 1.4 {
            // Mirror fits clearly better: restart with it.
            self.mapping = mirror
            diag.mapping = mirror.description
            diag.mappingAgreementMM = e1 * 1000
            diag.mappingSwitches += 1
            mappingVerified = true
            estimateCorrection(mapping: mirror)
            restartFusion()
            return true
        }
        if ratio < 0.8 || mirrorEvidence.count >= 15 { mappingVerified = true }
        return false
    }

    private var mappingVerified = false
    /// True while the engine still wants ARKit face vertices (calibration and mirroring check).
    var needsFaceVertices: Bool { mapping == nil || !mappingVerified }

    private func restartFusion() {
        guard let old = volume else { return }
        volume = TSDFVolume(origin: old.origin, extent: Vec3(old.dims) * old.voxelSize, voxelSize: old.voxelSize,
                            truncFront: old.truncFront, truncBack: old.truncBack)
        counts = (0, 0, 0, 0, 0)
        step = .lookStraight
        eyeSum = (.zero, .zero)
        eyeCount = 0
        lastPose = nil
        previousPose = nil
        rmsSum = 0
        diag.fusedFrames = 0
    }

    /// Fits ARKit's face mesh to the TrueDepth points of the recent tracked frames.
    private func fitCorrection(mapping: DepthMapping) -> PoseCorrection.Fit? {
        let samples = faceFrames.compactMap { fr -> PoseCorrection.Sample? in
            guard let pose = fr.arPose, let v = fr.faceVertices else { return nil }
            return PoseCorrection.Sample(image: mapping.image(fr), arPose: pose, vertices: v)
        }
        guard let fit = PoseCorrection.estimate(samples),
              fit.correction.scale > 0.7, fit.correction.scale < 1.4,
              simd_length(fit.correction.translation) < 0.15,
              fit.rmsAfter <= fit.rmsBefore else { return nil }
        return fit
    }

    private func estimateCorrection(mapping: DepthMapping) {
        use(fitCorrection(mapping: mapping))
    }

    private func use(_ fit: PoseCorrection.Fit?) {
        guard let fit else {
            correction = .identity
            return
        }
        correction = fit.correction
        diag.arkitScale = fit.correction.scale
        diag.arkitOffsetMM = fit.correction.translation * 1000
        diag.faceFitBeforeMM = fit.rmsBefore * 1000
        diag.faceFitAfterMM = fit.rmsAfter * 1000
    }

    private func calibrate(_ f: FusionFrame) -> ScanStatus {
        guard let arPose = f.arPose, let verts = f.faceVertices, let l = f.leftEye, let r = f.rightEye, f.motion < 20 else {
            return status(.searching, pose: f.arPose)
        }
        let cameraFromFace = arPose.inverse
        for m in DepthMapping.all {
            if m.useCalibrationData && f.calibrationK == nil { continue }
            if let e = DepthMapping.faceAgreement(m.image(f), cameraFromFace: cameraFromFace, faceVertices: verts) {
                mappingErrors[m, default: []].append(e)
            }
        }
        calibrationEyes.append((l, r))
        guard calibrationEyes.count >= 8 else { return status(.arkit, pose: arPose) }
        // Score each mapping by how well ARKit's face mesh fits the depth after the ARKit correction;
        // fall back to the raw agreement when the fit fails.
        var fits: [DepthMapping: PoseCorrection.Fit] = [:]
        var scored: [(DepthMapping, Float)] = []
        for m in DepthMapping.all where !(m.useCalibrationData && f.calibrationK == nil) {
            if let fit = fitCorrection(mapping: m) {
                fits[m] = fit
                scored.append((m, fit.rmsAfter))
            } else if let raw = mappingErrors[m]?.median() {
                scored.append((m, raw + 0.05))
            }
        }
        guard var best = scored.min(by: { $0.1 < $1.1 }) else { return status(.arkit, pose: arPose) }
        // A centred, near-symmetric face cannot tell mirrored from direct, and depth calibration from
        // colour intrinsics barely differ: prefer direct + calibration data unless clearly worse.
        // Mirroring is re-checked on turned frames during the scan.
        func score(_ m: DepthMapping) -> Float? { scored.first { $0.0 == m }?.1 }
        if best.0.mirrored, let direct = score(DepthMapping(mirrored: false, useCalibrationData: best.0.useCalibrationData)),
           direct < best.1 * 1.25 + 0.0003 {
            best = (DepthMapping(mirrored: false, useCalibrationData: best.0.useCalibrationData), direct)
        }
        if !best.0.useCalibrationData, let cal = score(DepthMapping(mirrored: best.0.mirrored, useCalibrationData: true)),
           cal < best.1 * 1.25 + 0.0003 {
            best = (DepthMapping(mirrored: best.0.mirrored, useCalibrationData: true), cal)
        }
        mapping = best.0
        diag.mapping = best.0.description
        diag.mappingAgreementMM = (mappingErrors[best.0]?.median()).map { $0 * 1000 }
        use(fits[best.0])
        let mid = calibrationEyes.reduce(Vec3.zero) { $0 + ($1.0 + $1.1) / 2 } / Float(calibrationEyes.count)
        eyeMid = correction.headPoint(mid)
        // Head box relative to the eyes: ears are ~8–10 cm behind the eyes, the crown is cut off.
        volume = TSDFVolume(origin: eyeMid + Vec3(-0.125, -0.16, -0.21), extent: Vec3(0.25, 0.29, 0.30),
                            voxelSize: Self.voxelSize, truncFront: 0.004, truncBack: 0.003)
        return status(.arkit, pose: arPose)
    }

    /// Constant-velocity extrapolation of the last fused poses (nil when tracking is stale).
    private func predictFromMotion(at t: Double) -> simd_float4x4? {
        guard let last = lastPose, t - lastPoseTime < 0.8 else { return nil }
        guard let prev = previousPose, lastPoseTime - prev.time > 0.01, lastPoseTime - prev.time < 0.4 else { return last }
        let factor = Float(min(3, (t - lastPoseTime) / (lastPoseTime - prev.time)))
        return RigidMath.extrapolate(prev.pose, last, factor)
    }

    /// Direction from the head to the camera, in head coordinates.
    private func viewAngles(_ pose: simd_float4x4) -> (yaw: Float, pitch: Float, distance: Float) {
        let headCenter = eyeMid + Vec3(0, 0, -0.06)
        let v = pose.translation - headCenter
        let d = simd_normalize(v)
        return (atan2(d.x, d.z) * 180 / .pi, asin(max(-1, min(1, d.y))) * 180 / .pi, simd_length(v))
    }

    private func updateCoverage(_ view: (yaw: Float, pitch: Float, distance: Float)) {
        if abs(view.yaw) < 15 && abs(view.pitch) < 12 { counts.front += 1 }
        if view.yaw > 55 { counts.left += 1 }   // camera on the subject's left side
        if view.yaw < -55 { counts.right += 1 }
        if view.pitch < -15 && abs(view.yaw) < 35 { counts.up += 1 }
        if view.pitch > 15 && abs(view.yaw) < 35 { counts.down += 1 }

        switch step {
        case .lookStraight: if counts.front >= targets.front { step = .turnLeft }
        case .turnLeft: if counts.right >= targets.right { step = .centerAfterLeft }
        case .centerAfterLeft: if abs(view.yaw) < 15 { step = counts.left >= targets.left ? .chinUp : .turnRight }
        case .turnRight: if counts.left >= targets.left { step = .centerAfterRight }
        case .centerAfterRight: if abs(view.yaw) < 15 { step = counts.right >= targets.right ? .chinUp : .turnLeft }
        case .chinUp: if counts.up >= targets.up { step = .chinDown }
        case .chinDown: if counts.down >= targets.down { step = .done }
        case .done: break
        }
    }

    private func status(_ tracking: ScanStatus.Tracking, pose: simd_float4x4? = nil, rms: Float? = nil) -> ScanStatus {
        var s = ScanStatus()
        s.tracking = tracking
        s.calibrating = mapping == nil
        s.step = step
        s.alignmentRMSmm = rms.map { $0 * 1000 }
        s.diagnostics = diag
        func frac(_ a: Int, _ b: Int) -> Float { min(1, Float(a) / Float(b)) }
        s.coverage = ScanCoverage(front: frac(counts.front, targets.front), leftEar: frac(counts.left, targets.left),
                                  rightEar: frac(counts.right, targets.right), chinUp: frac(counts.up, targets.up),
                                  chinDown: frac(counts.down, targets.down))
        if let pose, mapping != nil {
            let v = viewAngles(pose)
            s.yawDegrees = v.yaw
            s.pitchDegrees = v.pitch
            s.distance = v.distance
        }
        return s
    }
}
