import ARKit
import AVFoundation
import Combine
import CoreImage
import UIKit

/// Runs the ARKit face-tracking session, feeds TrueDepth frames to the `FusionEngine` and speaks
/// the guidance. Optionally records every frame for offline replay (Tools/Replay).
final class ScanController: NSObject, ObservableObject, ARSessionDelegate {
    enum Phase: Equatable {
        case idle, scanning, reconstructing(String), finished, failed(String)
    }

    static let recordCaptureKey = "recordRawCapture"

    let session = ARSession()
    @Published private(set) var status = ScanStatus()
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var result: FitSession?
    @Published private(set) var captureDirectory: URL?

    private let arQueue = DispatchQueue(label: "faceframe.ar", qos: .userInteractive)
    private let fusionQueue = DispatchQueue(label: "faceframe.fusion", qos: .userInitiated)
    private let busyLock = NSLock()
    private var fusionBusy = false

    // arQueue state
    private var stopped = true
    private var lastDepthTimestamp: TimeInterval = -1
    private var poses = PoseHistory()
    private var needsFaceVertices = true
    private var colorRecorder: ColorRecorder?
    private var colorFrameCounter = 0
    /// Near-frontal colour frames kept in memory for measuring the pupils (always, not only when recording).
    private var pupilSamples: [PupilEstimator.Sample] = []
    /// The views GNM's photo fit uses (always, not only when recording): the most frontal frame and the
    /// most turned one to each side so far, by ARKit's head yaw.
    private var photoFrames: [PhotoSlot: (score: Float, frame: PhotoObserver.Frame)] = [:]
    private enum PhotoSlot { case front, left, right }
    private let imageContext = CIContext(options: [.cacheIntermediates: false])

    // fusionQueue state
    private var engine = FusionEngine()
    private var recorder: CaptureRecorder?
    private var lastStep: ScanStep = .lookStraight
    private var lastSlowDownPrompt: TimeInterval = 0

    private let speech = AVSpeechSynthesizer()
    private let haptics = UINotificationFeedbackGenerator()

    static var isSupported: Bool { ARFaceTrackingConfiguration.isSupported }

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = arQueue
    }

    // MARK: - Control

    func start() {
        guard Self.isSupported else {
            phase = .failed("This device has no TrueDepth camera. Face scanning needs an iPhone or iPad with Face ID.")
            return
        }
        let record = UserDefaults.standard.bool(forKey: Self.recordCaptureKey)
        let newRecorder = record ? try? CaptureRecorder(directory: CaptureReplay.newCapture()) : nil
        captureDirectory = newRecorder?.directory
        let newColor = newRecorder.flatMap { try? ColorRecorder(directory: $0.directory) }
        fusionQueue.async {
            self.engine = FusionEngine()
            self.recorder = newRecorder
            self.lastStep = .lookStraight
        }
        let config = ARFaceTrackingConfiguration()
        config.maximumNumberOfTrackedFaces = 1
        config.isLightEstimationEnabled = false
        if let best = ARFaceTrackingConfiguration.supportedVideoFormats.max(by: { $0.framesPerSecond < $1.framesPerSecond }) {
            config.videoFormat = best
        }
        status = ScanStatus()
        result = nil
        phase = .scanning
        arQueue.async {
            self.stopped = false
            self.lastDepthTimestamp = -1
            self.poses.reset()
            self.needsFaceVertices = true
            self.colorRecorder = newColor
            self.colorFrameCounter = 0
            self.pupilSamples = []
            self.photoFrames = [:]
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        UIApplication.shared.isIdleTimerDisabled = true
        say(ScanStep.lookStraight.spoken)
    }

    func cancel() {
        arQueue.async { self.stopped = true }
        session.pause()
        speech.stopSpeaking(at: .immediate)
        UIApplication.shared.isIdleTimerDisabled = false
        phase = .idle
    }

    /// Stops capturing and builds the mesh + fit analysis.
    func finish() {
        arQueue.async { self.stopped = true }
        session.pause()
        UIApplication.shared.isIdleTimerDisabled = false
        phase = .reconstructing("Building mesh…")
        fusionQueue.async {
            guard let volume = self.engine.volume, volume.integratedFrames > 10, let eyes = self.engine.eyes else {
                DispatchQueue.main.async {
                    let saved = self.captureDirectory.map { " The raw capture was saved to Files › FaceFrame › Captures › \($0.lastPathComponent)." } ?? ""
                    self.phase = .failed("Not enough data was captured. Keep your face in view, move slowly and try again." + saved)
                }
                return
            }
            let diagnostics = self.engine.diagnostics
            let (samples, photoFrames) = self.arQueue.sync { (self.pupilSamples, self.photoFrames.values.map(\.frame)) }
            let pupils = self.engine.rawEyes.map {
                HeadReconstruction.PupilInput(samples: samples, arkitEyes: $0, correction: self.engine.correction)
            }
            let photos = self.engine.rawEyes.map { HeadReconstruction.PhotoInput(frames: photoFrames, arkitEyes: $0) }
            let session = HeadReconstruction.build(volume: volume, eyeLeft: eyes.left, eyeRight: eyes.right,
                                                   diagnostics: diagnostics, pupils: pupils, photos: photos) { message in
                DispatchQueue.main.async { self.phase = .reconstructing(message) }
            }
            DispatchQueue.main.async {
                if let session {
                    session.captureDirectory = self.captureDirectory
                    self.result = session
                    self.phase = .finished
                } else {
                    self.phase = .failed("The scan did not produce a usable surface. Try again in even lighting.")
                }
            }
        }
    }

    // MARK: - ARSessionDelegate (arQueue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard !stopped else { return }
        let face = frame.anchors.compactMap { $0 as? ARFaceAnchor }.first
        if let face, face.isTracked {
            poses.append(time: frame.timestamp, pose: face.transform.inverse * frame.camera.transform)
        }

        guard let depthData = frame.capturedDepthData, frame.capturedDepthDataTimestamp != lastDepthTimestamp else { return }
        let depthTime = frame.capturedDepthDataTimestamp
        lastDepthTimestamp = depthTime

        // Near-frontal colour frames for the pupils: every 3rd depth frame, up to 30.
        if pupilSamples.count < 30, colorFrameCounter % 3 == 0, let face, face.isTracked,
           let pose = poses.pose(at: frame.timestamp) {
            let eyeMid = (face.leftEyeTransform.translation + face.rightEyeTransform.translation) / 2
            let toCam = simd_normalize(pose.translation - eyeMid)
            let yaw = atan2(toCam.x, toCam.z) * 180 / .pi, pitch = asin(toCam.y) * 180 / .pi
            if abs(yaw) < 12, abs(pitch) < 12,
               let cg = imageContext.createCGImage(CIImage(cvPixelBuffer: frame.capturedImage),
                                                   from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(frame.capturedImage),
                                                                height: CVPixelBufferGetHeight(frame.capturedImage))) {
                pupilSamples.append(PupilEstimator.Sample(image: cg, intrinsics: frame.camera.intrinsics, arPose: pose))
            }
        }

        // Views for the photo fit: keep a frame when it is more frontal, or more turned, than the kept one.
        if let face, face.isTracked, let pose = poses.pose(at: frame.timestamp) {
            let t = pose.columns.3
            let yaw = atan2(t.x, t.z) * 180 / .pi, pitch = atan2(t.y, hypot(t.x, t.z)) * 180 / .pi
            let candidates: [(PhotoSlot, Float)] = [(.front, -(abs(yaw) + abs(pitch))), (.left, yaw - abs(pitch)), (.right, -yaw - abs(pitch))]
            for (slot, score) in candidates where score > (photoFrames[slot]?.score ?? -.infinity) + 1 {
                if slot != .front && score < 30 { continue }
                if let cg = imageContext.createCGImage(CIImage(cvPixelBuffer: frame.capturedImage),
                                                       from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(frame.capturedImage),
                                                                    height: CVPixelBufferGetHeight(frame.capturedImage))) {
                    photoFrames[slot] = (score, PhotoObserver.Frame(image: cg, intrinsics: frame.camera.intrinsics, arPose: pose))
                }
            }
        }

        // Colour frames (when recording): every 2nd depth frame, ~7 per second.
        colorFrameCounter += 1
        if let colorRecorder {
            if colorFrameCounter % 2 == 0 {
                colorRecorder.append(frame.capturedImage, time: frame.timestamp, intrinsics: frame.camera.intrinsics,
                                     arPose: poses.pose(at: frame.timestamp))
            }
        }

        busyLock.lock()
        if fusionBusy { busyLock.unlock(); return }
        fusionBusy = true
        busyLock.unlock()

        let arPose = poses.pose(at: depthTime)
        var f = FusionFrame(time: depthTime, width: 0, height: 0, depth: [],
                            calibrationK: nil, calibrationReference: nil,
                            intrinsics: frame.camera.intrinsics,
                            imageResolution: SIMD2(Float(frame.camera.imageResolution.width), Float(frame.camera.imageResolution.height)),
                            arPose: arPose, motion: poses.motion(at: depthTime),
                            leftEye: face?.leftEyeTransform.translation, rightEye: face?.rightEyeTransform.translation,
                            faceVertices: needsFaceVertices && arPose != nil ? face?.geometry.vertices : nil)
        fusionQueue.async {
            DepthConversion.fill(&f, from: depthData)
            self.recorder?.append(f)
            let s = self.engine.process(f)
            let wantsVertices = self.engine.needsFaceVertices
            self.arQueue.async { self.needsFaceVertices = wantsVertices }
            self.react(to: s, time: f.time)
            DispatchQueue.main.async {
                if case .scanning = self.phase { self.status = s }
            }
            self.busyLock.lock()
            self.fusionBusy = false
            self.busyLock.unlock()
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.phase = .failed(error.localizedDescription) }
    }

    // MARK: - Guidance (fusionQueue)

    private func react(to s: ScanStatus, time: Double) {
        if s.step != lastStep {
            lastStep = s.step
            let step = s.step
            DispatchQueue.main.async {
                self.haptics.notificationOccurred(.success)
                self.say(step.spoken)
            }
        } else if s.tracking == .tooFast, time - lastSlowDownPrompt > 4 {
            lastSlowDownPrompt = time
            DispatchQueue.main.async { self.say("Slower, please.") }
        }
    }

    private func say(_ text: String) {
        speech.stopSpeaking(at: .word)
        let u = AVSpeechUtterance(string: text)
        u.rate = AVSpeechUtteranceDefaultSpeechRate
        speech.speak(u)
    }
}
