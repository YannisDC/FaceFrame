import ARKit
import SceneKit
import SwiftUI

struct ScanView: View {
    @ObservedObject var controller: ScanController
    var onCancel: () -> Void

    var body: some View {
        ZStack {
            ARFacePreview(session: controller.session)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                instructionCard
                Spacer()
                yawGauge
                coverageRow
                HStack(spacing: 12) {
                    Button(role: .cancel) {
                        controller.cancel()
                        onCancel()
                    } label: {
                        Text("Cancel").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)

                    Button {
                        controller.finish()
                    } label: {
                        Text(controller.status.step == .done ? "Build model" : "Finish now").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(controller.status.diagnostics.fusedFrames < 20)
                }
                .controlSize(.large)
            }
            .padding()
        }
        .onChange(of: controller.status.step) { _, step in
            if step == .done { controller.finish() }
        }
    }

    private var instructionCard: some View {
        VStack(spacing: 6) {
            Text(controller.status.calibrating ? "Hold still — calibrating depth camera" : controller.status.step.instruction)
                .font(.title3.weight(.semibold))
                .multilineTextAlignment(.center)
            HStack(spacing: 10) {
                Label(trackingText, systemImage: trackingIcon)
                if controller.status.distance > 0 {
                    Text(distanceText).foregroundStyle(distanceColor)
                }
                Text("\(controller.status.diagnostics.fusedFrames) frames")
                if let rms = controller.status.alignmentRMSmm {
                    Text(String(format: "fit %.2f mm", rms))
                }
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            diagnosticsLine
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var diagnosticsLine: some View {
        let d = controller.status.diagnostics
        return HStack(spacing: 8) {
            if controller.captureDirectory != nil {
                Label("REC", systemImage: "record.circle").foregroundStyle(.red)
            }
            if let after = d.faceFitAfterMM {
                Text(String(format: "face fit %.1f mm", after)).foregroundStyle(after > 4 ? .orange : .secondary)
            } else if let a = d.mappingAgreementMM {
                Text(String(format: "depth match %.1f mm", a)).foregroundStyle(a > 5 ? .orange : .secondary)
            }
            if d.rejectedFast + d.rejectedAlignment > 0 {
                Text("dropped: \(d.rejectedFast) fast, \(d.rejectedAlignment) misaligned")
            }
        }
        .font(.caption2.monospacedDigit())
        .foregroundStyle(.secondary)
    }

    private var trackingText: String {
        switch controller.status.tracking {
        case .searching: return "Finding face"
        case .arkit: return "Tracking"
        case .geometry: return "Tracking (geometry)"
        case .lost: return "Lost — turn back"
        case .tooFast: return "Too fast — slow down"
        }
    }

    private var trackingIcon: String {
        switch controller.status.tracking {
        case .searching: return "faceid"
        case .arkit, .geometry: return "checkmark.circle"
        case .lost, .tooFast: return "exclamationmark.triangle"
        }
    }

    private var distanceText: String {
        let cm = controller.status.distance * 100
        if cm < 24 { return String(format: "%.0f cm · move back", cm) }
        if cm > 45 { return String(format: "%.0f cm · move closer", cm) }
        return String(format: "%.0f cm", cm)
    }

    private var distanceColor: Color {
        let cm = controller.status.distance * 100
        return cm < 24 || cm > 45 ? .orange : .secondary
    }

    /// Head yaw relative to the camera; the ear zones are beyond ±55°.
    private var yawGauge: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let x = CGFloat((max(-90, min(90, controller.status.yawDegrees)) + 90) / 180) * w // mirrors the preview
            ZStack(alignment: .leading) {
                Capsule().fill(.black.opacity(0.35))
                Rectangle().fill(.green.opacity(0.35)).frame(width: w * 35 / 180)
                Rectangle().fill(.green.opacity(0.35)).frame(width: w * 35 / 180).offset(x: w * 145 / 180)
                Circle().fill(.white).frame(width: 16, height: 16).offset(x: x - 8)
            }
            .clipShape(Capsule())
        }
        .frame(height: 16)
        .accessibilityLabel("Head turn \(Int(controller.status.yawDegrees)) degrees")
    }

    private var coverageRow: some View {
        let c = controller.status.coverage
        return HStack(spacing: 8) {
            CoverageChip(title: "R ear", value: c.rightEar)
            CoverageChip(title: "Front", value: c.front)
            CoverageChip(title: "L ear", value: c.leftEar)
            CoverageChip(title: "Up", value: c.chinUp)
            CoverageChip(title: "Down", value: c.chinDown)
        }
    }
}

private struct CoverageChip: View {
    var title: String
    var value: Float

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                Circle().stroke(.white.opacity(0.25), lineWidth: 4)
                Circle().trim(from: 0, to: CGFloat(value))
                    .stroke(value >= 1 ? Color.green : Color.white, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                if value >= 1 { Image(systemName: "checkmark").font(.caption.bold()).foregroundStyle(.green) }
            }
            .frame(width: 34, height: 34)
            Text(title).font(.caption2)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Camera preview with ARKit's face mesh drawn as a wireframe.
struct ARFacePreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.delegate = context.coordinator
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        context.coordinator.device = view.device
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, ARSCNViewDelegate {
        var device: MTLDevice?

        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard anchor is ARFaceAnchor, let device, let geometry = ARSCNFaceGeometry(device: device) else { return nil }
            geometry.firstMaterial?.fillMode = .lines
            geometry.firstMaterial?.diffuse.contents = UIColor.white.withAlphaComponent(0.35)
            geometry.firstMaterial?.lightingModel = .constant
            return SCNNode(geometry: geometry)
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let face = anchor as? ARFaceAnchor, let geometry = node.geometry as? ARSCNFaceGeometry else { return }
            geometry.update(from: face.geometry)
        }
    }
}
