import SwiftUI

@main
struct FaceFrameApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

struct RootView: View {
    @StateObject private var scanner = ScanController()
    @State private var screen = Screen.home
    @State private var demoSession: FitSession?
    @State private var buildingDemo = false
    @State private var replayStatus: String?

    enum Screen { case home, scanning }

    var body: some View {
        Group {
            switch screen {
            case .home:
                HomeView(scanSupported: ScanController.isSupported, buildingDemo: buildingDemo,
                         onScan: {
                             screen = .scanning
                             scanner.start()
                         },
                         onDemo: loadDemo,
                         onReplay: replay,
                         onOpen: open)
            case .scanning:
                scanningContent
            }
        }
        .fullScreenCover(item: $demoSession) { session in
            ReviewView(session: session) { demoSession = nil }
        }
        .overlay {
            if let replayStatus {
                VStack(spacing: 16) {
                    ProgressView()
                    Text(replayStatus).font(.headline).multilineTextAlignment(.center)
                }
                .padding()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.systemBackground))
            }
        }
    }

    @ViewBuilder private var scanningContent: some View {
        switch scanner.phase {
        case .idle, .scanning:
            ScanView(controller: scanner) { screen = .home }
        case .reconstructing(let message):
            VStack(spacing: 16) {
                ProgressView()
                Text(message).font(.headline)
                Text("Fusing depth frames into a 0.5 mm model").font(.footnote).foregroundStyle(.secondary)
            }
        case .finished:
            if let result = scanner.result {
                ReviewView(session: result) { screen = .home }
            }
        case .failed(let message):
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                Text(message).multilineTextAlignment(.center)
                Button("Back") { screen = .home }.buttonStyle(.borderedProminent)
            }
            .padding()
        }
    }

    /// Reopens a saved scan in the review screen.
    private func open(_ entry: SavedScans.Entry) {
        replayStatus = "Opening scan…"
        DispatchQueue.global(qos: .userInitiated).async {
            let session = try? SavedScans.load(entry)
            DispatchQueue.main.async {
                replayStatus = nil
                demoSession = session
            }
        }
    }

    /// Runs a recorded capture through the same pipeline as a live scan, then opens its review.
    private func replay(_ url: URL) {
        replayStatus = "Loading capture…"
        DispatchQueue.global(qos: .userInitiated).async {
            let session = try? CaptureReplay.run(url) { message in DispatchQueue.main.async { replayStatus = message } }
            DispatchQueue.main.async {
                replayStatus = nil
                demoSession = session
            }
        }
    }

    private func loadDemo() {
        buildingDemo = true
        DispatchQueue.global(qos: .userInitiated).async {
            let session = HeadReconstruction.demo()
            DispatchQueue.main.async {
                buildingDemo = false
                demoSession = session
            }
        }
    }
}
