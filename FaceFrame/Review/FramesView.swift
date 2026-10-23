import SwiftUI
import UniformTypeIdentifiers
import WebKit

/// Designs printable frames for the scanned head with GNMFrames (bundled in Resources/GNMFrames by
/// Tools/sync_gnmframes.sh), shown in a web view and started on the GNM head fitted to the scan.
/// Exported files are saved next to the scan's other exports, in a `frames` folder.
struct FramesView: View {
    let session: FitSession
    var onDone: () -> Void

    @State private var files: [URL] = []
    @State private var showShare = false
    @State private var error: String?
    @StateObject private var bridge = FramesBridge()

    var body: some View {
        NavigationStack {
            Group {
                if let head {
                    FramesWebView(head: head, bridge: bridge) { exported in save(exported) }
                        .ignoresSafeArea(edges: .bottom)
                } else {
                    ContentUnavailableView("No head model", systemImage: "eyeglasses",
                                           description: Text("The head model could not be fitted to this scan, so frames cannot be designed for it."))
                }
            }
            .navigationTitle("Frames")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done", action: onDone) }
                ToolbarItem(placement: .primaryAction) {
                    Button { bridge.pickGlasses() } label: { Label("Try on glasses", systemImage: "eyeglasses") }
                        .disabled(head == nil)
                }
            }
            .sheet(isPresented: $showShare) { ShareSheet(items: files) }
            .onAppear { bridge.onError = { error = $0 } }
            .alert("Saving failed", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(error ?? "")
            }
        }
    }

    /// What the page receives as `window.faceframeHead`.
    private var head: [String: Any]? {
        guard let m = session.modelFit else { return nil }
        var source: [String: Any] = ["app": "FaceFrame", "scan": ScanExporter.name(session),
                                     "fit": m.source == "photos" ? "GNM photo fit on the colour frames, sized on the TrueDepth scan"
                                         : "GNM identity fitted to the TrueDepth scan (metric)",
                                     "headScale": Double((m.headScale * 10000).rounded()) / 10000,
                                     "fitRMSmm": Double((m.rmsMM * 100).rounded()) / 100]
        if let p = m.prior { source["prior"] = p }
        return ["identity": m.identity, "headScale": m.headScale,
                "title": (m.source == "photos" ? String(format: "Fitted to your photos, sized on your scan (×%.2f)", m.headScale) : "Fitted to your scan") + (m.prior.map { " · prior \($0.replacingOccurrences(of: "/", with: ", ").replacingOccurrences(of: "_", with: " "))" } ?? ""),
                "source": source]
    }

    private func save(_ exported: [(name: String, data: Data)]) {
        do {
            let dir = try ScanExporter.directory(session).appendingPathComponent("frames", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            files = try exported.map { f in
                // Names come from the page: keep only the last path component.
                let url = dir.appendingPathComponent(URL(fileURLWithPath: f.name).lastPathComponent)
                try f.data.write(to: url)
                return url
            }
            showShare = !files.isEmpty
        } catch {
            self.error = error.localizedDescription
        }
    }
}

/// Native side of the frame designer page: its web view, and the Files picker for glasses to try on
/// (web views cannot open file inputs from script). Presented with UIKit: SwiftUI's file importer did not
/// appear from this full-screen cover.
final class FramesBridge: NSObject, ObservableObject, UIDocumentPickerDelegate {
    weak var webView: WKWebView?
    var onError: (String) -> Void = { _ in }

    func pickGlasses() {
        guard var top = webView?.window?.rootViewController else { return }
        while let next = top.presentedViewController { top = next }
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType(filenameExtension: "obj") ?? .data, .data], asCopy: true)
        picker.delegate = self
        top.present(picker, animated: true)
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let url = urls.first else { return }
        do {
            tryOn(name: url.lastPathComponent, text: try String(contentsOf: url, encoding: .utf8))
        } catch {
            onError("Could not read \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// Hands an OBJ to the page's try-on.
    func tryOn(name: String, text: String) {
        guard let data = try? JSONSerialization.data(withJSONObject: [name, text]), let args = String(data: data, encoding: .utf8) else { return }
        webView?.evaluateJavaScript("window.faceframeTryOn(...\(args))")
    }
}

private struct FramesWebView: UIViewRepresentable {
    let head: [String: Any]
    let bridge: FramesBridge
    var onExport: ([(name: String, data: Data)]) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(bridge: bridge, onExport: onExport) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(BundleFiles(), forURLScheme: BundleFiles.scheme)
        let controller = config.userContentController
        if let json = try? JSONSerialization.data(withJSONObject: head), let text = String(data: json, encoding: .utf8) {
            controller.addUserScript(WKUserScript(source: "window.faceframeHead = \(text);", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        controller.add(context.coordinator, name: "faceframe")
        let view = WKWebView(frame: .zero, configuration: config)
        view.navigationDelegate = context.coordinator
        #if DEBUG
        view.isInspectable = true
        #endif
        view.load(URLRequest(url: URL(string: "\(BundleFiles.scheme)://app/gnmframes.html")!))
        bridge.webView = view
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {
        context.coordinator.onExport = onExport
    }

    static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) {
        view.configuration.userContentController.removeScriptMessageHandler(forName: "faceframe")
    }

    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
        /// Testing: FACEFRAME_TRYON_OBJ=<path> in the launch environment loads that OBJ, as if picked in Files.
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            guard let path = ProcessInfo.processInfo.environment["FACEFRAME_TRYON_OBJ"],
                  let text = try? String(contentsOfFile: path, encoding: .utf8) else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                self.bridge.tryOn(name: URL(fileURLWithPath: path).lastPathComponent, text: text)
            }
        }

        var onExport: ([(name: String, data: Data)]) -> Void
        let bridge: FramesBridge
        init(bridge: FramesBridge, onExport: @escaping ([(name: String, data: Data)]) -> Void) {
            self.bridge = bridge
            self.onExport = onExport
        }

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            if (message.body as? [String: Any])?["type"] as? String == "pickGlasses" {
                DispatchQueue.main.async { self.bridge.pickGlasses() }
                return
            }
            guard let body = message.body as? [String: Any], body["type"] as? String == "export",
                  let list = body["files"] as? [[String: Any]] else { return }
            let files: [(name: String, data: Data)] = list.compactMap { f in
                guard let name = f["name"] as? String else { return nil }
                if let text = f["text"] as? String { return (name, Data(text.utf8)) }
                if let b64 = f["base64"] as? String, let data = Data(base64Encoded: b64) { return (name, data) }
                return nil
            }
            onExport(files)
        }
    }
}

/// Serves the frame designer's page, script and GNM model files from the app bundle, so the page
/// can `fetch` them (WKWebView does not allow fetch on file URLs). Paths resolve by file name.
private final class BundleFiles: NSObject, WKURLSchemeHandler {
    static let scheme = "faceframe-frames"
    private static let types = ["html": "text/html", "js": "text/javascript", "json": "application/json", "bin": "application/octet-stream"]
    private static let files: Set<String> = ["gnmframes.html", "gnmframes.js", "meta.json", "template.bin", "identity_basis.bin",
                                             "triangles.bin", "mirror.bin", "groups.bin", "joint_template.bin", "joint_basis.bin"]
    private var stopped = Set<ObjectIdentifier>()

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url, Self.files.contains(url.lastPathComponent),
              let file = Bundle.main.url(forResource: url.deletingPathExtension().lastPathComponent, withExtension: url.pathExtension) else {
            task.didFailWithError(URLError(.fileDoesNotExist))
            return
        }
        let id = ObjectIdentifier(task)
        DispatchQueue.global(qos: .userInitiated).async {
            let data = try? Data(contentsOf: file, options: .mappedIfSafe)
            DispatchQueue.main.async {
                guard !self.stopped.contains(id) else { self.stopped.remove(id); return }
                guard let data else { task.didFailWithError(URLError(.cannotOpenFile)); return }
                let headers = ["Content-Type": Self.types[url.pathExtension] ?? "application/octet-stream", "Content-Length": "\(data.count)"]
                task.didReceive(HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!)
                task.didReceive(data)
                task.didFinish()
            }
        }
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {
        stopped.insert(ObjectIdentifier(task))
    }
}
