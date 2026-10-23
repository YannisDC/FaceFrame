import SwiftUI

struct ReviewView: View {
    @ObservedObject var session: FitSession
    var onDone: () -> Void

    @State private var tab = Tab.summary
    @State private var adjusting: AdjustableLandmark?
    @State private var exportedFiles: [URL] = []
    @State private var showShare = false
    @State private var exportError: String?
    @State private var exporting = false
    @State private var display = HeadSceneView.Display.scan
    @State private var designingFrames = false
    @State private var saving = false
    @State private var savedMessage: String?

    enum Tab: String, CaseIterable { case summary = "Summary", nose = "Nose", ears = "Ears" }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ZStack(alignment: .top) {
                    HeadSceneView(mesh: session.mesh, modelMesh: session.modelFit?.mesh, display: display,
                                  result: session.result, onTap: adjusting == nil ? nil : { p in
                        if let l = adjusting { session.overrides[l] = p }
                    })
                    VStack(spacing: 8) {
                        if session.modelFit != nil {
                            Picker("Show", selection: $display) {
                                ForEach(HeadSceneView.Display.allCases) { Text($0.rawValue).tag($0) }
                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 260)
                            .padding(.top, 8)
                        }
                        if let l = adjusting {
                            adjustBanner(l)
                        }
                    }
                }
                .frame(maxHeight: .infinity)

                Picker("Section", selection: $tab) {
                    ForEach(Tab.allCases, id: \.self) { Text($0.rawValue) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.vertical, 8)

                List {
                    if tab == .summary, !r.warnings.isEmpty { warnings }
                    switch tab {
                    case .summary: summary
                    case .nose: nose
                    case .ears: ears
                    }
                }
                .listStyle(.insetGrouped)
                .contentMargins(.top, 0, for: .scrollContent)
                .frame(height: 290)
            }
            .background(Color(.systemGroupedBackground))
            .safeAreaInset(edge: .bottom) {
                if session.modelFit != nil {
                    Button {
                        designingFrames = true
                    } label: {
                        Label("Design frames", systemImage: "eyeglasses")
                            .font(.headline)
                            .frame(maxWidth: .infinity, minHeight: 34)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(.bar)
                }
            }
            .navigationTitle(session.isDemo ? "Demo head" : "Head scan")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done", action: onDone)
                }
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        save()
                    } label: {
                        if saving { ProgressView() } else { Label("Save", systemImage: "tray.and.arrow.down") }
                    }
                    .disabled(saving)
                    Menu {
                        ForEach(AdjustableLandmark.allCases) { l in
                            Button(l.title) { adjusting = l }
                        }
                        if let dir = session.captureDirectory {
                            Divider()
                            Button("Share raw capture (zip)") { shareCapture(dir) }
                        }
                    } label: {
                        Label("Adjust", systemImage: "hand.point.up.left")
                    }
                    Button {
                        export()
                    } label: {
                        if exporting { ProgressView() } else { Label("Export", systemImage: "square.and.arrow.up") }
                    }
                    .disabled(exporting)
                }
            }
            .sheet(isPresented: $showShare) {
                ShareSheet(items: exportedFiles)
            }
            .fullScreenCover(isPresented: $designingFrames) {
                FramesView(session: session) { designingFrames = false }
            }
            .alert("Scan saved", isPresented: Binding(get: { savedMessage != nil }, set: { if !$0 { savedMessage = nil } })) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(savedMessage ?? "")
            }
            .alert("Export failed", isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(exportError ?? "")
            }
        }
    }

    private func adjustBanner(_ l: AdjustableLandmark) -> some View {
        VStack(spacing: 8) {
            Text("Tap the model to place the \(l.title.lowercased())").font(.subheadline.weight(.semibold))
            HStack {
                Button("Reset to automatic") { session.overrides[l] = nil }
                    .buttonStyle(.bordered)
                Button("Done") { adjusting = nil }
                    .buttonStyle(.borderedProminent)
            }
            .controlSize(.small)
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(8)
    }

    private var r: FitResult { session.result }

    /// Warnings fold into one row: open it to read them.
    private var warnings: some View {
        Section {
            DisclosureGroup {
                ForEach(r.warnings, id: \.self) { w in
                    Text(w).font(.footnote)
                }
            } label: {
                Label(r.warnings.count == 1 ? "1 thing to check" : "\(r.warnings.count) things to check",
                      systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            }
        }
    }

    private var summary: some View {
        Group {
            Section("Eyes") {
                row("Pupillary distance", mm(r.pd), detail: r.pupilsFromImages ? "From the photos" : "From ARKit's face model")
                row("Mono PD (L / R)", "\(mmShort(r.monoPDLeft)) / \(mmShort(r.monoPDRight)) mm")
                row("Vertex distance", mm(r.vertexDistance))
            }
            Section("Bridge") {
                row("Width at 5 / 10 / 15 mm", "\(mmShort(r.dbr(at: 0.005))) / \(mmShort(r.dbr(at: 0.010))) / \(mmShort(r.dbr(at: 0.015))) mm",
                    detail: "Below the crest, in the frame's back plane")
                row("Pad splay", deg(r.padSplay))
            }
            Section("Temples") {
                row("Length to bend (L / R)", "\(mmShort(r.lengthToBend(.left))) / \(mmShort(r.lengthToBend(.right))) mm")
                row("Head width at the ears", mm(r.headWidthAtEars))
            }
            if let d = session.diagnostics {
                Section {
                    DisclosureGroup("Scan quality") {
                        row("Depth match to face mesh", d.mappingAgreementMM.map { String(format: "%.1f mm", $0) } ?? "—")
                        row("ARKit correction", d.arkitScale.map { sc in
                            String(format: "×%.3f · %.1f → %.1f mm", sc, d.faceFitBeforeMM ?? 0, d.faceFitAfterMM ?? 0)
                        } ?? "none")
                        row("Frames fused", "\(d.fusedFrames)")
                        row("Dropped (fast / misaligned / other)", "\(d.rejectedFast) / \(d.rejectedAlignment) / \(d.rejectedOther)")
                        row("Mean alignment residual", d.meanAlignmentRMSmm.map { String(format: "%.2f mm", $0) } ?? "—")
                        if let m = session.modelFit {
                            row("Head model", m.source == "photos" ? String(format: "Photos, ×%.3f", m.headScale) : "Fitted to scan")
                            row("Head model fit", String(format: "%.2f mm", m.rmsMM))
                            row("Head model prior", m.prior?.replacingOccurrences(of: "/", with: ", ").replacingOccurrences(of: "_", with: " ").capitalized ?? "GNM default")
                            row("Ear coverage (L / R)", String(format: "%.0f%% / %.0f%%", (m.earCoverage[.left] ?? 0) * 100, (m.earCoverage[.right] ?? 0) * 100))
                        }
                    }
                }
            }
        }
    }

    private var nose: some View {
        Group {
            Section {
                NoseProfileView(result: r)
                    .padding(.vertical, 4)
            }
            Section(r.crestIsManual ? "Crest (placed by hand)" : "Crest") {
                row("Below the sellion", mm(r.sellion.y - r.crest.y))
                row("Pupils above the crest", mm(r.pupilHeightAboveCrest))
                row("Sellion above the pupils", mm(r.sellionAbovePupils))
                row("Frontal angle", deg(r.frontalAngle))
            }
            Section("Width by level") {
                ForEach(r.slices.filter { Int(($0.levelBelowCrest * 1000).rounded()) % 2 == 0 && $0.levelBelowCrest <= 0.02 }, id: \.levelBelowCrest) { s in
                    row("\(Int((s.levelBelowCrest * 1000).rounded())) mm below crest",
                        "\(mm(s.backPlaneWidth)) · \(deg(avg(s.splayLeft, s.splayRight)))")
                }
            }
        }
    }

    private var ears: some View {
        Group {
            ForEach([HeadSide.left, .right], id: \.self) { side in
                Section(side == .left ? "Left ear" : "Right ear") {
                    if let e = r.ear(side) {
                        row("Length to bend", mm(r.lengthToBend(side)), detail: "Seat from the \(seatMethod(e.method))")
                        row("Below the pupils", mm(r.templeDrop(side)))
                        row("Temple declination", deg(r.templeDeclination(side)))
                        row("Out from the midline", mm(abs(e.seat.x - r.midlineX)))
                    } else {
                        Text("Not found. Use Adjust to place it.").font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }
            if let l = r.earLeft, let rr = r.earRight {
                Section("Left minus right") {
                    row("Height", mm(l.seat.y - rr.seat.y))
                    row("Setback", mm(l.seat.z - rr.seat.z))
                }
            }
        }
    }

    // MARK: - Helpers

    private func row(_ title: String, _ value: String, detail: String? = nil) -> some View {
        LabeledContent {
            Text(value).monospacedDigit()
        } label: {
            Text(title)
            if let detail { Text(detail) }
        }
    }

    private func seatMethod(_ m: String) -> String {
        switch m {
        case "crease": return "groove in the scan"
        case "outline": return "ear outline in the scan"
        case "model": return "head model"
        case "model (prior)": return "head model, little scan data"
        default: return "your adjustment"
        }
    }

    private func mm(_ v: Float?) -> String { v.map { String(format: "%.1f mm", $0 * 1000) } ?? "—" }
    private func mmShort(_ v: Float?) -> String { v.map { String(format: "%.1f", $0 * 1000) } ?? "—" }
    private func deg(_ v: Float?) -> String { v.map { String(format: "%.0f°", $0) } ?? "—" }
    private func avg(_ a: Float?, _ b: Float?) -> Float? {
        let v = [a, b].compactMap { $0 }
        return v.isEmpty ? nil : v.reduce(0, +) / Float(v.count)
    }

    private func shareCapture(_ dir: URL) {
        exporting = true
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let zip = try ScanExporter.zipCapture(dir)
                DispatchQueue.main.async {
                    exporting = false
                    exportedFiles = [zip]
                    showShare = true
                }
            } catch {
                DispatchQueue.main.async {
                    exporting = false
                    exportError = error.localizedDescription
                }
            }
        }
    }

    /// Keeps the scan, head model and current adjustments on the phone (home screen › Your scans).
    private func save() {
        saving = true
        let s = session
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try SavedScans.save(s)
                DispatchQueue.main.async {
                    saving = false
                    savedMessage = "Reopen it from Your scans on the home screen. Saving again keeps your latest adjustments."
                }
            } catch {
                DispatchQueue.main.async {
                    saving = false
                    exportError = error.localizedDescription
                }
            }
        }
    }

    private func export() {
        exporting = true
        let s = session
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let files = try ScanExporter.export(s)
                DispatchQueue.main.async {
                    exporting = false
                    exportedFiles = files
                    showShare = true
                }
            } catch {
                DispatchQueue.main.async {
                    exporting = false
                    exportError = error.localizedDescription
                }
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
