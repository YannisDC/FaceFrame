import SwiftUI

struct HomeView: View {
    var scanSupported: Bool
    var buildingDemo: Bool
    var onScan: () -> Void
    var onDemo: () -> Void
    var onReplay: (URL) -> Void
    var onOpen: (SavedScans.Entry) -> Void
    @State private var captures: [URL] = []
    @State private var saved: [SavedScans.Entry] = []
    @State private var deleting: SavedScans.Entry?
    @State private var showTips = false
    @AppStorage(ScanController.recordCaptureKey) private var recordCapture = false
    @AppStorage(GNMIdentityPrior.sexKey) private var priorSex = ""
    @AppStorage(GNMIdentityPrior.ethnicityKey) private var priorEthnicity = ""

    var body: some View {
        NavigationStack {
            List {
                hero
                if !saved.isEmpty { scans }
                headModel
                more
            }
            .listStyle(.insetGrouped)
            .navigationTitle("FaceFrame")
            .navigationDestination(isPresented: $showTips) { ScanTipsView() }
            .onAppear {
                captures = CaptureReplay.captures()
                saved = SavedScans.list()
            }
            .onReceive(NotificationCenter.default.publisher(for: SavedScans.didChange)) { _ in saved = SavedScans.list() }
            .confirmationDialog("Delete this scan?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                                titleVisibility: .visible) {
                Button("Delete scan and its files", role: .destructive) {
                    if let d = deleting { try? SavedScans.delete(d) }
                    saved = SavedScans.list()
                }
            } message: {
                Text("Removes the scan, its head model, exports and frames from this phone.")
            }
        }
    }

    // MARK: - Sections

    /// What the app does, in three steps, and the one thing to do now.
    private var hero: some View {
        Section {
            VStack(spacing: 22) {
                HStack(alignment: .top, spacing: 0) {
                    step("faceid", "Scan")
                    arrow
                    step("person.crop.circle", "Head model")
                    arrow
                    step("eyeglasses", "Frames")
                }
                .padding(.top, 8)

                Button(action: onScan) {
                    Label("Start scan", systemImage: "faceid")
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 34)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!scanSupported)

                if scanSupported {
                    Button("Tips for a good scan", systemImage: "lightbulb") { showTips = true }
                        .font(.subheadline.weight(.medium))
                        .buttonStyle(.borderless)
                } else {
                    VStack(spacing: 6) {
                        Text("Scanning needs a Face ID iPhone. Try the demo head below.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                        Button("Tips for a good scan", systemImage: "lightbulb") { showTips = true }
                            .font(.subheadline.weight(.medium))
                            .buttonStyle(.borderless)
                    }
                }
            }
            .listRowBackground(Color.clear)
            .listRowInsets(EdgeInsets(top: 0, leading: 4, bottom: 4, trailing: 4))
        }
    }

    private var scans: some View {
        Section {
            ForEach(saved) { entry in
                Button { onOpen(entry) } label: { scanRow(entry) }
                    .tint(.primary)
                    .swipeActions {
                        Button("Delete", systemImage: "trash", role: .destructive) { deleting = entry }
                    }
                    .contextMenu {
                        Button("Delete", systemImage: "trash", role: .destructive) { deleting = entry }
                    }
            }
        } header: {
            Text("Your scans")
        } footer: {
            Text("Exports and frames are in Files › FaceFrame › Scans.")
        }
    }

    private var headModel: some View {
        Section {
            Picker("Sex", selection: $priorSex) {
                Text("Not set").tag("")
                ForEach(GNMIdentityPrior.sexes, id: \.self) { Text($0.capitalized).tag($0) }
            }
            Picker("Ethnicity", selection: $priorEthnicity) {
                Text("Not set").tag("")
                ForEach(GNMIdentityPrior.ethnicities, id: \.self) { Text($0.replacingOccurrences(of: "_", with: " ").capitalized).tag($0) }
            }
            if priorSex.isEmpty != priorEthnicity.isEmpty {
                Label("Set both, or neither.", systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        } header: {
            Text("Head model")
        } footer: {
            Text("Optional. Starts the head model from typical shapes for this group, which helps where the scan sees little.")
        }
        .pickerStyle(.menu)
    }

    private var more: some View {
        Section {
            Button(action: onDemo) {
                HStack {
                    Label(buildingDemo ? "Building demo head…" : "Try with a demo head", systemImage: "cube")
                    Spacer()
                    if buildingDemo { ProgressView() }
                }
            }
            .disabled(buildingDemo)

            if !captures.isEmpty {
                Menu {
                    ForEach(captures, id: \.self) { url in
                        Button(url.lastPathComponent) { onReplay(url) }
                    }
                } label: {
                    Label("Replay a recorded capture", systemImage: "arrow.counterclockwise")
                }
                .disabled(buildingDemo)
            }

            Toggle(isOn: $recordCapture) {
                Label("Record raw capture", systemImage: "record.circle")
            }
        } header: {
            Text("More")
        } footer: {
            Text("Recording keeps every depth and colour frame of the next scans (80–200 MB each) to replay them later.")
        }
    }

    // MARK: - Pieces

    private func step(_ icon: String, _ title: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2.weight(.medium))
                .foregroundStyle(.tint)
                .frame(width: 56, height: 56)
                .background(Color.accentColor.opacity(0.12), in: Circle())
            Text(title)
                .font(.footnote.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
    }

    private var arrow: some View {
        Image(systemName: "chevron.right")
            .font(.footnote.weight(.semibold))
            .foregroundStyle(.tertiary)
            .frame(height: 56)
            .accessibilityHidden(true)
    }

    private func scanRow(_ e: SavedScans.Entry) -> some View {
        let s = e.session
        let model = s.headModel.map { h in h.source == "photos" ? String(format: "Head from photos · ×%.3f", h.headScale) : "Head fitted to scan" }
        return HStack(spacing: 14) {
            Image(systemName: s.isDemo ? "cube" : "faceid")
                .font(.body.weight(.medium))
                .foregroundStyle(.tint)
                .frame(width: 36, height: 36)
                .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(s.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.body.weight(.medium))
                Text([s.isDemo ? "Demo" : nil, model].compactMap { $0 }.joined(separator: " · "))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Image(systemName: "chevron.right")
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.tertiary)
        }
        .contentShape(Rectangle())
    }
}

/// The advice for a good scan, one row per tip.
struct ScanTipsView: View {
    private let tips: [(icon: String, title: String, detail: String)] = [
        ("eyeglasses", "Take off glasses and earbuds", "Also earrings: anything on the ears is scanned as part of the ear."),
        ("ear", "Tuck your hair behind your ears", "The scanner must see where each ear meets the head."),
        ("lightbulb", "Even, indirect light", "Avoid direct sunlight: it blinds the depth sensor."),
        ("iphone", "Phone at eye level, about 30 cm away", "Keep it still and move your head, slowly."),
        ("speaker.wave.2", "Turn the sound on", "Spoken instructions guide you while your head is turned."),
        ("face.smiling", "Neutral expression", "Keep your mouth closed."),
        ("mustache", "Beards are fine", "They scan noisily, but that only affects the chin, which glasses don't need."),
    ]

    var body: some View {
        List(tips, id: \.title) { tip in
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: tip.icon)
                    .font(.body.weight(.medium))
                    .foregroundStyle(.tint)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 3) {
                    Text(tip.title).font(.body.weight(.medium))
                    Text(tip.detail).font(.subheadline).foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 4)
            .accessibilityElement(children: .combine)
        }
        .navigationTitle("Scanning tips")
        .navigationBarTitleDisplayMode(.inline)
    }
}
