import Combine
import Foundation
import simd

/// A finished scan: the head mesh, the automatic fit analysis and any landmarks the user moved.
final class FitSession: ObservableObject, Identifiable {
    let id = UUID()
    let mesh: TriMesh
    let analyzer: FitAnalyzer
    let createdAt: Date
    let isDemo: Bool
    let diagnostics: ScanDiagnostics?
    /// Raw TrueDepth capture recorded during this scan, if recording was on.
    var captureDirectory: URL?
    /// GNM head model fitted to the scan (complete head, skin only), if the fit succeeded.
    let modelFit: ModelFitSummary?

    @Published var overrides = FitOverrides() {
        didSet { result = analyzer.analyze(overrides: overrides) }
    }
    @Published private(set) var result: FitResult

    init(mesh: TriMesh, analyzer: FitAnalyzer, isDemo: Bool = false, diagnostics: ScanDiagnostics? = nil,
         modelFit: ModelFitSummary? = nil, createdAt: Date = Date()) {
        self.createdAt = createdAt
        self.mesh = mesh
        self.analyzer = analyzer
        self.isDemo = isDemo
        self.diagnostics = diagnostics
        self.modelFit = modelFit
        result = analyzer.analyze()
    }
}

enum HeadReconstruction {
    static var extractionMinWeight: UInt8 = 6
    static var fitModel = true
    static var removeHiddenSurface = true
    static var cleanWithModel = true
    /// Taubin passes on the fused surface (removes the ~1 mm "cobblestone" sensor noise; volume preserving).
    static var smoothingIterations = 4
    /// Fit the head model to the colour frames (GNM's photo fit) and size it on the scan; when off, or
    /// when the frames do not show a front and a side view, the model is fitted to the scan instead.
    static var photoFit = true
    /// Anchor the photo fit to the pupils and ear seats measured on the scan.
    static var useAnchors = true
    /// Detailed progress (the replay tool prints it).
    static var log: (String) -> Void = { _ in }
    /// Sees the photo fit's observations and result (the replay tool saves them).
    static var onPhotoFit: ([PhotoObservation], PhotoFitter.Result) -> Void = { _, _ in }

    /// Volume -> cleaned, lightly smoothed mesh -> fit analysis.
    /// Colour frames for measuring the pupils: samples plus ARKit's raw eye centres and the scan's
    /// ARKit correction.
    struct PupilInput {
        var samples: [PupilEstimator.Sample]
        var arkitEyes: (left: Vec3, right: Vec3)
        var correction: PoseCorrection
    }

    /// Colour frames for the photo fit, with ARKit's eye centres (its face frame).
    struct PhotoInput {
        var frames: [PhotoObserver.Frame]
        var arkitEyes: (left: Vec3, right: Vec3)
    }

    static func build(volume: TSDFVolume, eyeLeft: Vec3, eyeRight: Vec3, isDemo: Bool = false, diagnostics: ScanDiagnostics? = nil,
                      pupils: PupilInput? = nil, photos: PhotoInput? = nil, prior: GNMIdentityPrior? = GNMIdentityPrior.selected,
                      progress: (String) -> Void = { _ in }) -> FitSession? {
        progress("Extracting surface…")
        // Frames add 1–4 weight depending on viewing angle: require ~2 frontal or ~6 grazing observations.
        var mesh = SurfaceNets.extract(volume, minWeight: isDemo ? 1 : extractionMinWeight)
        guard mesh.triangles.count > 10_000 else { return nil }
        progress("Cleaning mesh…")
        if !isDemo && removeHiddenSurface { mesh.keepOutwardVisible() }
        mesh.keepLargestComponent()
        mesh.taubinSmooth(iterations: smoothingIterations)
        mesh.computeNormals()
        // Pupils from the colour frames, measured before the head model so that the model's eyes,
        // which the depth scan cannot see, are fitted to them.
        var measuredPupils: PupilEstimator.Result?
        if let pupils, !pupils.samples.isEmpty {
            progress("Measuring pupils…")
            measuredPupils = PupilEstimator().estimate(pupils.samples, arkitEyes: pupils.arkitEyes, correction: pupils.correction, surface: mesh)
            if measuredPupils == nil { log("pupils: not measured in \(pupils.samples.count) frames (\(PupilEstimator.lastFailure))") }
        }
        // Fit the head model first: it tells real skin from artifacts (earbuds, stubble fuzz, the
        // smear under the jaw), which are removed before measuring.
        // The head model fitted to the scan cleans the scan (it tells real skin from artifacts: earbuds,
        // stubble fuzz, the smear under the jaw) and anchors the nose and ear measurements. The head the
        // frames are made for is GNM's photo fit on the colour frames, its identity fixed and its size
        // taken from the scan; without usable photos, it is this scan fit.
        var fit: (model: GNMModel, result: GNMFitter.Result)? = nil
        var photoHead: GNMFitter.Result?
        if fitModel, let model = GNMModel.bundled() {
            progress("Fitting head model…")
            var fitter = GNMFitter(model: model)
            fitter.prior = prior
            if let p = measuredPupils {
                fitter.landmarks = [.init(vertices: model.pupilVertices(.left), target: p.left),
                                    .init(vertices: model.pupilVertices(.right), target: p.right)]
            }
            fit = fitter.fit(scan: mesh, leftEye: eyeLeft, rightEye: eyeRight).map { (model, $0) }
            if let fit, !isDemo && cleanWithModel {
                progress("Removing artifacts…")
                mesh = cleanScan(mesh, model: model, fit: fit.result, eyeLeft: eyeLeft, eyeRight: eyeRight)
            }
        }
        progress("Measuring nose and ears…")
        let analyzer = FitAnalyzer(mesh: mesh, eyeCenterLeft: eyeLeft, eyeCenterRight: eyeRight)
        var modelFit = fit.map { applyHeadModel($0.model, fit: $0.result, eyeLeft: eyeLeft, eyeRight: eyeRight, analyzer: analyzer) }
        modelFit?.prior = prior.map { "\($0.sex)/\($0.ethnicity)" }
        if let p = measuredPupils {
            analyzer.setMeasuredPupils(left: p.left, right: p.right, leftShare: p.leftShare)
        }
        // The photo fit, anchored to what the scan measures: the pupils and the ear seats.
        if photoFit, let photos, let fit {
            var anchors: [PhotoFitter.Anchor] = []
            func d(_ v: Vec3) -> SIMD3<Double> { SIMD3(Double(v.x), Double(v.y), Double(v.z)) }
            if let p = measuredPupils {
                anchors += [.init(name: "left pupil", vertices: fit.model.pupilVertices(.left), target: d(p.left)),
                            .init(name: "right pupil", vertices: fit.model.pupilVertices(.right), target: d(p.right))]
            }
            let measured = analyzer.analyze()
            for side in HeadSide.allCases {
                guard let seat = measured.ear(side)?.seat else { continue }
                // The GNM vertex at that seat on the head fitted to the scan.
                let v = fit.result.statisticalVertices
                if let i = (0..<v.count).filter({ fit.model.member("ears", $0) && side.sign * v[$0].x > 0 })
                    .min(by: { simd_distance(v[$0], seat) < simd_distance(v[$1], seat) }) {
                    anchors.append(.init(name: "\(side.rawValue) ear seat", vertices: [i], target: d(seat)))
                }
            }
            photoHead = fitToPhotos(photos, scan: mesh, eyeLeft: eyeLeft, eyeRight: eyeRight, prior: prior, anchors: anchors, progress: progress)
        }
        if let head = photoHead, let model = fit?.model {
            modelFit?.source = "photos"
            modelFit?.identity = head.coefficients
            modelFit?.headScale = head.scale
            modelFit?.mesh = head.skinMesh(model)
            modelFit?.rmsMM = head.rmsMM
        }
        return FitSession(mesh: mesh, analyzer: analyzer, isDemo: isDemo, diagnostics: diagnostics, modelFit: modelFit)
    }

    /// GNM's photo fit on the colour frames (identity and expression), then the head, identity fixed,
    /// aligned with the scan for its pose and metric scale (Tools/gnm_align_to_scan.py).
    static func fitToPhotos(_ photos: PhotoInput, scan: TriMesh, eyeLeft: Vec3, eyeRight: Vec3, prior: GNMIdentityPrior?,
                            anchors: [PhotoFitter.Anchor] = [], progress: (String) -> Void) -> GNMFitter.Result? {
        progress("Finding the face in the photos…")
        let start = Date()
        func elapsed() -> String { String(format: "%.1f s", Date().timeIntervalSince(start)) }
        guard let dir = GNMModel.resourceDirectory, let photoModel = try? GNMPhotoModel(directory: dir) else {
            log("photo fit: model files missing"); return nil
        }
        guard let observer = try? PhotoObserver(model: photoModel) else { log("photo fit: Core ML models missing"); return nil }
        observer.log = log
        let views = PhotoObserver.selectViews(photos.frames)
        let observations = views.compactMap { observer.observe($0, arkitEyes: photos.arkitEyes) }
        log("photo fit: \(observations.count) of \(views.count) views usable (\(photos.frames.count) frames), detection done at \(elapsed())")
        guard observations.count >= 2, observations.contains(where: { PhotoFitter.isSideView($0.initialRotation) }) else { return nil }
        progress("Fitting head model to the photos…")
        var fitter = PhotoFitter(model: photoModel)
        fitter.prior = prior
        fitter.anchors = useAnchors ? anchors : []
        let photo = fitter.fit(observations, log: log)
        onPhotoFit(observations, photo)
        guard photo.identity.allSatisfy(\.isFinite) else { log("photo fit: diverged"); return nil }
        log("photo fit: done at \(elapsed())")
        log("photo fit: landmark error per view (px) " + photo.landmarkErrorPixels.map { String(format: "%.1f", $0) }.joined(separator: " / "))
        progress("Sizing the head on the scan…")
        // With anchors, their similarity gives the scale: they span the head (pupils to ear seats), where the
        // midline profile's error barely changes between scales 0.94 and 1.04 on a real scan.
        guard let aligned = GNMScaleAligner(model: photoModel, identity: photo.identity, expression: photo.expression)
            .align(scan: scan, eyeLeft: eyeLeft, eyeRight: eyeRight, scale: photo.anchorScale.map(Float.init), log: log) else { return nil }
        log("photo fit: sized on the scan at \(elapsed())")
        let grid = PointGrid(points: scan.vertices, normals: scan.normals, cell: 0.004)
        let support: [Float?] = aligned.vertices.map { p in grid.nearest(p, maxDistance: 0.006).map { simd_distance($0.0, p) } }
        return GNMFitter.Result(coefficients: photo.identity, scale: aligned.scale, rotation: aligned.rotation, translation: aligned.translation,
                                vertices: aligned.vertices, statisticalVertices: aligned.vertices, rmsMM: aligned.rmsMM, support: support)
    }

    /// Removes scan surface the fitted head model identifies as artifacts: everything below the chin
    /// (throat, neck, collar: not needed for glasses and badly seen by TrueDepth), and, on the face,
    /// cheeks, temples and ears, surface more than `maxDistance` off the fitted skin (earbuds,
    /// stubble fuzz, the smear under the jaw). Hair over the cranium is kept (the model has no hair).
    static func cleanScan(_ scan: TriMesh, model: GNMModel, fit r: GNMFitter.Result, eyeLeft: Vec3, eyeRight: Vec3,
                          maxDistance: Float = 0.005) -> TriMesh {
        let w = r.vertices
        let midX = (eyeLeft.x + eyeRight.x) / 2
        let chinY = (0..<model.vertexCount)
            .filter { model.member("hockey_mask", $0) && model.member("skin", $0) && abs(w[$0].x - midX) < 0.012 }
            .map { w[$0].y }.min() ?? -.greatestFiniteMagnitude
        var skin = TriMesh()
        skin.vertices = w
        skin.triangles = model.triangles.filter {
            model.member("skin", Int($0.x)) && model.member("skin", Int($0.y)) && model.member("skin", Int($0.z))
        }
        skin.computeNormals()
        let groups = ["hockey_mask", "ears", "left_parotid_region", "right_parotid_region", "left_zygomatic_region",
                      "right_zygomatic_region", "left_temple_region", "right_temple_region"]
        let checked = (0..<model.vertexCount).filter { i in model.member("skin", i) && groups.contains { model.member($0, i) } }
        // Cells of half the search radius: queries up to 20 mm look at 125 cells, not 1331.
        let grid = PointGrid(points: checked.map { w[$0] }, normals: checked.map { skin.normals[$0] }, cell: 0.01)

        var out = scan
        out.triangles = scan.triangles.filter { t in
            let c = (scan.vertices[Int(t.x)] + scan.vertices[Int(t.y)] + scan.vertices[Int(t.z)]) / 3
            if c.y < chinY - 0.003 { return false }
            guard let (q, n) = grid.nearest(c, maxDistance: 0.02) else { return true }
            return abs(simd_dot(n, c - q)) <= maxDistance
        }
        out.compact()
        out.keepLargestComponent()
        out.computeNormals()
        return out
    }

    /// Takes the ear seats and the crest height from the fitted head model: TrueDepth only sees the
    /// ears obliquely and noisily, while the fitted model has a complete, clean ear anchored to the
    /// measured surface. The nose stays measured on the (cleaned) scan.
    static func applyHeadModel(_ model: GNMModel, fit r: GNMFitter.Result, eyeLeft: Vec3, eyeRight: Vec3,
                               analyzer: FitAnalyzer) -> ModelFitSummary {
        // Anchor the crest height from the model's sellion: found on the smooth fitted surface it is
        // stable across scans, where the scan's own profile let it jump by up to ~10 mm.
        if let sv = model.sellionVertex { analyzer.anchorSellion(y: r.vertices[sv].y) }
        let seats = r.earSeats(model, eyeLeft: eyeLeft, eyeRight: eyeRight)
        let scanOnly = analyzer.analyze()
        var warnings: [String] = []
        var support: [HeadSide: (Float, Float)] = [:]
        var tagged: [HeadSide: EarFit] = [:]
        for (side, seat) in seats {
            let s = r.earSupport(model, side: side)
            support[side] = s
            var e = seat
            if s.fraction < 0.5 || s.median > 0.0015 {
                e.method = "model (prior)"
                warnings.append(String(format: "The %@ ear is mostly estimated from the head model (%.0f%% covered by the scan). Rescan turning further to that side for a measured seat.",
                                       side.rawValue, s.fraction * 100))
            } else {
                e.method = "model"
            }
            if let scanEar = scanOnly.ear(side), simd_distance(scanEar.seat, e.seat) > 0.008 {
                warnings.append(String(format: "Scan and head model disagree on the %@ ear seat by %.0f mm; check it in the 3D view and adjust if needed.",
                                       side.rawValue, simd_distance(scanEar.seat, e.seat) * 1000))
            }
            tagged[side] = e
        }
        analyzer.setModelEars(tagged, warnings: warnings)
        return ModelFitSummary(mesh: r.skinMesh(model), rmsMM: r.rmsMM, identity: r.coefficients,
                               earCoverage: support.mapValues { $0.0 },
                               earDistanceMM: support.filter { $0.value.1.isFinite }.mapValues { $0.1 * 1000 })
    }

    /// Synthetic head for trying the review and export flow without a TrueDepth device.
    static func demo(progress: (String) -> Void = { _ in }) -> FitSession? {
        let head = SyntheticHead()
        let mid = (head.eyeCenterLeft + head.eyeCenterRight) / 2
        let volume = TSDFVolume(origin: mid + Vec3(-0.125, -0.16, -0.21), extent: Vec3(0.25, 0.29, 0.30),
                                voxelSize: FusionEngine.voxelSize)
        progress("Generating demo head…")
        volume.fill(sdf: head.distance)
        return build(volume: volume, eyeLeft: head.eyeCenterLeft, eyeRight: head.eyeCenterRight, isDemo: true, progress: progress)
    }
}

struct ModelFitSummary {
    /// The head model's skin surface in the head frame (metres): the photo-fitted head when there is one.
    var mesh: TriMesh
    var rmsMM: Float
    /// GNM identity coefficients of the statistical fit (metric: GNM and TrueDepth are both in metres).
    var identity: [Float]
    /// The sex/ethnicity class the identity prior was centred on ("male/white"), nil for GNM's default prior.
    var prior: String?
    /// "photos": identity from GNM's photo fit, scale from the scan; "scan": fitted to the scan.
    var source = "scan"
    /// Uniform scale of the head (GNMFrames' headScale): the metric size the scan gives a photo fit.
    var headScale: Float = 1
    var earCoverage: [HeadSide: Float]
    var earDistanceMM: [HeadSide: Float]
}
