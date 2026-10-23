import SceneKit
import SwiftUI
import simd

/// Interactive 3D view of the head mesh with the fit overlay (landmarks, pad contact lines,
/// frame back plane, temple guides). Taps report the hit point on the mesh in head coordinates.
struct HeadSceneView: UIViewRepresentable {
    enum Display: String, CaseIterable, Identifiable {
        case scan = "Scan", model = "Model", both = "Both"
        var id: String { rawValue }
    }

    let mesh: TriMesh
    /// Fitted GNM head (skin), shown instead of / over the scan depending on `display`.
    var modelMesh: TriMesh? = nil
    var display: Display = .scan
    let result: FitResult
    var showEarOutlines = true
    var onTap: ((Vec3) -> Void)?

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero)
        view.backgroundColor = .secondarySystemBackground
        view.antialiasingMode = .multisampling4X
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        let scene = SCNScene()
        view.scene = scene

        let meshNode = SCNNode(geometry: Self.geometry(for: mesh))
        meshNode.name = "head"
        scene.rootNode.addChildNode(meshNode)
        context.coordinator.meshNode = meshNode

        if let modelMesh {
            let model = SCNNode(geometry: Self.geometry(for: modelMesh, color: UIColor(red: 0.45, green: 0.62, blue: 0.86, alpha: 1)))
            model.name = "model"
            scene.rootNode.addChildNode(model)
            context.coordinator.modelNode = model
        }

        let overlay = SCNNode()
        overlay.name = "overlay"
        scene.rootNode.addChildNode(overlay)
        context.coordinator.overlay = overlay

        let target = SCNVector3((result.eyeCenterLeft + result.eyeCenterRight) / 2 + Vec3(0, -0.01, -0.03))
        let camera = SCNCamera()
        camera.zNear = 0.01
        camera.zFar = 3
        camera.fieldOfView = 35
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(target.x + 0.22, target.y + 0.06, target.z + 0.38)
        cameraNode.look(at: target)
        scene.rootNode.addChildNode(cameraNode)
        view.pointOfView = cameraNode
        view.defaultCameraController.target = target
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.inertiaEnabled = true

        let key = SCNNode()
        key.light = SCNLight()
        key.light?.type = .directional
        key.light?.intensity = 900
        key.eulerAngles = SCNVector3(-0.5, 0.6, 0)
        scene.rootNode.addChildNode(key)
        let fill = SCNNode()
        fill.light = SCNLight()
        fill.light?.type = .directional
        fill.light?.intensity = 450
        fill.eulerAngles = SCNVector3(-0.2, -2.2, 0)
        scene.rootNode.addChildNode(fill)
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 250
        scene.rootNode.addChildNode(ambient)

        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tap)
        context.coordinator.view = view
        context.coordinator.update(result: result, showEarOutlines: showEarOutlines)
        context.coordinator.apply(display)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        context.coordinator.onTap = onTap
        context.coordinator.update(result: result, showEarOutlines: showEarOutlines)
        context.coordinator.apply(display)
    }

    func makeCoordinator() -> Coordinator {
        let c = Coordinator()
        c.onTap = onTap
        return c
    }

    final class Coordinator: NSObject {
        weak var view: SCNView?
        var meshNode: SCNNode?
        var modelNode: SCNNode?

        /// Scan only, model only, or the model as a translucent shell over the scan.
        func apply(_ display: Display) {
            guard let modelNode else {
                meshNode?.isHidden = false
                return
            }
            meshNode?.isHidden = display == .model
            modelNode.isHidden = display == .scan
            modelNode.opacity = display == .both ? 0.45 : 1
            modelNode.geometry?.firstMaterial?.writesToDepthBuffer = display != .both
        }
        var overlay: SCNNode?
        var onTap: ((Vec3) -> Void)?

        @objc func handleTap(_ g: UITapGestureRecognizer) {
            guard let view, let meshNode, let onTap else { return }
            let target = meshNode.isHidden ? (modelNode ?? meshNode) : meshNode
            let hits = view.hitTest(g.location(in: view), options: [.rootNode: target, .searchMode: SCNHitTestSearchMode.closest.rawValue])
            if let hit = hits.first {
                let p = hit.localCoordinates
                onTap(Vec3(Float(p.x), Float(p.y), Float(p.z)))
            }
        }

        func update(result r: FitResult, showEarOutlines: Bool) {
            guard let overlay else { return }
            overlay.childNodes.forEach { $0.removeFromParentNode() }

            func sphere(_ p: Vec3, _ color: UIColor, _ radius: CGFloat = 0.0013) {
                let s = SCNSphere(radius: radius)
                s.firstMaterial = HeadSceneView.flat(color)
                let n = SCNNode(geometry: s)
                n.position = SCNVector3(p)
                overlay.addChildNode(n)
            }

            sphere(r.pupilLeft, .systemBlue)
            sphere(r.pupilRight, .systemBlue)
            sphere(r.sellion, .systemYellow, 0.001)
            sphere(r.noseTip, .systemYellow, 0.001)
            sphere(r.crest, .systemOrange, 0.0016)
            overlay.addChildNode(HeadSceneView.polyline(r.padContactLeft, color: .systemPink, radius: 0.0005))
            overlay.addChildNode(HeadSceneView.polyline(r.padContactRight, color: .systemPink, radius: 0.0005))

            // Frame back plane (vertical, resting on the crest)
            let plane = SCNPlane(width: 0.15, height: 0.06)
            let m = HeadSceneView.flat(UIColor.systemTeal.withAlphaComponent(0.18))
            m.isDoubleSided = true
            m.writesToDepthBuffer = false
            plane.firstMaterial = m
            let planeNode = SCNNode(geometry: plane)
            planeNode.position = SCNVector3(r.crest.x, (r.pupilLeft.y + r.pupilRight.y) / 2, r.backPlaneZ)
            overlay.addChildNode(planeNode)

            let pupilY = (r.pupilLeft.y + r.pupilRight.y) / 2
            for ear in [r.earLeft, r.earRight].compactMap({ $0 }) {
                sphere(ear.seat, ear.method == "manual" ? .systemPurple : .systemGreen, 0.0018)
                // Temple guide from a hinge at pupil height in the back plane to the ear seat.
                let hinge = Vec3(ear.seat.x + ear.side.sign * 0.004, pupilY, r.backPlaneZ)
                overlay.addChildNode(HeadSceneView.polyline([hinge, ear.seat], color: .systemGreen, radius: 0.0006))
                if showEarOutlines, !ear.outline.isEmpty {
                    overlay.addChildNode(HeadSceneView.points(ear.outline, color: UIColor.systemGreen.withAlphaComponent(0.7)))
                }
            }
        }
    }

    // MARK: - Geometry helpers

    static func geometry(for mesh: TriMesh, color: UIColor = UIColor(red: 0.86, green: 0.78, blue: 0.72, alpha: 1)) -> SCNGeometry {
        let vData = mesh.vertices.withUnsafeBufferPointer { Data(buffer: $0) }
        let nData = mesh.normals.withUnsafeBufferPointer { Data(buffer: $0) }
        let stride = MemoryLayout<Vec3>.stride
        let vSource = SCNGeometrySource(data: vData, semantic: .vertex, vectorCount: mesh.vertices.count,
                                        usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: stride)
        let nSource = SCNGeometrySource(data: nData, semantic: .normal, vectorCount: mesh.normals.count,
                                        usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4,
                                        dataOffset: 0, dataStride: stride)
        var indices = [UInt32]()
        indices.reserveCapacity(mesh.triangles.count * 3)
        for t in mesh.triangles { indices.append(t.x); indices.append(t.y); indices.append(t.z) }
        let element = SCNGeometryElement(data: indices.withUnsafeBufferPointer { Data(buffer: $0) },
                                         primitiveType: .triangles, primitiveCount: mesh.triangles.count,
                                         bytesPerIndex: 4)
        let g = SCNGeometry(sources: [vSource, nSource], elements: [element])
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = color
        m.roughness.contents = 0.75
        m.isDoubleSided = true
        g.firstMaterial = m
        return g
    }

    static func flat(_ color: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = color
        return m
    }

    static func polyline(_ pts: [Vec3], color: UIColor, radius: CGFloat) -> SCNNode {
        let node = SCNNode()
        guard pts.count >= 2 else { return node }
        let material = flat(color)
        for i in 0..<(pts.count - 1) {
            let a = pts[i], b = pts[i + 1]
            let len = simd_distance(a, b)
            if len < 1e-5 { continue }
            let c = SCNCylinder(radius: radius, height: CGFloat(len))
            c.radialSegmentCount = 8
            c.firstMaterial = material
            let n = SCNNode(geometry: c)
            n.simdPosition = (a + b) / 2
            n.simdOrientation = simd_quatf(from: Vec3(0, 1, 0), to: simd_normalize(b - a))
            node.addChildNode(n)
        }
        return node
    }

    static func points(_ pts: [Vec3], color: UIColor) -> SCNNode {
        let data = pts.withUnsafeBufferPointer { Data(buffer: $0) }
        let source = SCNGeometrySource(data: data, semantic: .vertex, vectorCount: pts.count, usesFloatComponents: true,
                                       componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0,
                                       dataStride: MemoryLayout<Vec3>.stride)
        let idx = (0..<UInt32(pts.count)).map { $0 }
        let element = SCNGeometryElement(data: idx.withUnsafeBufferPointer { Data(buffer: $0) }, primitiveType: .point,
                                         primitiveCount: pts.count, bytesPerIndex: 4)
        element.pointSize = 3
        element.minimumPointScreenSpaceRadius = 1.5
        element.maximumPointScreenSpaceRadius = 3
        let g = SCNGeometry(sources: [source], elements: [element])
        g.firstMaterial = flat(color)
        return SCNNode(geometry: g)
    }
}
