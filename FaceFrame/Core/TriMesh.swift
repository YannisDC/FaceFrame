import Foundation
import simd

/// Indexed triangle mesh in the head frame (meters).
struct TriMesh {
    var vertices: [Vec3] = []
    var normals: [Vec3] = []
    var triangles: [SIMD3<UInt32>] = []

    var isEmpty: Bool { triangles.isEmpty }

    var bounds: (min: Vec3, max: Vec3) {
        var lo = Vec3(repeating: .greatestFiniteMagnitude)
        var hi = Vec3(repeating: -.greatestFiniteMagnitude)
        for v in vertices {
            lo = simd_min(lo, v)
            hi = simd_max(hi, v)
        }
        return (lo, hi)
    }

    mutating func computeNormals() {
        var n = [Vec3](repeating: .zero, count: vertices.count)
        for t in triangles {
            let a = vertices[Int(t.x)], b = vertices[Int(t.y)], c = vertices[Int(t.z)]
            let fn = simd_cross(b - a, c - a) // area weighted
            n[Int(t.x)] += fn
            n[Int(t.y)] += fn
            n[Int(t.z)] += fn
        }
        normals = n.map { simd_length($0) > 0 ? simd_normalize($0) : Vec3(0, 0, 1) }
    }

    /// Keeps only the largest connected component (drops floating noise such as stray hair or hands).
    mutating func keepLargestComponent() {
        guard !triangles.isEmpty else { return }
        var parent = [Int32](0..<Int32(vertices.count))
        func find(_ x: Int32) -> Int32 {
            var x = x
            while parent[Int(x)] != x {
                parent[Int(x)] = parent[Int(parent[Int(x)])]
                x = parent[Int(x)]
            }
            return x
        }
        for t in triangles {
            let a = find(Int32(t.x)), b = find(Int32(t.y)), c = find(Int32(t.z))
            if a != b { parent[Int(a)] = b }
            let bb = find(b)
            if c != bb { parent[Int(c)] = bb }
        }
        var counts = [Int32: Int]()
        for t in triangles { counts[find(Int32(t.x)), default: 0] += 1 }
        guard let best = counts.max(by: { $0.value < $1.value })?.key else { return }
        triangles = triangles.filter { find(Int32($0.x)) == best }
        compact()
    }

    /// Removes unreferenced vertices.
    mutating func compact() {
        var remap = [Int32](repeating: -1, count: vertices.count)
        var newVerts: [Vec3] = []
        newVerts.reserveCapacity(vertices.count)
        for t in triangles {
            for i in [t.x, t.y, t.z] where remap[Int(i)] < 0 {
                remap[Int(i)] = Int32(newVerts.count)
                newVerts.append(vertices[Int(i)])
            }
        }
        triangles = triangles.map { SIMD3(UInt32(remap[Int($0.x)]), UInt32(remap[Int($0.y)]), UInt32(remap[Int($0.z)])) }
        vertices = newVerts
        normals = []
    }

    /// Taubin (lambda/mu) smoothing: removes sensor noise without shrinking the surface.
    mutating func taubinSmooth(iterations: Int, lambda: Float = 0.5, mu: Float = -0.53) {
        guard iterations > 0, !vertices.isEmpty else { return }
        // CSR adjacency
        var degree = [Int32](repeating: 0, count: vertices.count)
        for t in triangles {
            degree[Int(t.x)] += 2; degree[Int(t.y)] += 2; degree[Int(t.z)] += 2
        }
        var offsets = [Int](repeating: 0, count: vertices.count + 1)
        for i in 0..<vertices.count { offsets[i + 1] = offsets[i] + Int(degree[i]) }
        var adj = [UInt32](repeating: 0, count: offsets[vertices.count])
        var fill = offsets
        for t in triangles {
            let e = [(t.x, t.y), (t.y, t.z), (t.z, t.x)]
            for (a, b) in e {
                adj[fill[Int(a)]] = b; fill[Int(a)] += 1
                adj[fill[Int(b)]] = a; fill[Int(b)] += 1
            }
        }
        var pos = vertices
        var tmp = vertices
        func pass(_ factor: Float) {
            for i in 0..<pos.count {
                let s = offsets[i], e = offsets[i + 1]
                if s == e { tmp[i] = pos[i]; continue }
                var avg = Vec3.zero
                for k in s..<e { avg += pos[Int(adj[k])] }
                avg /= Float(e - s)
                tmp[i] = pos[i] + factor * (avg - pos[i])
            }
            swap(&pos, &tmp)
        }
        for _ in 0..<iterations {
            pass(lambda)
            pass(mu)
        }
        vertices = pos
    }

    /// Removes surface that cannot be seen from outside: TSDF fusion of noisy depth leaves buried
    /// "inner skins" a few millimetres under the real one. Each triangle is kept if, from at least one
    /// of `views` directions around the object, nothing lies more than `tolerance` in front of it.
    mutating func keepOutwardVisible(views: Int = 26, resolution: Float = 0.0005, tolerance: Float = 0.002) {
        guard !triangles.isEmpty else { return }
        let verts = vertices
        let tris = triangles
        let centroids = tris.map { (verts[Int($0.x)] + verts[Int($0.y)] + verts[Int($0.z)]) / 3 }
        // Directions on a Fibonacci sphere.
        let dirs: [Vec3] = (0..<views).map { i in
            let y = 1 - 2 * (Float(i) + 0.5) / Float(views)
            let r = (1 - y * y).squareRoot()
            let phi = Float(i) * Float.pi * (3 - Float(5).squareRoot())
            return Vec3(cos(phi) * r, y, sin(phi) * r)
        }
        var visible = [UInt8](repeating: 0, count: tris.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: dirs.count) { di in
            let d = dirs[di]
            let helper = abs(d.y) < 0.9 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)
            let u = simd_normalize(simd_cross(helper, d)), v = simd_cross(d, u)
            var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            let proj = verts.map { p -> SIMD3<Float> in
                let q = SIMD3<Float>(simd_dot(p, u), simd_dot(p, v), simd_dot(p, d))
                return q
            }
            for q in proj { lo = simd_min(lo, SIMD2(q.x, q.y)); hi = simd_max(hi, SIMD2(q.x, q.y)) }
            let w = Int((hi.x - lo.x) / resolution) + 2, h = Int((hi.y - lo.y) / resolution) + 2
            var depth = [Float](repeating: -.greatestFiniteMagnitude, count: w * h)
            // Depth buffer (largest coordinate along d = closest to a viewer at +d).
            for t in tris {
                let a = proj[Int(t.x)], b = proj[Int(t.y)], c = proj[Int(t.z)]
                let ax = (a.x - lo.x) / resolution, ay = (a.y - lo.y) / resolution
                let bx = (b.x - lo.x) / resolution, by = (b.y - lo.y) / resolution
                let cx = (c.x - lo.x) / resolution, cy = (c.y - lo.y) / resolution
                // Conservative depth buffer: each triangle claims its bounding box at its front-most
                // depth. Overestimates occlusion by < 1 mm on steep surfaces, which `tolerance` absorbs,
                // and leaves no gaps through which buried layers would look visible.
                let x0 = max(0, Int(min(ax, bx, cx))), x1 = min(w - 1, Int(max(ax, bx, cx)) + 1)
                let y0 = max(0, Int(min(ay, by, cy))), y1 = min(h - 1, Int(max(ay, by, cy)) + 1)
                let zmax = max(a.z, b.z, c.z)
                for y in y0...y1 { for x in x0...x1 { let i = y * w + x; if zmax > depth[i] { depth[i] = zmax } } }
            }
            var local = [Int32]()
            for (ti, cen) in centroids.enumerated() {
                // Compare against the frontmost surface in the surrounding 2x2 pixels (sub-pixel
                // triangles often cover no pixel centre themselves; empty pixels are ignored).
                let fx = (simd_dot(cen, u) - lo.x) / resolution, fy = (simd_dot(cen, v) - lo.y) / resolution
                let x = Int(fx), y = Int(fy)
                guard x >= 0, y >= 0, x + 1 < w, y + 1 < h else { continue }
                let cd = simd_dot(cen, d)
                let front = max(depth[y * w + x], depth[y * w + x + 1], depth[(y + 1) * w + x], depth[(y + 1) * w + x + 1])
                if cd >= front - tolerance { local.append(Int32(ti)) }
            }
            lock.lock()
            for ti in local { visible[Int(ti)] = 1 }
            lock.unlock()
        }
        triangles = zip(tris, visible).filter { $0.1 != 0 }.map { $0.0 }
        compact()
    }

    func transformed(_ m: simd_float4x4, scale: Float = 1) -> TriMesh {
        var out = self
        out.vertices = vertices.map { m.transformPoint($0) * scale }
        out.normals = normals.map { simd_normalize(m.transformDirection($0)) }
        return out
    }
}

// MARK: - Export

extension TriMesh {
    /// Binary STL. `scale` converts meters to the export unit (1000 = millimetres).
    func stlData(scale: Float = 1000) -> Data {
        var data = Data(capacity: 84 + triangles.count * 50)
        var header = [UInt8](repeating: 0, count: 80)
        let title = Array("FaceFrame head scan (mm)".utf8)
        header.replaceSubrange(0..<title.count, with: title)
        data.append(contentsOf: header)
        appendLE(&data, UInt32(triangles.count))
        for t in triangles {
            let a = vertices[Int(t.x)] * scale, b = vertices[Int(t.y)] * scale, c = vertices[Int(t.z)] * scale
            var n = simd_cross(b - a, c - a)
            n = simd_length(n) > 0 ? simd_normalize(n) : .zero
            for v in [n, a, b, c] {
                appendLE(&data, v.x.bitPattern); appendLE(&data, v.y.bitPattern); appendLE(&data, v.z.bitPattern)
            }
            appendLE(&data, UInt16(0))
        }
        return data
    }

    func objData(scale: Float = 1000) -> Data {
        var s = "# FaceFrame head scan, units: millimetres\n"
        s.reserveCapacity(vertices.count * 40 + triangles.count * 30)
        for v in vertices {
            s += String(format: "v %.4f %.4f %.4f\n", v.x * scale, v.y * scale, v.z * scale)
        }
        for n in normals {
            s += String(format: "vn %.4f %.4f %.4f\n", n.x, n.y, n.z)
        }
        let hasN = normals.count == vertices.count
        for t in triangles {
            let a = t.x + 1, b = t.y + 1, c = t.z + 1
            s += hasN ? "f \(a)//\(a) \(b)//\(b) \(c)//\(c)\n" : "f \(a) \(b) \(c)\n"
        }
        return Data(s.utf8)
    }

    func plyData(scale: Float = 1000) -> Data {
        let hasN = normals.count == vertices.count
        var header = "ply\nformat binary_little_endian 1.0\ncomment FaceFrame head scan, units: millimetres\n"
        header += "element vertex \(vertices.count)\nproperty float x\nproperty float y\nproperty float z\n"
        if hasN { header += "property float nx\nproperty float ny\nproperty float nz\n" }
        header += "element face \(triangles.count)\nproperty list uchar uint vertex_indices\nend_header\n"
        var data = Data(header.utf8)
        for (i, v) in vertices.enumerated() {
            let p = v * scale
            appendLE(&data, p.x.bitPattern); appendLE(&data, p.y.bitPattern); appendLE(&data, p.z.bitPattern)
            if hasN {
                let n = normals[i]
                appendLE(&data, n.x.bitPattern); appendLE(&data, n.y.bitPattern); appendLE(&data, n.z.bitPattern)
            }
        }
        for t in triangles {
            data.append(3)
            appendLE(&data, t.x); appendLE(&data, t.y); appendLE(&data, t.z)
        }
        return data
    }
}

@inline(__always) private func appendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}
