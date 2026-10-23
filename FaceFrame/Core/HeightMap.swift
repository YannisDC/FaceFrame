import simd

/// Orthographic "depth image" of a mesh: for every (u, v) cell it stores the largest coordinate of
/// the surface along `depthAxis`, i.e. the surface closest to a viewer looking down -depthAxis.
struct HeightMap {
    let uAxis: Vec3
    let vAxis: Vec3
    let depthAxis: Vec3
    let u0: Float
    let v0: Float
    let resolution: Float
    let width: Int
    let height: Int
    private(set) var values: [Float]

    static let empty = -Float.greatestFiniteMagnitude

    init(mesh: TriMesh, uAxis: Vec3, vAxis: Vec3, depthAxis: Vec3,
         uRange: ClosedRange<Float>, vRange: ClosedRange<Float>, resolution: Float,
         include: ((Vec3, Vec3) -> Bool)? = nil) {
        self.uAxis = uAxis
        self.vAxis = vAxis
        self.depthAxis = depthAxis
        self.u0 = uRange.lowerBound
        self.v0 = vRange.lowerBound
        self.resolution = resolution
        width = Int(((uRange.upperBound - uRange.lowerBound) / resolution).rounded(.up)) + 1
        height = Int(((vRange.upperBound - vRange.lowerBound) / resolution).rounded(.up)) + 1
        values = [Float](repeating: HeightMap.empty, count: width * height)

        let verts = mesh.vertices
        let proj = verts.map { SIMD3<Float>(simd_dot($0, uAxis), simd_dot($0, vAxis), simd_dot($0, depthAxis)) }
        let w = width, h = height
        values.withUnsafeMutableBufferPointer { out in
            for t in mesh.triangles {
                let a = proj[Int(t.x)], b = proj[Int(t.y)], c = proj[Int(t.z)]
                if let include {
                    let pa = verts[Int(t.x)], pb = verts[Int(t.y)], pc = verts[Int(t.z)]
                    let n = simd_cross(pb - pa, pc - pa)
                    let nn = simd_length(n) > 0 ? simd_normalize(n) : Vec3.zero
                    if !include((pa + pb + pc) / 3, nn) { continue }
                }
                // Grid-space coordinates
                let ax = (a.x - u0) / resolution, ay = (a.y - v0) / resolution
                let bx = (b.x - u0) / resolution, by = (b.y - v0) / resolution
                let cx = (c.x - u0) / resolution, cy = (c.y - v0) / resolution
                let minX = max(0, Int(min(ax, bx, cx).rounded(.up)))
                let maxX = min(w - 1, Int(max(ax, bx, cx).rounded(.down)))
                let minY = max(0, Int(min(ay, by, cy).rounded(.up)))
                let maxY = min(h - 1, Int(max(ay, by, cy).rounded(.down)))
                if minX > maxX || minY > maxY { continue }
                let den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                if abs(den) < 1e-12 { continue }
                let inv = 1 / den
                for y in minY...maxY {
                    let py = Float(y)
                    for x in minX...maxX {
                        let px = Float(x)
                        let l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) * inv
                        let l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) * inv
                        let l2 = 1 - l0 - l1
                        if l0 < -1e-4 || l1 < -1e-4 || l2 < -1e-4 { continue }
                        let d = l0 * a.z + l1 * b.z + l2 * c.z
                        let i = y * w + x
                        if d > out[i] { out[i] = d }
                    }
                }
            }
        }
    }

    @inline(__always) func value(col: Int, row: Int) -> Float? {
        guard col >= 0, row >= 0, col < width, row < height else { return nil }
        let v = values[row * width + col]
        return v == HeightMap.empty ? nil : v
    }

    @inline(__always) func col(forU u: Float) -> Int { Int(((u - u0) / resolution).rounded()) }
    @inline(__always) func row(forV v: Float) -> Int { Int(((v - v0) / resolution).rounded()) }
    @inline(__always) func u(col: Int) -> Float { u0 + Float(col) * resolution }
    @inline(__always) func v(row: Int) -> Float { v0 + Float(row) * resolution }

    func value(u: Float, v: Float) -> Float? { value(col: col(forU: u), row: row(forV: v)) }

    /// World position of a cell given its depth value.
    func point(col: Int, row: Int, depth: Float) -> Vec3 {
        uAxis * u(col: col) + vAxis * v(row: row) + depthAxis * depth
    }

    func point(u: Float, v: Float, depth: Float) -> Vec3 {
        uAxis * u + vAxis * v + depthAxis * depth
    }
}
