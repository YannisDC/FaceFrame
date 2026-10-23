import simd

/// Extracts the zero level set of a `TSDFVolume` with naive Surface Nets: one vertex per
/// sign-changing cell (placed at the mean of its edge crossings), one quad per sign-changing edge.
/// The grid points are voxel centers.
enum SurfaceNets {
    private static let cornerOffsets: [SIMD3<Int32>] = (0..<8).map {
        SIMD3<Int32>(Int32($0 & 1), Int32(($0 >> 1) & 1), Int32(($0 >> 2) & 1))
    }
    private static let edges: [(Int, Int)] = {
        var e: [(Int, Int)] = []
        for i in 0..<8 {
            for axis in 0..<3 where i & (1 << axis) == 0 {
                e.append((i, i | (1 << axis)))
            }
        }
        return e
    }()

    static func extract(_ volume: TSDFVolume, minWeight: UInt8 = 3) -> TriMesh {
        let dims = volume.dims
        @inline(__always) func key(_ v: SIMD3<Int32>) -> Int {
            Int(v.x) + Int(dims.x) * (Int(v.y) + Int(dims.y) * Int(v.z))
        }

        var mesh = TriMesh()
        var cellVertex = [Int: UInt32]()
        var cells: [(coord: SIMD3<Int32>, mask: UInt8)] = []
        let tsdf = volume.tsdf
        let weight = volume.weight
        let skip = Int16(0.6 * 32767)

        for (blockIndex, b) in volume.blockCoords.enumerated() {
            let base = b &<< TSDFVolume.blockShift
            for li in 0..<TSDFVolume.voxelsPerBlock {
                let s0 = blockIndex << 9 | li
                if weight[s0] < minWeight || abs(tsdf[s0]) > skip { continue }
                let l = SIMD3<Int32>(Int32(li & 7), Int32((li >> 3) & 7), Int32(li >> 6))
                let c = base &+ l
                var values = SIMD8<Float>()
                var mask: UInt8 = 0
                var ok = true
                for i in 0..<8 {
                    let idx = i == 0 ? s0 : volume.storageIndex(c &+ cornerOffsets[i])
                    if idx < 0 || weight[idx] < minWeight { ok = false; break }
                    let v = Float(tsdf[idx])
                    values[i] = v
                    if v < 0 { mask |= 1 << UInt8(i) }
                }
                if !ok || mask == 0 || mask == 0xFF { continue }

                var acc = Vec3.zero
                var n: Float = 0
                for (a, bb) in edges {
                    let va = values[a], vb = values[bb]
                    if (va < 0) == (vb < 0) { continue }
                    let t = va / (va - vb)
                    acc += Vec3(cornerOffsets[a]) + t * Vec3(cornerOffsets[bb] &- cornerOffsets[a])
                    n += 1
                }
                let p = volume.origin + (Vec3(c) + 0.5 + acc / n) * volume.voxelSize
                cellVertex[key(c)] = UInt32(mesh.vertices.count)
                mesh.vertices.append(p)
                cells.append((c, mask))
            }
        }

        for (c, mask) in cells {
            guard let v0 = cellVertex[key(c)] else { continue }
            let inside0 = mask & 1 != 0
            for axis in 0..<3 {
                let inside1 = mask & (1 << UInt8(1 << axis)) != 0 // corner (1 << axis) is the edge's other end
                if inside0 == inside1 { continue }
                let iu = (axis + 1) % 3, iv = (axis + 2) % 3
                var du = SIMD3<Int32>(repeating: 0); du[iu] = 1
                var dv = SIMD3<Int32>(repeating: 0); dv[iv] = 1
                guard let vu = cellVertex[key(c &- du)],
                      let vuv = cellVertex[key(c &- du &- dv)],
                      let vv = cellVertex[key(c &- dv)] else { continue }
                let quad: (UInt32, UInt32, UInt32, UInt32) = inside0 ? (v0, vu, vuv, vv) : (v0, vv, vuv, vu)
                appendQuad(&mesh, quad)
            }
        }
        return mesh
    }

    /// Splits a quad along its shorter diagonal.
    private static func appendQuad(_ mesh: inout TriMesh, _ q: (UInt32, UInt32, UInt32, UInt32)) {
        let p = mesh.vertices
        let d02 = simd_distance_squared(p[Int(q.0)], p[Int(q.2)])
        let d13 = simd_distance_squared(p[Int(q.1)], p[Int(q.3)])
        if d02 <= d13 {
            mesh.triangles.append(SIMD3(q.0, q.1, q.2))
            mesh.triangles.append(SIMD3(q.0, q.2, q.3))
        } else {
            mesh.triangles.append(SIMD3(q.0, q.1, q.3))
            mesh.triangles.append(SIMD3(q.1, q.2, q.3))
        }
    }
}
