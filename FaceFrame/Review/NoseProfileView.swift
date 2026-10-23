import SwiftUI

/// Top-view cross-sections of the nose at 5/10/15 mm below the crest, with the frame back plane.
/// The pads sit where each section crosses the back plane.
struct NoseProfileView: View {
    let result: FitResult
    private let levels: [(Float, Color)] = [(0.005, .orange), (0.010, .pink), (0.015, .purple)]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Canvas { ctx, size in
                let scale = size.width / 0.04 // 40 mm across
                let cx = size.width / 2
                let top: CGFloat = 12
                func point(x: Float, depth: Float) -> CGPoint {
                    // Viewed from above, nose pointing up: the subject's left is on the left.
                    CGPoint(x: cx + CGFloat(x) * scale, y: top + CGFloat(depth) * scale)
                }
                // millimetre grid
                var grid = Path()
                for mm in stride(from: -20, through: 20, by: 5) {
                    let x = cx + CGFloat(mm) / 1000 * scale
                    grid.move(to: CGPoint(x: x, y: 0)); grid.addLine(to: CGPoint(x: x, y: size.height))
                }
                for mm in stride(from: 0, through: 20, by: 5) {
                    let y = top + CGFloat(mm) / 1000 * scale
                    grid.move(to: CGPoint(x: 0, y: y)); grid.addLine(to: CGPoint(x: size.width, y: y))
                }
                ctx.stroke(grid, with: .color(.secondary.opacity(0.15)), lineWidth: 0.5)

                for (level, color) in levels {
                    guard let s = result.slice(at: level) else { continue }
                    var path = Path()
                    let left = s.halfWidths.compactMap { e in e.left.map { (e.depth, $0) } }.reversed()
                    let right = s.halfWidths.compactMap { e in e.right.map { (e.depth, $0) } }
                    var first = true
                    for (d, w) in left {
                        let p = point(x: -w, depth: d)
                        if first { path.move(to: p); first = false } else { path.addLine(to: p) }
                    }
                    let tip = point(x: 0, depth: 0)
                    if first { path.move(to: tip); first = false } else { path.addLine(to: tip) }
                    for (d, w) in right { path.addLine(to: point(x: w, depth: d)) }
                    ctx.stroke(path, with: .color(color), lineWidth: 2)

                    // back plane crossing for this level
                    let bp = s.ridge.z - result.backPlaneZ
                    if bp > 0, let w = s.backPlaneWidth, w > 0 {
                        for x in [w / 2, -w / 2] {
                            let p = point(x: x, depth: bp)
                            ctx.fill(Path(ellipseIn: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)), with: .color(color))
                        }
                    }
                }
                ctx.draw(Text("L").font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: 10, y: 8))
                ctx.draw(Text("R").font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: size.width - 10, y: 8))
            }
            .frame(height: 170)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))

            HStack(spacing: 14) {
                ForEach(levels, id: \.0) { level, color in
                    Label("\(Int(level * 1000)) mm", systemImage: "circle.fill").foregroundStyle(color)
                }
            }
            .font(.caption)
            Text("Top view, 5 mm grid. Dots: where each section meets the frame back plane (pad seat).")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
}
