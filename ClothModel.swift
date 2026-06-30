import simd

// One distance constraint. Layout matches `struct Constraint` in the Metal
// source exactly: two int indices, rest length, compliance — 16 bytes, align 4.
struct GPUConstraint {
    var a: Int32
    var b: Int32
    var rest: Float
    var compliance: Float
}

// Builds the particle grid and the graph-colored constraint set on the CPU.
//
// Constraints are partitioned into 8 conflict-free colors so that no two
// constraints in a color share a particle. That lets each color be a single
// race-free compute dispatch — no atomics. The coloring is analytic for a grid:
//   colors 0,1  horizontal structural, by i&1
//   colors 2,3  vertical   structural, by j&1
//   colors 4,5  '\' shear,             by i&1
//   colors 6,7  '/' shear,             by i&1
struct ClothModel {
    let gn: Int
    let m: Int
    var pos: [SIMD4<Float>]      // xyz + invMass (0 = pinned)
    var prev: [SIMD4<Float>]
    var constraints: [GPUConstraint]
    var colorOffset: [Int]       // start index into `constraints`, per color
    var colorCount: [Int]
    var indices: [UInt32]

    static let structCompliance: Float = 0.0
    static let shearCompliance: Float = 2e-5

    init(gn: Int, width: Float) {
        self.gn = gn
        self.m = gn * gn

        var pos = [SIMD4<Float>](repeating: .zero, count: m)
        var prev = [SIMD4<Float>](repeating: .zero, count: m)

        let top = width * 0.5
        func idx(_ i: Int, _ j: Int) -> Int { j * gn + i }

        for j in 0..<gn {
            for i in 0..<gn {
                let x = (Float(i) / Float(gn - 1) - 0.5) * width
                let y = top - Float(j) / Float(gn - 1) * width
                let invMass: Float = (j == 0) ? 0.0 : 1.0   // pin the top edge (banner)
                let p = SIMD4<Float>(x, y, 0, invMass)
                pos[idx(i, j)] = p
                prev[idx(i, j)] = SIMD4<Float>(x, y, 0, 0)
            }
        }

        // bucket constraints by color
        var buckets = [[GPUConstraint]](repeating: [], count: 8)
        func p3(_ i: Int, _ j: Int) -> SIMD3<Float> {
            let v = pos[idx(i, j)]; return SIMD3<Float>(v.x, v.y, v.z)
        }
        func add(_ color: Int, _ i0: Int, _ j0: Int, _ i1: Int, _ j1: Int, _ comp: Float) {
            let a = idx(i0, j0), b = idx(i1, j1)
            let rest = simd_length(p3(i0, j0) - p3(i1, j1))
            buckets[color].append(GPUConstraint(a: Int32(a), b: Int32(b), rest: rest, compliance: comp))
        }

        for j in 0..<gn {
            for i in 0..<gn {
                if i + 1 < gn { add(i & 1, i, j, i + 1, j, ClothModel.structCompliance) }            // H structural
                if j + 1 < gn { add(2 + (j & 1), i, j, i, j + 1, ClothModel.structCompliance) }      // V structural
                if i + 1 < gn && j + 1 < gn { add(4 + (i & 1), i, j, i + 1, j + 1, ClothModel.shearCompliance) } // '\'
                if i + 1 < gn && j + 1 < gn { add(6 + (i & 1), i + 1, j, i, j + 1, ClothModel.shearCompliance) } // '/'
            }
        }

        var flat: [GPUConstraint] = []
        var offsets = [Int](repeating: 0, count: 8)
        var counts = [Int](repeating: 0, count: 8)
        for c in 0..<8 {
            offsets[c] = flat.count
            counts[c] = buckets[c].count
            flat.append(contentsOf: buckets[c])
        }

        // static render index buffer (two triangles per quad)
        var idxArr = [UInt32]()
        idxArr.reserveCapacity((gn - 1) * (gn - 1) * 6)
        for j in 0..<(gn - 1) {
            for i in 0..<(gn - 1) {
                let a = UInt32(idx(i, j)), b = UInt32(idx(i + 1, j))
                let c = UInt32(idx(i, j + 1)), d = UInt32(idx(i + 1, j + 1))
                idxArr.append(a); idxArr.append(b); idxArr.append(c)
                idxArr.append(b); idxArr.append(d); idxArr.append(c)
            }
        }

        self.pos = pos
        self.prev = prev
        self.constraints = flat
        self.colorOffset = offsets
        self.colorCount = counts
        self.indices = idxArr
    }
}
