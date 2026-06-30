import MetalKit
import simd
import QuartzCore

// CPU-side mirrors of the Metal structs (must match field order/layout exactly).
struct Params {
    var windDir: SIMD4<Float>
    var dt: Float
    var time: Float
    var gravity: Float
    var damp: Float
    var baseBreeze: Float
    var curlStrength: Float
    var noiseFreq: Float
    var scrollSpeed: Float
    var windScale: Float
    var friction: Float
    var floorY: Float
    var windOn: Int32
    var count: Int32
}

struct SolveParams {
    var offset: Int32
    var count: Int32
    var dt: Float
    var _pad: Int32 = 0
}

struct Lighting {
    var cam: SIMD4<Float>
    var light: SIMD4<Float>
    var front: SIMD4<Float>
    var back: SIMD4<Float>
}

final class Renderer: NSObject, MTKViewDelegate {
    // tunables
    let gn = 320                 // 320*320 = 102,400 particles (lower to 256 if heavy)
    let width: Float = 3.0
    let subSteps = 4
    let iters = 4
    let dt: Float = 1.0 / 240.0  // subSteps*dt = 1/60 s per frame
    let gravity: Float = 9.8
    let damp: Float = 0.99
    let floorY: Float = -3.0

    private let device: MTLDevice
    private let queue: MTLCommandQueue

    // compute + render pipelines
    private var psPredict: MTLComputePipelineState!
    private var psClear: MTLComputePipelineState!
    private var psSolve: MTLComputePipelineState!
    private var psCollide: MTLComputePipelineState!
    private var psBuild: MTLComputePipelineState!
    private var renderPipe: MTLRenderPipelineState!
    private var floorPipe: MTLRenderPipelineState!
    private var depthState: MTLDepthStencilState!

    // buffers
    private var posBuf, prevBuf, lamBuf, conBuf, vtxBuf, idxBuf: MTLBuffer!
    private var floorVtx, floorIdx: MTLBuffer!
    private var fabricTex: MTLTexture!
    private var fabricSampler: MTLSamplerState!

    private var model: ClothModel!
    private var conCount = 0
    private var idxCount = 0
    private var m = 0

    // pristine copies for reset
    private var pos0: [SIMD4<Float>] = []
    private var prev0: [SIMD4<Float>] = []

    // camera / state
    private var yaw: Float = 0.5
    private var pitch: Float = 0.2
    private var dist: Float = 6.5
    private var aspect: Float = 1.0
    private var windOn: Int32 = 1
    private var windScale: Float = 1.0
    private let startTime = CACurrentMediaTime()

    init(view: MTKView, device: MTLDevice) {
        self.device = device
        self.queue = device.makeCommandQueue()!
        super.init()

        buildModel()
        buildBuffers()
        buildPipelines(view: view)
        buildFloor()
        fabricTex = makeFabricTexture(512)
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear; sd.magFilter = .linear; sd.mipFilter = .linear
        sd.sAddressMode = .repeat; sd.tAddressMode = .repeat
        fabricSampler = device.makeSamplerState(descriptor: sd)
    }

    private func buildFloor() {
        let s: Float = 9.0, fy = floorY
        let verts: [Float] = [
            -s, fy, -s,  0, 1, 0,  0, 0,
             s, fy, -s,  0, 1, 0,  1, 0,
            -s, fy,  s,  0, 1, 0,  0, 1,
             s, fy,  s,  0, 1, 0,  1, 1,
        ]
        let idx: [UInt16] = [0, 2, 1, 1, 2, 3]
        floorVtx = device.makeBuffer(bytes: verts, length: verts.count * 4, options: .storageModeShared)
        floorIdx = device.makeBuffer(bytes: idx, length: idx.count * 2, options: .storageModeShared)
    }

    // Procedural plain-weave fabric: alternating warp/weft threads, each raised
    // thread caught by a soft highlight, plus a little per-thread noise.
    private func makeFabricTexture(_ s: Int) -> MTLTexture {
        var px = [UInt8](repeating: 255, count: s * s * 4)
        let threads = 8
        func hash(_ x: Int, _ y: Int) -> Float {
            var h = x &* 374761393 &+ y &* 668265263
            h = (h ^ (h >> 13)) &* 1274126177
            return Float((h ^ (h >> 16)) & 0xFFFF) / 65535.0
        }
        func b(_ f: Float) -> UInt8 { UInt8(max(0, min(255, Int(f * 255)))) }
        for y in 0..<s {
            for x in 0..<s {
                let fx = Float(x) / Float(s) * Float(threads)
                let fy = Float(y) / Float(s) * Float(threads)
                let cx = Int(fx.rounded(.down)), cy = Int(fy.rounded(.down))
                let tx = fx - Float(cx), ty = fy - Float(cy)
                let over = ((cx + cy) & 1) == 0
                var shade = over ? sin(ty * .pi) : sin(tx * .pi)
                shade = 0.45 + 0.55 * shade
                let n = hash(cx, cy) * 0.08 - 0.04
                let o = (y * s + x) * 4
                px[o]   = b(shade * 0.96 + n)
                px[o+1] = b(shade * 0.93 + n)
                px[o+2] = b(shade * 0.90 + n)
                px[o+3] = 255
            }
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: s, height: s, mipmapped: true)
        desc.usage = [.shaderRead]
        let tex = device.makeTexture(descriptor: desc)!
        tex.replace(region: MTLRegionMake2D(0, 0, s, s), mipmapLevel: 0, withBytes: px, bytesPerRow: s * 4)
        if let cmd = queue.makeCommandBuffer(), let blit = cmd.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: tex)
            blit.endEncoding()
            cmd.commit()
        }
        return tex
    }

    private func buildModel() {
        model = ClothModel(gn: gn, width: width)
        m = model.m
        conCount = model.constraints.count
        idxCount = model.indices.count
        pos0 = model.pos
        prev0 = model.prev
    }

    private func buildBuffers() {
        let opt: MTLResourceOptions = .storageModeShared
        posBuf  = device.makeBuffer(bytes: model.pos,  length: m * MemoryLayout<SIMD4<Float>>.stride, options: opt)
        prevBuf = device.makeBuffer(bytes: model.prev, length: m * MemoryLayout<SIMD4<Float>>.stride, options: opt)
        lamBuf  = device.makeBuffer(length: conCount * MemoryLayout<Float>.stride, options: opt)
        conBuf  = device.makeBuffer(bytes: model.constraints, length: conCount * MemoryLayout<GPUConstraint>.stride, options: opt)
        vtxBuf  = device.makeBuffer(length: m * 8 * MemoryLayout<Float>.stride, options: opt)
        idxBuf  = device.makeBuffer(bytes: model.indices, length: idxCount * MemoryLayout<UInt32>.stride, options: opt)
    }

    private func buildPipelines(view: MTKView) {
        let lib: MTLLibrary
        do {
            lib = try device.makeLibrary(source: metalSource, options: nil)
        } catch {
            fatalError("Shader compile failed: \(error)")
        }

        func compute(_ name: String) -> MTLComputePipelineState {
            let fn = lib.makeFunction(name: name)!
            return try! device.makeComputePipelineState(function: fn)
        }
        psPredict = compute("predict")
        psClear   = compute("clearLambda")
        psSolve   = compute("solve")
        psCollide = compute("collide")
        psBuild   = compute("buildMesh")

        let rpd = MTLRenderPipelineDescriptor()
        rpd.vertexFunction = lib.makeFunction(name: "vmain")
        rpd.fragmentFunction = lib.makeFunction(name: "fmain")
        rpd.colorAttachments[0].pixelFormat = view.colorPixelFormat
        rpd.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        renderPipe = try! device.makeRenderPipelineState(descriptor: rpd)

        let fpd = MTLRenderPipelineDescriptor()
        fpd.vertexFunction = lib.makeFunction(name: "floorV")
        fpd.fragmentFunction = lib.makeFunction(name: "floorF")
        fpd.colorAttachments[0].pixelFormat = view.colorPixelFormat
        fpd.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        floorPipe = try! device.makeRenderPipelineState(descriptor: fpd)

        let dsd = MTLDepthStencilDescriptor()
        dsd.depthCompareFunction = .less
        dsd.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: dsd)
    }

    // ---- per-frame --------------------------------------------------------

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspect = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }

        let time = Float(CACurrentMediaTime() - startTime)

        var params = Params(
            windDir: SIMD4<Float>(0.9428, 0.0471, 0.3300, 0), // normalize(1,0.05,0.35)
            dt: dt, time: time, gravity: gravity, damp: damp,
            baseBreeze: 5.0, curlStrength: 7.0, noiseFreq: 1.1, scrollSpeed: 0.6,
            windScale: windScale, friction: 0.25, floorY: floorY, windOn: windOn, count: Int32(m))

        // ---- compute (one serial encoder orders all dispatches) ----
        let ce = cmd.makeComputeCommandEncoder()!
        var conCountI = Int32(conCount)
        var gnI = Int32(gn)

        for _ in 0..<subSteps {
            // predict
            ce.setComputePipelineState(psPredict)
            ce.setBuffer(posBuf, offset: 0, index: 0)
            ce.setBuffer(prevBuf, offset: 0, index: 1)
            ce.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            dispatch(ce, m)

            // clear lambda
            ce.setComputePipelineState(psClear)
            ce.setBuffer(lamBuf, offset: 0, index: 0)
            ce.setBytes(&conCountI, length: MemoryLayout<Int32>.stride, index: 1)
            dispatch(ce, conCount)

            // graph-colored solve
            ce.setComputePipelineState(psSolve)
            ce.setBuffer(posBuf, offset: 0, index: 0)
            ce.setBuffer(lamBuf, offset: 0, index: 1)
            ce.setBuffer(conBuf, offset: 0, index: 2)
            for _ in 0..<iters {
                for c in 0..<8 {
                    let cnt = model.colorCount[c]
                    if cnt == 0 { continue }
                    var sp = SolveParams(offset: Int32(model.colorOffset[c]), count: Int32(cnt), dt: dt)
                    ce.setBytes(&sp, length: MemoryLayout<SolveParams>.stride, index: 3)
                    dispatch(ce, cnt)
                }
            }

            // floor collision
            ce.setComputePipelineState(psCollide)
            ce.setBuffer(posBuf, offset: 0, index: 0)
            ce.setBuffer(prevBuf, offset: 0, index: 1)
            ce.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
            dispatch(ce, m)
        }

        // build render mesh (positions + normals) from the solved positions
        ce.setComputePipelineState(psBuild)
        ce.setBuffer(posBuf, offset: 0, index: 0)
        ce.setBuffer(vtxBuf, offset: 0, index: 1)
        ce.setBytes(&gnI, length: MemoryLayout<Int32>.stride, index: 2)
        dispatch(ce, m)
        ce.endEncoding()

        // ---- render ----
        let eye = SIMD3<Float>(
            dist * cos(pitch) * sin(yaw),
            dist * sin(pitch),
            dist * cos(pitch) * cos(yaw))
        let target = SIMD3<Float>(0, -0.6, 0)
        var mvp = perspective(0.85, aspect, 0.05, 60) * lookAt(eye, target, SIMD3<Float>(0, 1, 0))
        var light = Lighting(
            cam: SIMD4<Float>(eye.x, eye.y, eye.z, 1),
            light: SIMD4<Float>(0.4, 0.85, 0.5, 0),
            front: SIMD4<Float>(0.85, 0.12, 0.22, 1),
            back: SIMD4<Float>(0.20, 0.10, 0.35, 1))

        let re = cmd.makeRenderCommandEncoder(descriptor: rpd)!
        re.setDepthStencilState(depthState)

        // floor
        re.setRenderPipelineState(floorPipe)
        re.setVertexBuffer(floorVtx, offset: 0, index: 0)
        re.setVertexBytes(&mvp, length: MemoryLayout<simd_float4x4>.stride, index: 1)
        re.setFragmentBytes(&light, length: MemoryLayout<Lighting>.stride, index: 0)
        re.drawIndexedPrimitives(type: .triangle, indexCount: 6,
                                 indexType: .uint16, indexBuffer: floorIdx, indexBufferOffset: 0)

        // cloth (textured)
        re.setRenderPipelineState(renderPipe)
        re.setVertexBuffer(vtxBuf, offset: 0, index: 0)
        re.setVertexBytes(&mvp, length: MemoryLayout<simd_float4x4>.stride, index: 1)
        re.setFragmentBytes(&light, length: MemoryLayout<Lighting>.stride, index: 0)
        re.setFragmentTexture(fabricTex, index: 0)
        re.setFragmentSamplerState(fabricSampler, index: 0)
        re.drawIndexedPrimitives(type: .triangle, indexCount: idxCount,
                                 indexType: .uint32, indexBuffer: idxBuf, indexBufferOffset: 0)
        re.endEncoding()

        cmd.present(drawable)
        cmd.commit()
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ count: Int) {
        if count <= 0 { return }
        let tg = 256
        let groups = (count + tg - 1) / tg
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    }

    // ---- input ------------------------------------------------------------

    func orbit(dx: Float, dy: Float) {
        yaw -= dx * 0.006
        pitch += dy * 0.006
        pitch = max(-1.45, min(1.45, pitch))
    }

    func zoom(_ d: Float) {
        dist *= 1.0 - d * 0.02
        dist = max(1.5, min(20.0, dist))
    }

    func key(_ code: UInt16) {
        switch code {
        case 13: windOn ^= 1                                   // W: toggle wind
        case 15: reset()                                       // R: reset
        case 49: releaseAll()                                  // Space: unpin (let it blow away)
        case 126: windScale = min(windScale * 1.25, 20)        // Up
        case 125: windScale = max(windScale / 1.25, 0)         // Down
        default: break
        }
    }

    private func releaseAll() {
        let p = posBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: m)
        for k in 0..<m { p[k].w = 1.0 }
    }

    private func reset() {
        memcpy(posBuf.contents(), pos0, m * MemoryLayout<SIMD4<Float>>.stride)
        memcpy(prevBuf.contents(), prev0, m * MemoryLayout<SIMD4<Float>>.stride)
    }

    // ---- math (Metal NDC: z in [0,1], right-handed, looking down -z) ------

    private func perspective(_ fovy: Float, _ aspect: Float, _ near: Float, _ far: Float) -> simd_float4x4 {
        let ys = 1.0 / tan(fovy * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4<Float>(xs, 0, 0, 0),
            SIMD4<Float>(0, ys, 0, 0),
            SIMD4<Float>(0, 0, zs, -1),
            SIMD4<Float>(0, 0, zs * near, 0)))
    }

    private func lookAt(_ eye: SIMD3<Float>, _ center: SIMD3<Float>, _ up: SIMD3<Float>) -> simd_float4x4 {
        let f = simd_normalize(center - eye)
        let s = simd_normalize(simd_cross(f, up))
        let u = simd_cross(s, f)
        return simd_float4x4(columns: (
            SIMD4<Float>(s.x, u.x, -f.x, 0),
            SIMD4<Float>(s.y, u.y, -f.y, 0),
            SIMD4<Float>(s.z, u.z, -f.z, 0),
            SIMD4<Float>(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)))
    }
}
