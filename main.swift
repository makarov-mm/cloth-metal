import AppKit
import MetalKit

// MTKView subclass that forwards mouse/scroll/keyboard to the renderer.
final class ClothView: MTKView {
    weak var input: Renderer?
    override var acceptsFirstResponder: Bool { true }
    override func mouseDragged(with e: NSEvent) { input?.orbit(dx: Float(e.deltaX), dy: Float(e.deltaY)) }
    override func scrollWheel(with e: NSEvent) { input?.zoom(Float(e.deltaY)) }
    override func keyDown(with e: NSEvent) { input?.key(e.keyCode) }
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device. This requires a Mac with a Metal-capable GPU.")
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)

let frame = NSRect(x: 0, y: 0, width: 1000, height: 1000)
let window = NSWindow(
    contentRect: frame,
    styleMask: [.titled, .closable, .resizable, .miniaturizable],
    backing: .buffered,
    defer: false)
window.title = "Cloth — Metal XPBD (compute)  ·  drag rotate · wheel zoom · W wind · Up/Down strength · Space release · R reset"

let view = ClothView(frame: frame, device: device)
view.colorPixelFormat = .bgra8Unorm
view.depthStencilPixelFormat = .depth32Float
view.clearColor = MTLClearColor(red: 0.02, green: 0.022, blue: 0.03, alpha: 1.0)
view.preferredFramesPerSecond = 60

let renderer = Renderer(view: view, device: device)
view.delegate = renderer
view.input = renderer

window.contentView = view
window.makeFirstResponder(view)
window.center()
window.makeKeyAndOrderFront(nil)

app.activate(ignoringOtherApps: true)
app.run()
