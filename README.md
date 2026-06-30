# Cloth — Metal

A real-time GPU cloth simulation in Swift + Metal. It is the Metal counterpart to the OpenGL/C# cloth project: the same graph-colored **XPBD** solver and **curl-noise wind**, but running entirely in Metal compute shaders on a high-resolution grid, with the mesh rendered straight from the GPU buffer.

`320 x 320 = 102,400` particles, ~407k distance constraints, solved on the GPU every frame. Positions never leave VRAM.

## What it does

- **XPBD on the GPU.** Verlet prediction, a compliance-based constraint solve (stiffness is a material property, not a function of iteration count), and the render-mesh build all run as Metal compute kernels.
- **Graph coloring.** Structural + shear constraints are split into 8 conflict-free colors — no two constraints in a color share a particle — so each color is a single race-free dispatch. No atomics. The coloring is analytic for a grid (structural by row/column parity, shear by parity).
- **Curl-noise wind.** The wind is the curl of a noise vector potential (divergence-free, so it reads as real swirling air), ported to Metal Shading Language, with a steady breeze and slow gusts. Dependency-free 3D value noise.
- **Direct GPU rendering.** A build kernel writes positions + recomputed normals into a buffer that doubles as the render vertex buffer; the draw call pulls from it. Two-sided lit shading.

This mirrors the OpenGL project's **GPU path**. Self-collision, tearing, and sphere draping live on that project's CPU path and are intentionally not part of this Metal version (they need atomic spatial hashing / dynamic topology).

## Requirements

- macOS 13+ with a Metal-capable GPU (Apple Silicon recommended).
- Swift toolchain (Xcode or the Swift command-line tools).

## Build and run

From the project directory:

```bash
swift run -c release
```

Or build a proper `.app` bundle (launches as a normal foreground GUI app — recommended):

```bash
chmod +x build.sh
./build.sh run      # build release, wrap into build/ClothMetal.app, and launch
./build.sh          # build only
./build.sh clean    # remove build artifacts
```

## Controls

| Input | Action |
| --- | --- |
| Drag | Orbit the camera |
| Scroll | Zoom |
| `W` | Toggle wind |
| `Up` / `Down` | Increase / decrease wind strength |
| `Space` | Release all pins (the sheet blows away) |
| `R` | Reset to the pinned banner |

## Project structure

```text
Package.swift       SwiftPM executable manifest (flat layout: path ".")
build.sh            Build release + wrap into a .app bundle
main.swift          App, window, MTKView subclass, input
Renderer.swift      Metal setup, buffers, pipelines, per-frame compute + draw, camera
ClothModel.swift    Grid + graph-colored constraint construction (CPU)
Shaders.swift       Metal source (compute kernels + render), compiled at runtime
```

## How it works

The grid is pinned along its top edge. Each frame, for each substep, a `predict` kernel integrates gravity and the curl-noise wind (Verlet), a `clearLambda` kernel resets the XPBD multipliers, and then the 8 constraint colors are dispatched in turn. A final `buildMesh` kernel writes positions and normals into the render buffer, and an indexed draw renders the two-sided lit sheet.

Unlike OpenGL, no explicit memory barriers are needed between passes: a serial Metal compute command encoder automatically orders dependent dispatches on tracked buffers, and Metal's hazard tracking handles the compute-write → vertex-read transition into the render pass.

The XPBD update per constraint is:

```
dLambda = (-C - alphaTilde * lambda) / (w_a + w_b + alphaTilde),   alphaTilde = compliance / dt^2
```

Compliance `0` is rigid (structural); a small compliance softens shear.

## Tuning

Top of `Renderer.swift`:

```swift
let gn = 320          // grid resolution; lower to 256 if it's heavy, raise to 384 for ~147k
let subSteps = 4
let iters = 4
let dt: Float = 1/240 // subSteps*dt = 1/60 s per frame
```

Wind (`baseBreeze`, `curlStrength`, `noiseFreq`, `scrollSpeed`, `windDir`) is set in `draw(in:)`. Compliance is in `ClothModel.swift`.

## Notes

- Shaders are compiled from source at startup via `device.makeLibrary(source:)`, mirroring the OpenGL project where the GLSL also lived as runtime-compiled strings.
- Buffers use `.storageModeShared` (unified memory on Apple Silicon makes this cheap), which also lets reset/release rewrite particle data directly from the CPU.
- Dispatches use `dispatchThreadgroups` with in-kernel bounds checks, so non-uniform threadgroup support is not required.

## License

MIT
