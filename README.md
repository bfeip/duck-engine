# Duck Engine

Duck Engine is a suite of crates that compose a real-time 3D graphics engine primarily
targeting CAD applications, written in Rust using [wgpu](https://wgpu.rs/), and
targeting both native platforms and the web (WebAssembly / WebGL). It is built
around a strict layering: a GPU-free scene model sits underneath the wgpu renderer,
so scene data can be created, loaded, and queried without a graphics device.

On top of the core it provides asset import (glTF, USD, Assimp, and CAD via
OpenCASCADE) and an interactive CAD modeler in early development. The workspace root is
`duck/`; run all `cargo` commands from there.

## Crates

The Cargo workspace lives in `duck/`. Crates depend on each other in a chain
(`common → scene → render-core/renderer → viewer`), and each re-exports its major
dependency under a short alias.

| Crate | Purpose |
|---|---|
| `common` | Math/geometry primitives: `Ray`, `Aabb`, `Plane`, `Transform`, `RgbaColor`, etc. |
| `scene` | GPU-free scene graph, camera, materials (Face/Line/Point), geometry queries, environment maps.|
| `render-core` | Agnostic GPU plumbing: device/queue, render targets, headless readback, pipeline & shader caches. Implementation independent|
| `renderer` | Renderer implimentation using `render-core` and `scene`: PBR/surface shading, lights, IBL, headless rendering. |
| `import-export` | Configurable import: glTF, USD, Assimp, and CAD (STEP/IGES); CAD export. |
| `viewer` | Windowing, input, operator/event system, selection. Native + WASM. |
| `modeler` | Interactive CAD modeler app (egui + viewer): boolean ops, extrude, snapping, tool manager. |
| `egui-demo` | Desktop demo application using egui. |

## Requirements

- A recent Rust toolchain (workspace uses edition 2024).
- For CAD support: initialize the opencascade submodule (`third-part/opencascade-rs`).
- If building with a C++ dependency (`assimp`, `OCCT`) a C++ compiler is required.
