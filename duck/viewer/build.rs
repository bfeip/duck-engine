fn main() {
    cfg_aliases::cfg_aliases! {
        // Emscripten is a wasm32 target with a real `std` and no wasm-bindgen
        // runtime; it reaches the GPU through wgpu's GLES backend. `web` is the
        // wasm-bindgen browser target, which reaches it through WebGPU/WebGL.
        emscripten: { all(target_arch = "wasm32", target_os = "emscripten") },
        web: { all(target_arch = "wasm32", not(emscripten)) },
    }

    if std::env::var("TARGET").is_ok_and(|t| t.contains("emscripten")) {
        // GLES3/WebGL2 only: the GLES backend needs ES 3.0, and emscripten
        // defaults to a WebGL1 context.
        for arg in [
            "-sMIN_WEBGL_VERSION=2",
            "-sMAX_WEBGL_VERSION=2",
            "-sFULL_ES3=1",
            "-sALLOW_MEMORY_GROWTH=1",
            "-sSTACK_SIZE=8MB",
            // wgpu-hal statically references the EGL 1.5 entry points
            // (`eglGetPlatformDisplay`, `eglCreatePlatformWindowSurface`), but
            // emscripten only implements EGL 1.4. Its emscripten path never
            // calls them: `wsi.kind` is always `Unknown` there, which selects
            // the 1.4 `eglCreateWindowSurface` arm. Upstream wgpu's own
            // `raw-gles` example passes this same flag for this reason.
            "-sERROR_ON_UNDEFINED_SYMBOLS=0",
        ] {
            println!("cargo::rustc-link-arg-examples={arg}");
        }
    }
}
