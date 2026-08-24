fn main() {
    cfg_aliases::cfg_aliases! {
        emscripten: { all(target_arch = "wasm32", target_os = "emscripten") },
        web: { all(target_arch = "wasm32", not(emscripten)) },
    }

    if std::env::var("TARGET").is_ok_and(|t| t.contains("emscripten")) {
        for arg in [
            // The GLES backend needs ES 3.0; emscripten defaults to WebGL1.
            "-sMIN_WEBGL_VERSION=2",
            "-sMAX_WEBGL_VERSION=2",
            "-sFULL_ES3=1",
            "-sALLOW_MEMORY_GROWTH=1",
            "-sSTACK_SIZE=8MB",
            // wgpu-hal statically references the EGL 1.5 entry points, which
            // emscripten (EGL 1.4) does not provide. Its emscripten path never
            // calls them; upstream wgpu's `raw-gles` example does the same.
            "-sERROR_ON_UNDEFINED_SYMBOLS=0",
        ] {
            println!("cargo::rustc-link-arg-bins={arg}");
        }
    }
}
