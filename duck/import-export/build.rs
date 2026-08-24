fn main() {
    cfg_aliases::cfg_aliases! {
        // Emscripten is a wasm32 target with a real `std` and no wasm-bindgen
        // runtime. `web` is the wasm-bindgen browser target.
        emscripten: { all(target_arch = "wasm32", target_os = "emscripten") },
        web: { all(target_arch = "wasm32", not(emscripten)) },
    }
}
