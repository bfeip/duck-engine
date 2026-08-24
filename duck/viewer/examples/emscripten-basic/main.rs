//! Minimal emscripten canvas example.
//!
//! Proves the `wasm32-unknown-emscripten` path end to end: wgpu's GLES backend
//! acquires a WebGL2 context for a canvas addressed by CSS selector, the viewer
//! renders a scene into it, and `emscripten_set_main_loop` drives the frames.
//!
//! Build and serve:
//!
//! ```sh
//! source ~/src/emsdk/emsdk_env.sh
//! cargo build -p duck-engine-viewer --example emscripten-basic \
//!     --target wasm32-unknown-emscripten
//! cp viewer/examples/emscripten-basic/index.html \
//!     target/wasm32-unknown-emscripten/debug/examples/
//! python3 -m http.server -d target/wasm32-unknown-emscripten/debug/examples
//! ```

use std::ffi::c_int;

use duck_engine_viewer::common::{RgbaColor, Transform, Vector3};
use duck_engine_viewer::scene::resource::{FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType};
use duck_engine_viewer::scene::{PositionedCamera, Scene};
use duck_engine_viewer::{SurfacedViewer, ViewLayout};

const CANVAS: &str = "#canvas";
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

unsafe extern "C" {
    /// Emscripten's frame driver. `fps <= 0` means `requestAnimationFrame`, and
    /// `simulate_infinite_loop != 0` means the call never returns.
    fn emscripten_set_main_loop(
        func: extern "C" fn(),
        fps: c_int,
        simulate_infinite_loop: c_int,
    );
}

/// The viewer outlives `main` because `emscripten_set_main_loop` returns while
/// the frame callback keeps running.
static mut VIEWER: Option<SurfacedViewer<'static>> = None;

extern "C" fn frame() {
    let viewer = unsafe {
        let ptr = &raw mut VIEWER;
        match (*ptr).as_mut() {
            Some(viewer) => viewer,
            None => return,
        }
    };

    if let Err(err) = viewer.render() {
        log::error!("render failed: {err}");
    }
}

fn main() {
    // Emscripten routes stdout/stderr to the browser console.
    env_logger::builder().filter_level(log::LevelFilter::Info).init();

    // Emscripten's EGL instance is static, so adapter and device requests
    // resolve without yielding to the event loop and `block_on` will not
    // deadlock the way it would under wasm-bindgen.
    let mut viewer =
        pollster::block_on(SurfacedViewer::from_canvas_selector(CANVAS, WIDTH, HEIGHT));

    let scene = Scene::default();
    {
        let mut scene = scene.lock();

        let mesh_id = scene.add_mesh(Mesh::sphere(0.5, 32, 16, PrimitiveType::TriangleList));
        let mat_id = scene.add_face_material(
            FaceMaterial::new()
                .with_base_color_factor(RgbaColor::BLUE)
                .with_metallic_factor(0.1)
                .with_roughness_factor(0.4),
        );
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                Some("Sphere".to_string()),
                Transform::default(),
                NodeFlags::NONE,
            )
            .unwrap();
    }

    let view = viewer.add_view("main", scene, ViewLayout::FULL);
    viewer.view_mut(view).unwrap().set_camera(PositionedCamera {
        eye: (1.5, 1.0, 2.0).into(),
        target: (0.0, 0.0, 0.0).into(),
        up: Vector3::unit_y(),
        aspect: WIDTH as f32 / HEIGHT as f32,
        fovy: 45.0,
        znear: 0.01,
        zfar: 100.0,
        ortho: false,
    });

    log::info!("viewer initialised, entering main loop");

    unsafe {
        let ptr = &raw mut VIEWER;
        *ptr = Some(viewer);
        emscripten_set_main_loop(frame, 0, 0);
    }
}
