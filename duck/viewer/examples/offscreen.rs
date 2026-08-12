//! Headless render-to-texture example.
//!
//! Builds a small scene, renders it through an [`OffscreenViewer`] with its own
//! headless GPU context (no window/surface), reads the result back, and writes
//! it to a PNG. Demonstrates the offscreen path used for thumbnails and for
//! embedding the 3D view inside a UI panel.

use duck_engine_viewer::common::{RgbaColor, Transform, Vector3};
use duck_engine_viewer::scene::{PositionedCamera, Scene};
use duck_engine_viewer::scene::resource::{FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType};
use duck_engine_viewer::{OffscreenViewer, ViewLayout};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut viewer = pollster::block_on(OffscreenViewer::headless(WIDTH, HEIGHT))?;

    // Build a simple scene: one sphere with a PBR material.
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

    // A full-target view over the scene; this also adds a camera and default
    // lights.
    let view = viewer.add_view("main", scene, ViewLayout::FULL);

    // Point the camera at the sphere.
    let camera = PositionedCamera {
        eye: (1.5, 1.0, 2.0).into(),
        target: (0.0, 0.0, 0.0).into(),
        up: Vector3::unit_y(),
        aspect: WIDTH as f32 / HEIGHT as f32,
        fovy: 45.0,
        znear: 0.01,
        zfar: 100.0,
        ortho: false,
    };
    let mut view = viewer.view_mut(view).unwrap();
    view.set_camera(camera.clone());

    // Read back a still image and save it.
    let image = view.render_to_image(&camera)?;

    let out = "offscreen.png";
    image.save(out)?;
    println!("Wrote {} ({}x{})", out, image.width(), image.height());

    // Render into the owned offscreen texture as well (proves the composited
    // GPU render path).
    viewer.render()?;

    Ok(())
}
