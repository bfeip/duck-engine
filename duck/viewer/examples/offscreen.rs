//! Headless render-to-texture example.
//!
//! Builds a small scene, renders it through an [`OffscreenViewer`] with its own
//! headless GPU context (no window/surface), reads the result back, and writes
//! it to a PNG. Demonstrates the offscreen path used for thumbnails and for
//! embedding the 3D view inside a UI panel.

use duck_engine_viewer::common::{RgbaColor, Transform, Vector3};
use duck_engine_viewer::scene::{
    FaceMaterial, Instance, Mesh, NodeFlags, PositionedCamera, PrimitiveType,
};
use duck_engine_viewer::OffscreenViewer;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut viewer = pollster::block_on(OffscreenViewer::headless(WIDTH, HEIGHT))?;

    // Build a simple scene: one sphere with a PBR material plus default lights.
    {
        let scene_arc = viewer.scene();
        let mut scene = scene_arc.lock().unwrap();

        let camera_node = scene
            .add_node(None, Some("Camera".to_string()), Default::default(), NodeFlags::NONE)
            .unwrap();
        scene.set_default_light_nodes(camera_node);

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
    viewer.set_camera(camera.clone());

    // Render into the owned offscreen texture (proves the GPU render path), then
    // read back a still image and save it.
    viewer.render()?;
    let image = viewer.render_to_image(&camera)?;

    let out = "offscreen.png";
    image.save(out)?;
    println!("Wrote {} ({}x{})", out, image.width(), image.height());

    Ok(())
}
