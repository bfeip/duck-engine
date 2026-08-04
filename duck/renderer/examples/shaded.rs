//! Minimal headless render with the built-in [`ShadedWorkflow`].
//!
//! The counterpart to `gooch.rs`: where that example installs a *custom*
//! workflow, this one uses the renderer's default shaded workflow — the simplest
//! end-to-end use of the standard pipeline (PBR lit geometry, depth, MSAA off).
//! Run with `cargo run --example shaded -p duck-engine-renderer`.

use duck_engine_renderer::Renderer;
use duck_engine_renderer::scene::{
    FaceMaterial, Instance, Light, Mesh, NodePayload, PositionedCamera, PrimitiveType, SceneData,
    common::RgbaColor,
};
use duck_engine_common::{Point3, Vector3};
use duck_engine_scene::{NodeFlags, Scene};

fn main() -> anyhow::Result<()> {
    let (width, height) = (800u32, 600u32);
    let mut renderer = pollster::block_on(Renderer::new_headless(width, height));

    // A single lit sphere with a warm red material.
    let mut scene = SceneData::new();
    let mesh_id = scene.add_mesh(Mesh::sphere(1.0, 48, 24, PrimitiveType::TriangleList));
    let mat_id = scene.add_face_material(
        FaceMaterial::new().with_base_color_factor(RgbaColor { r: 0.8, g: 0.3, b: 0.2, a: 1.0 }),
    );
    scene.add_instance_node(
        None,
        Instance::new(mesh_id).with_face_material(mat_id),
        Some("sphere".to_string()),
        Default::default(),
        NodeFlags::NONE,
    )?;

    // A white directional light (its direction is the node's -Z axis).
    let light_id = scene
        .add_node(None, Some("DirectionalLight".to_string()), Default::default(), NodeFlags::NONE)
        .unwrap();
    scene.set_node_payload(
        light_id,
        NodePayload::Light(Light::directional(RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }, 2.0)),
    );

    let camera = PositionedCamera {
        eye: Point3::new(0.0, 0.0, 3.5),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: width as f32 / height as f32,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
        ortho: false,
    };

    // No `set_workflow` call: the renderer starts with the built-in ShadedWorkflow.
    let mut scene = Scene::new(scene);
    let image = renderer.render_scene_to_image(&camera, &mut scene, None)?;
    image.save("shaded.png")?;
    println!("Saved shaded.png ({width}×{height})");
    Ok(())
}
