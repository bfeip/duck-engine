//! Minimal headless render with the default shaded workflow.
//!
//! The simplest end-to-end use of the standard pipeline: PBR lit geometry,
//! depth, MSAA off. `gooch.rs` installs a wholly custom workflow instead, and
//! `custom_pass.rs` edits this one.
//! Run with `cargo run --example shaded -p duck-engine-renderer`.

use duck_engine_renderer::{Gpu, GpuOptions, RenderContext, Renderer, SceneResources};
use duck_engine_renderer::scene::{Light, PositionedCamera, PositionedLight, Projection, SceneData};
use duck_engine_renderer::scene::resource::{
    FaceMaterial, Instance, Mesh, PrimitiveType,
};
use duck_engine_renderer::scene::common::{RgbaColor, Transform};
use duck_engine_common::{Point3, Vector3};
use duck_engine_scene::resource::NodeFlags;
use duck_engine_scene::Scene;

fn main() -> anyhow::Result<()> {
    let (width, height) = (800u32, 600u32);
    let (gpu, caps) = pollster::block_on(Gpu::headless(GpuOptions::default()))?;
    let mut ctx =
        RenderContext::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb, 1, caps);
    let mut shared = SceneResources::new(&ctx);
    let mut renderer = Renderer::new(&mut ctx, width, height);

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

    // A white directional light (its direction is the transform's -Z axis).
    let lights = [PositionedLight::world(
        Light::directional(RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }, 2.0),
        Transform::IDENTITY,
    )];

    let camera = PositionedCamera {
        eye: Point3::new(0.0, 0.0, 3.5),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: width as f32 / height as f32,
        projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
    };

    // No `set_workflow` call: `Renderer::new` installs `workflow::shaded`.
    let mut scene = Scene::new(scene);
    let image =
        renderer.render_scene_to_image(&mut ctx, &mut shared, &mut scene, &camera, &lights, None)?;
    image.save("shaded.png")?;
    println!("Saved shaded.png ({width}×{height})");
    Ok(())
}
