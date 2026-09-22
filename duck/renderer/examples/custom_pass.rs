//! Editing a stock workflow instead of replacing it.
//!
//! A [`SceneWorkflow`] is a named, ordered, id-addressable pass list, so a
//! caller starts from [`workflow::shaded`] and adjusts it. This example does
//! all four kinds of edit:
//!
//! 1. **Insert a user-written pass** — `GhostPass` goes in right after the face
//!    pass, so occluded geometry shows through whatever hides it.
//! 2. **Reuse a built-in pass from another workflow** — `SilhouetteEdgesPass`
//!    normally belongs to [`workflow::hidden_line`]; here it is dropped into
//!    the shaded sequence.
//! 3. **Retune a pass in place** — `pass_mut` reaches the silhouette pass and
//!    changes its color and threshold, with no pipeline rebuilt.
//! 4. **Remove a pass** — the sub-geometry highlight pass is dropped.
//!
//! Note what the example never does: allocate or resize a texture. `GhostPass`
//! and the silhouette pass both read the shared depth buffer, which they get by
//! declaring it from `Pass::target_features`; the host allocates it and keeps
//! it in step with the window.

use duck_engine_renderer::{
    FrameTargets, Gpu, GpuOptions, Pass, PassBuilder, PassId, RenderContext, Renderer, SceneFrame,
    SceneFrames, SceneResources, TargetFeatures, abi, pass, workflow,
};
use duck_engine_renderer::scene::{Light, PositionedCamera, PositionedLight, Projection, SceneData};
use duck_engine_renderer::scene::resource::{FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType};
use duck_engine_renderer::scene::common::{RgbaColor, Transform};

use duck_engine_common::{Deg, Point3, Quaternion, Rotation3, Vector3};
use duck_engine_scene::Scene;

const GHOST_WESL: &str = include_str!("ghost.wesl");

/// Redraws scene geometry where it is *occluded*, as a translucent rim-lit
/// ghost — an x-ray effect.
///
/// A user-written pass is built from the same [`PassBuilder`] the built-in
/// passes take, and declares the attachments it needs the same way, so it drops
/// into a stock workflow at any point.
struct GhostPass {
    pipeline: wgpu::RenderPipeline,
}

impl GhostPass {
    fn new(builder: &PassBuilder<'_>) -> Self {
        let shader = builder
            .compile_wesl(GHOST_WESL)
            .expect("failed to compile ghost shader");
        let pipeline = builder
            .pipeline()
            .shader(&shader, "vs_main", "fs_main")
            .label("Ghost")
            .blend(wgpu::BlendState::ALPHA_BLENDING)
            // Draw only what the face pass has already covered up.
            .depth_compare(wgpu::CompareFunction::Greater)
            .depth_write(false)
            .build();
        Self { pipeline }
    }
}

impl Pass<SceneFrames> for GhostPass {
    /// Reads the depth the face pass wrote, so it must run after one that
    /// declares the shared depth buffer.
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::depth()
    }

    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        !frame.draw.all_batches().is_empty()
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let (color_view, resolve_target) = targets.color_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Ghost Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(abi::GROUP_CAMERA, frame.bindings.camera, &[]);
        render_pass.set_bind_group(abi::GROUP_LIGHTS, frame.bindings.lights, &[]);

        for batch in frame.draw.all_batches() {
            if batch.primitive_type == PrimitiveType::TriangleList {
                frame.draw_batch(gpu, &mut render_pass, batch);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let width = 800u32;
    let height = 600u32;

    let (gpu, caps) = pollster::block_on(Gpu::headless(GpuOptions::default()))?;
    let mut ctx = RenderContext::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb, 1, caps);
    let mut shared = SceneResources::new(&ctx);

    // A sphere partly behind a cube, so there is something to see through.
    let mut scene = SceneData::new();
    let sphere = scene.add_mesh(Mesh::sphere(1.0, 48, 24, PrimitiveType::TriangleList));
    let cube = scene.add_mesh(Mesh::cube(1.6, PrimitiveType::TriangleList));
    let blue = scene.add_face_material(
        FaceMaterial::new().with_base_color_factor(RgbaColor { r: 0.2, g: 0.4, b: 0.9, a: 1.0 }),
    );
    let grey = scene.add_face_material(
        FaceMaterial::new().with_base_color_factor(RgbaColor { r: 0.7, g: 0.7, b: 0.7, a: 1.0 }),
    );

    scene.add_instance_node(
        None,
        Instance::new(sphere).with_face_material(blue),
        Some("sphere".to_string()),
        Transform::from_position(Point3::new(0.0, 0.0, -1.2)),
        NodeFlags::NONE,
    )?;
    scene.add_instance_node(
        None,
        Instance::new(cube).with_face_material(grey),
        Some("cube".to_string()),
        Transform::from_position(Point3::new(0.6, -0.2, 0.8)),
        NodeFlags::NONE,
    )?;

    // A key light from over the camera's shoulder plus a dim fill, so every
    // face of the cube reads and the silhouette edges are the only black.
    let lights = [
        PositionedLight::world(
            Light::directional(RgbaColor { r: 1.0, g: 0.97, b: 0.9, a: 1.0 }, 2.5),
            Transform::from_rotation(Quaternion::from_angle_x(Deg(-35.0))),
        ),
        PositionedLight::world(
            Light::directional(RgbaColor { r: 0.6, g: 0.7, b: 0.9, a: 1.0 }, 1.2),
            Transform::from_rotation(Quaternion::from_angle_y(Deg(140.0))),
        ),
    ];

    let camera = PositionedCamera {
        eye: Point3::new(2.6, 2.0, 4.2),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: width as f32 / height as f32,
        projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
    };

    // Start from the stock shaded sequence, then edit it.
    let mut workflow = workflow::shaded(&mut ctx.pass_builder((width, height)));

    {
        let builder = ctx.pass_builder((width, height));

        // 1. A user-written pass, placed by name relative to a built-in one.
        workflow.insert_after(
            pass::ids::FACES,
            PassId("ghost"),
            GhostPass::new(&builder),
        )?;
    }
    {
        let mut builder = ctx.pass_builder((width, height));

        // 2. A built-in pass borrowed from the hidden-line workflow. Every pass
        //    takes the same builder, so they compose freely.
        workflow.insert_after(
            pass::ids::LINES_AND_POINTS,
            pass::ids::SILHOUETTE,
            pass::SilhouetteEdgesPass::new(&mut builder),
        )?;
    }

    // 3. Retune a pass in place — no pipeline is rebuilt.
    if let Some(silhouette) =
        workflow.pass_mut::<pass::SilhouetteEdgesPass>(pass::ids::SILHOUETTE)
    {
        silhouette.set_edge_color(RgbaColor { r: 0.1, g: 0.1, b: 0.15, a: 1.0 });
        silhouette.set_threshold(0.08);
    }

    // 4. Drop a pass this render has no use for.
    workflow.remove(pass::ids::SUB_GEOM_HIGHLIGHT);

    let order: Vec<_> = workflow.ids().map(|id| id.0).collect();
    println!("'{}' workflow: {}", workflow.name(), order.join(" -> "));

    let mut renderer = Renderer::with_workflow(&mut ctx, width, height, workflow);
    renderer.set_background_color(RgbaColor { r: 0.09, g: 0.09, b: 0.11, a: 1.0 });

    let mut scene = Scene::new(scene);
    let image = renderer.render_scene_to_image(
        &mut ctx,
        &mut shared,
        &mut scene,
        &camera,
        &lights,
        None,
    )?;
    image.save("custom_pass.png")?;
    println!("Saved custom_pass.png ({width}×{height})");
    Ok(())
}
