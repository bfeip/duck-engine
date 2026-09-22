use duck_engine_renderer::{
    FrameTargets, Gpu, GpuOptions, Pass, PassBuilder, PassId, RenderContext, Renderer, SceneFrame,
    SceneFrames, SceneResources, TargetFeatures, Workflow, abi,
};
use duck_engine_renderer::scene::{Light, PositionedCamera, PositionedLight, Projection, SceneData};
use duck_engine_renderer::scene::resource::{
    FaceMaterial, Instance, Mesh, PrimitiveType,
};
use duck_engine_renderer::scene::common::{RgbaColor, Transform};

use duck_engine_common::{Point3, Vector3};
use duck_engine_scene::resource::NodeFlags;
use duck_engine_scene::Scene;

const GOOCH_WESL: &str = include_str!("gooch.wesl");

struct GoochPass {
    pipeline: wgpu::RenderPipeline,
}

impl GoochPass {
    fn new(builder: &PassBuilder<'_>) -> Self {
        let shader = builder.compile_wesl(GOOCH_WESL)
            .expect("failed to compile gooch shader");
        let pipeline = builder.pipeline()
            .shader(&shader, "vs_main", "fs_main")
            .label("Gooch")
            .build();
        GoochPass { pipeline }
    }
}

impl Pass<SceneFrames> for GoochPass {
    // The pass clears and depth-tests, so it needs the shared depth buffer.
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::depth()
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gooch Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(frame.background_color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
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
    let mut ctx =
        RenderContext::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb, 1, caps);
    let mut shared = SceneResources::new(&ctx);

    // A workflow is just a named pass list, so a wholly custom stack is one
    // pass in one workflow — no wrapper type needed.
    let gooch = Workflow::new("Gooch")
        .with(PassId("gooch"), GoochPass::new(&ctx.pass_builder((width, height))));
    let mut renderer = Renderer::with_workflow(&mut ctx, width, height, gooch);

    // Build scene: UV sphere with a plain unlit material.
    // The Gooch pass ignores material bind groups and drives color purely from
    // normal direction, so any valid material works here.
    let mut scene = SceneData::new();
    let mesh_id = scene.add_mesh(Mesh::sphere(1.0, 48, 24, PrimitiveType::TriangleList));
    let mat_id = scene.add_face_material(FaceMaterial::new());
    scene.add_instance_node(
        None,
        Instance::new(mesh_id).with_face_material(mat_id),
        Some("sphere".to_string()),
        Default::default(),
        NodeFlags::NONE
    )?;

    // A warm directional light; direction is the transform's -Z axis
    // (identity = toward viewer).
    let lights = [PositionedLight::world(
        Light::directional(RgbaColor { r: 1.0, g: 0.95, b: 0.8, a: 1.0 }, 1.0),
        Transform::IDENTITY,
    )];

    let camera = PositionedCamera {
        eye: Point3::new(0.0, 0.0, 3.5),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: width as f32 / height as f32,
        projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
    };

    let mut scene = Scene::new(scene);
    let image =
        renderer.render_scene_to_image(&mut ctx, &mut shared, &mut scene, &camera, &lights, None)?;
    image.save("gooch.png")?;
    println!("Saved gooch.png ({width}×{height})");

    Ok(())
}
