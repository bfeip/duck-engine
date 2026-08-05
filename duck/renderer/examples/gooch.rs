use duck_engine_renderer::{
    FrameTargets, Gpu, Renderer, RenderWorkflow, SceneFrame, SceneFrames, SceneRenderPass, abi,
};
use duck_engine_renderer::scene::{
    FaceMaterial, Instance, Light, Mesh, NodePayload, PositionedCamera, PrimitiveType, SceneData,
    common::RgbaColor,
};

use duck_engine_common::{Point3, Vector3};
use duck_engine_scene::{NodeFlags, Scene};

const GOOCH_WESL: &str = include_str!("gooch.wesl");

struct GoochPass {
    pipeline: wgpu::RenderPipeline,
}

impl GoochPass {
    fn new(renderer: &Renderer) -> Self {
        let shader = renderer.compile_user_wesl(GOOCH_WESL)
            .expect("failed to compile gooch shader");
        let pipeline = renderer.custom_pipeline_builder()
            .shader(&shader, "vs_main", "fs_main")
            .label("Gooch")
            .build();
        GoochPass { pipeline }
    }
}

impl SceneRenderPass for GoochPass {
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

struct GoochWorkflow {
    pass: GoochPass,
}

impl GoochWorkflow {
    fn new(renderer: &Renderer) -> Self {
        Self { pass: GoochPass::new(renderer) }
    }
}

impl RenderWorkflow<SceneFrames> for GoochWorkflow {
    fn name(&self) -> &'static str { "Gooch" }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        self.pass.execute(gpu, targets, encoder, view, frame);
    }
}

fn main() -> anyhow::Result<()> {
    let width = 800u32;
    let height = 600u32;

    let mut renderer = pollster::block_on(Renderer::new_headless(width, height));

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

    // A warm directional light; direction is the node's -Z axis (identity = toward viewer).
    let light_id = scene.add_node(
        None, Some("DirectionalLight".to_string()), Default::default(), NodeFlags::NONE
    ).unwrap().id();
    scene.set_node_payload(light_id, NodePayload::Light(Light::directional(
        RgbaColor { r: 1.0, g: 0.95, b: 0.8, a: 1.0 },
        1.0,
    )));

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

    renderer.set_workflow(Box::new(GoochWorkflow::new(&renderer)));

    let mut scene = Scene::new(scene);
    let image = renderer.render_scene_to_image(&camera, &mut scene, None)?;
    image.save("gooch.png")?;
    println!("Saved gooch.png ({width}×{height})");

    Ok(())
}
