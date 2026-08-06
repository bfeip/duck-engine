use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::abi;
use crate::render_core::{FrameTargets, Gpu};
use crate::scene::resource::PrimitiveType;
use crate::scene::common::RgbaColor;

use super::super::batching::SubGeomBatch;
use super::super::mesh::{instance_buffer_layout, vertex_buffer_layout};
use super::super::pass_context::{SceneFrame, SceneRenderPass};

/// A writable flat-color GPU resource: a `vec4<f32>` uniform buffer plus its
/// bind group against the shared color material layout (group 2 /
/// `material_color.wesl`).
///
/// Wraps the buffer+bind-group pair otherwise re-created inline wherever a flat
/// color is pushed to the GPU. The buffer is `COPY_DST` so [`write`](Self::write)
/// can update the color per frame.
pub(crate) struct ColorResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl ColorResources {
    pub fn new(
        device: &wgpu::Device,
        color_bgl: &wgpu::BindGroupLayout,
        color: RgbaColor,
        label: &str,
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&color),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: color_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        Self { buffer, bind_group }
    }

    pub fn write(&self, queue: &wgpu::Queue, color: RgbaColor) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&color));
    }
}

/// The two color resources for one selection tier: the translucent face overlay
/// and the solid edge/point recolor.
///
/// Each needs its own buffer because `queue.write_buffer` is applied before any
/// GPU commands run, so a shared buffer would leave every draw seeing only the
/// last color written — the same constraint behind `OutlinePass`'s dual buffers.
struct TierColors {
    face: ColorResources,
    solid: ColorResources,
}

impl TierColors {
    fn new(device: &wgpu::Device, color_bgl: &wgpu::BindGroupLayout, tier: &str) -> Self {
        let transparent = RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
        Self {
            face: ColorResources::new(device, color_bgl, transparent, &format!("SubGeom {tier} Face Color")),
            solid: ColorResources::new(device, color_bgl, transparent, &format!("SubGeom {tier} Solid Color")),
        }
    }

    /// Updates both colors from a highlight config: faces take `face_alpha`,
    /// edges/points are solid (alpha 1.0). RGB is shared.
    fn write(&self, queue: &wgpu::Queue, cfg: &crate::highlight_query::HighlightConfig) {
        let [r, g, b, _] = cfg.color;
        self.face.write(queue, RgbaColor { r, g, b, a: cfg.face_alpha });
        self.solid.write(queue, RgbaColor { r, g, b, a: 1.0 });
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
    topology: wgpu::PrimitiveTopology,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_flat_color"),
            buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_flat_color"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology, cull_mode: None, ..Default::default() },
        // No depth attachment: sub-geometry highlights always draw on top.
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
        cache: None,
    })
}

/// Draws highlighted sub-geometry directly in the highlight color, on top of the
/// scene and not depth-tested:
/// - **Faces** as a translucent overlay (highlight color at `face_alpha`).
/// - **Edges / points** re-drawn solid in the highlight color.
///
/// Primary and secondary selections use their respective tier colors. Deliberately
/// mirrors [`FlatColorPass`](super::flat_color::FlatColorPass): it reuses the
/// `flat_color.wesl` shader (camera at group 0, `material_color` at group 2; the
/// lights group is an unused filler so the color stays at group 2) and
/// [`MeshGpuResources::draw_subgeom`](crate::renderer::mesh::MeshGpuResources::draw_subgeom).
pub(crate) struct SubGeomHighlightPass {
    triangle_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
    primary: TierColors,
    secondary: TierColors,
}

impl SubGeomHighlightPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
    ) -> Self {
        let shader = shader_generator
            .generate_flat_color_shader(device)
            .expect("Failed to generate flat color shader");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SubGeom Highlight Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, lights_bgl, material_color_bgl],
            push_constant_ranges: &[],
        });

        let make = |topology, label| build_pipeline(device, &pipeline_layout, &shader, surface_format, sample_count, topology, label);
        let triangle_pipeline = make(wgpu::PrimitiveTopology::TriangleList, "SubGeom Highlight Triangle Pipeline");
        let line_pipeline = make(wgpu::PrimitiveTopology::LineList, "SubGeom Highlight Line Pipeline");
        let point_pipeline = make(wgpu::PrimitiveTopology::PointList, "SubGeom Highlight Point Pipeline");

        Self {
            triangle_pipeline,
            line_pipeline,
            point_pipeline,
            pipeline_layout,
            shader,
            surface_format,
            sample_count,
            primary: TierColors::new(device, material_color_bgl, "Primary"),
            secondary: TierColors::new(device, material_color_bgl, "Secondary"),
        }
    }

    /// Draws one tier's sub-geom batches. Camera (0) and lights (1) bind groups
    /// must already be set; this sets the pipeline and color (2) per batch.
    fn draw_tier(
        &self,
        gpu: &Gpu,
        render_pass: &mut wgpu::RenderPass<'_>,
        frame: &SceneFrame<'_>,
        batches: &[SubGeomBatch],
        colors: &TierColors,
    ) {
        for batch in batches {
            let (pipeline, color) = match batch.primitive_type {
                PrimitiveType::TriangleList => (&self.triangle_pipeline, &colors.face),
                PrimitiveType::LineList => (&self.line_pipeline, &colors.solid),
                PrimitiveType::PointList => (&self.point_pipeline, &colors.solid),
            };
            let Some(gpu_mesh) = frame.gpu_meshes.get(batch.mesh_id) else { continue };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(abi::GROUP_MATERIAL, &color.bind_group, &[]);
            gpu_mesh.draw_subgeom(
                &gpu.device,
                render_pass,
                batch.primitive_type,
                &batch.instance_transform,
                batch.first_index,
                batch.index_count,
            );
        }
    }
}

impl SceneRenderPass for SubGeomHighlightPass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        frame.draw.has_sub_geom_highlights()
    }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        let sample_count = targets.sample_count();
        if self.sample_count == sample_count {
            return;
        }
        self.sample_count = sample_count;
        let make = |topology, label| build_pipeline(&gpu.device, &self.pipeline_layout, &self.shader, self.surface_format, sample_count, topology, label);
        self.triangle_pipeline = make(wgpu::PrimitiveTopology::TriangleList, "SubGeom Highlight Triangle Pipeline");
        self.line_pipeline = make(wgpu::PrimitiveTopology::LineList, "SubGeom Highlight Line Pipeline");
        self.point_pipeline = make(wgpu::PrimitiveTopology::PointList, "SubGeom Highlight Point Pipeline");
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        // Resolve the per-tier colors for this frame before recording draws.
        if let Some(cfg) = frame.draw.highlight_config() {
            self.primary.write(&gpu.queue, cfg);
        }
        if let Some(cfg) = frame.draw.secondary_highlight_config() {
            self.secondary.write(&gpu.queue, cfg);
        }

        let (color_view, resolve_target) = targets.color_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SubGeom Highlight Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Camera (0) and lights (1) are shared across every pipeline (same layout),
        // so bind them once; only pipeline + color (2) vary per batch.
        render_pass.set_bind_group(abi::GROUP_CAMERA, frame.bindings.camera, &[]);
        render_pass.set_bind_group(abi::GROUP_LIGHTS, frame.bindings.lights, &[]);

        self.draw_tier(gpu, &mut render_pass, frame, frame.draw.highlight_sub_geom_batches(), &self.primary);
        self.draw_tier(gpu, &mut render_pass, frame, frame.draw.secondary_highlight_sub_geom_batches(), &self.secondary);
    }
}
