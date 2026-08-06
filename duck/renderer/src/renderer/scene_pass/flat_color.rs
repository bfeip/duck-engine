use crate::abi;
use crate::render_core::{FrameTargets, Gpu, GpuTexture};
use crate::scene::resource::PrimitiveType;
use crate::scene::common::RgbaColor;

use super::super::mesh::{instance_buffer_layout, vertex_buffer_layout};
use super::super::pass_context::{SceneFrame, SceneRenderPass};

/// Per-instance configuration for [`FlatColorPass`].
///
/// Encodes everything that distinguishes different flat-color pass variants
/// so all can be driven by one struct + one pipeline builder.
pub(crate) struct FlatColorPassDesc {
    pub label: &'static str,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_compare: wgpu::CompareFunction,
    pub depth_write: bool,
    pub depth_bias: wgpu::DepthBiasState,
    /// `Some` → `LoadOp::Clear` with this color. `None` → `LoadOp::Load`.
    pub clear_color: Option<wgpu::Color>,
    /// Only batches whose primitive type matches this value are drawn.
    pub primitive_filter: PrimitiveType,
    pub color: RgbaColor,
}

fn build_flat_color_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
    desc: &FlatColorPassDesc,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(desc.label),
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
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: match desc.primitive_filter {
                PrimitiveType::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                PrimitiveType::LineList => wgpu::PrimitiveTopology::LineList,
                PrimitiveType::PointList => wgpu::PrimitiveTopology::PointList,
            },
            cull_mode: desc.cull_mode,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: GpuTexture::DEPTH_FORMAT,
            depth_write_enabled: desc.depth_write,
            depth_compare: desc.depth_compare,
            stencil: Default::default(),
            bias: desc.depth_bias,
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

/// A flat-color geometry pass parameterized by [`FlatColorPassDesc`].
pub(crate) struct FlatColorPass {
    pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
    sample_count: u32,
    color_bind_group: wgpu::BindGroup,
    desc: FlatColorPassDesc,
}

impl FlatColorPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
        desc: FlatColorPassDesc,
    ) -> Self {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};

        let shader = shader_generator
            .generate_flat_color_shader(device)
            .expect("Failed to generate flat color shader");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(desc.label),
            bind_group_layouts: &[camera_bgl, lights_bgl, material_color_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = build_flat_color_pipeline(device, &pipeline_layout, &shader, surface_format, sample_count, &desc);

        let color_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(desc.label),
            contents: bytemuck::bytes_of(&desc.color),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(desc.label),
            layout: material_color_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });

        Self { pipeline, pipeline_layout, shader, surface_format, sample_count, color_bind_group, desc }
    }
}

impl SceneRenderPass for FlatColorPass {
    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        let sample_count = targets.sample_count();
        if self.sample_count != sample_count {
            self.sample_count = sample_count;
            self.pipeline = build_flat_color_pipeline(&gpu.device, &self.pipeline_layout, &self.shader, self.surface_format, sample_count, &self.desc);
        }
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let load_op = match self.desc.clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };
        let (color_view, resolve_target) = targets.color_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.desc.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations { load: load_op, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: if self.desc.clear_color.is_some() { wgpu::LoadOp::Clear(1.0) } else { wgpu::LoadOp::Load },
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
        render_pass.set_bind_group(abi::GROUP_MATERIAL, &self.color_bind_group, &[]);

        let filter = self.desc.primitive_filter;
        for batch in frame.draw.all_batches() {
            if batch.primitive_type != filter { continue; }
            frame.draw_batch(gpu, &mut render_pass, batch);
        }
    }
}
