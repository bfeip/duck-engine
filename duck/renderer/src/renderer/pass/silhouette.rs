use crate::render_core::{FrameTargets, Gpu, Pass, TargetFeatures};
use crate::scene::common::RgbaColor;

use super::super::PassBuilder;
use super::super::pass_context::{SceneFrame, SceneFrames};

/// GPU uniform for silhouette edge rendering.
/// Must match the layout in `silhouette_edges.wesl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SilhouetteUniform {
    pub edge_color: [f32; 4],
    pub threshold: f32,
    /// 1.0 for a perspective depth curve, 0.0 for a linear (orthographic) one.
    /// Selects how the shader normalizes the depth gradient.
    pub perspective_depth: f32,
    pub _pad: [f32; 2],
}

impl Default for SilhouetteUniform {
    fn default() -> Self {
        Self {
            edge_color: [0.0, 0.0, 0.0, 1.0],
            threshold: 0.08,
            perspective_depth: 1.0,
            _pad: [0.0; 2],
        }
    }
}

/// Silhouette edge detection pass.
///
/// A fullscreen screenspace compositor that reads the depth buffer and draws a
/// dark edge wherever neighboring pixel depths differ by more than a threshold.
/// This gives triangle-only geometry a visible outline at its silhouette even
/// when no explicit `LineList` primitives are present.
///
/// This pass owns no size-dependent state other than the bind group, which
/// references the shared depth texture view. The bind group is lazily created
/// (or recreated) in `execute` whenever it has been invalidated by a resize.
///
/// Not every backend can sample a depth texture. Where the adapter cannot, the
/// pipeline — whose fragment shader is a depth `textureLoad` — cannot be built
/// at all, so it is left `None` and the pass reports itself inactive.
pub struct SilhouetteEdgesPass {
    /// `None` where the backend cannot sample depth textures.
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    uniform: SilhouetteUniform,
}

impl SilhouetteEdgesPass {
    #[must_use]
    pub fn new(builder: &mut PassBuilder<'_>) -> Self {
        use wgpu::util::DeviceExt;

        let device = builder.device();
        let surface_format = builder.format();
        let sample_count = builder.sample_count();
        let depth_multisampled = sample_count > 1;
        let available = builder.capabilities().samples_depth_textures();
        if !available {
            log::warn!(
                "Silhouette edges unavailable on the {:?} backend: it cannot sample depth textures",
                builder.capabilities().backend,
            );
        }

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Silhouette Uniform Buffer"),
            contents: bytemuck::cast_slice(&[SilhouetteUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Silhouette Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: depth_multisampled,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = available.then(|| {
            let (shaders, device) = builder.shaders();
            let shader = shaders
                .generate_silhouette_shader(device, depth_multisampled)
                .expect("Failed to generate silhouette edges shader");
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Silhouette Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Silhouette Edges Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_silhouette"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group: None,
            uniform_buffer,
            uniform: SilhouetteUniform::default(),
        }
    }

    /// Set the color silhouette edges are drawn in.
    pub fn set_edge_color(&mut self, color: RgbaColor) {
        self.uniform.edge_color = [color.r, color.g, color.b, color.a];
    }

    /// Set how large a depth discontinuity counts as an edge. Larger values
    /// detect fewer edges.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.uniform.threshold = threshold;
    }

    fn make_bind_group(&self, device: &wgpu::Device, depth_view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Silhouette Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

impl Pass<SceneFrames> for SilhouetteEdgesPass {
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::none().with_sampled_depth()
    }

    fn is_active(&self, _frame: &SceneFrame<'_>) -> bool {
        self.pipeline.is_some()
    }

    fn resize(&mut self, _gpu: &Gpu, _targets: &FrameTargets) {
        self.bind_group = None;
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let Some(depth_view) = targets.sampled_depth_view() else { return };

        if self.bind_group.is_none() {
            self.bind_group = Some(self.make_bind_group(&gpu.device, depth_view));
        }
        // Absent where the backend cannot sample depth; `is_active` already
        // skips the pass there, but a hand-built workflow may not consult it.
        let Some(pipeline) = self.pipeline.as_ref() else { return };

        // The depth curve changes with the projection, so the normalization the
        // shader applies has to follow the camera.
        gpu.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[SilhouetteUniform {
                perspective_depth: if frame.projection.is_ortho() { 0.0 } else { 1.0 },
                ..self.uniform
            }]),
        );

        let (color_view, resolve_target) = targets.color_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Silhouette Edges Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        render_pass.draw(0..3, 0..1);
    }
}
