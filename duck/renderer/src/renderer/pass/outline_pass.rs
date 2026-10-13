use crate::abi;
use crate::highlight_query::HighlightConfig;
use crate::render_core::{
    AuxKind, AuxTarget, FrameTargets, Gpu, GpuTexture, MaskChannels, Pass, TargetFeatures,
};
use crate::scene::resource::PrimitiveType;

use super::super::PassBuilder;
use super::super::batching::DrawBatch;
use super::super::mesh::{instance_buffer_layout, vertex_buffer_layout};
use super::super::pass_context::{SceneFrame, SceneFrames};

/// One channel per highlight tier: red for primary, green for secondary.
const MASK_CHANNELS: MaskChannels = MaskChannels::Two;

/// Widest outline the composite shader will search for, in pixels.
const MAX_OUTLINE_WIDTH: f32 = 4.0;

/// The shader searches one texel past the widest band it must draw.
fn max_search_radius() -> f64 {
    f64::from(MAX_OUTLINE_WIDTH.ceil() + 1.0)
}

/// The mask both outline passes share, declared identically by each: the mask
/// pass writes it, the composite pass samples it, and the host owns it.
fn mask_target() -> AuxTarget {
    AuxTarget::new(super::aux::OUTLINE_MASK, AuxKind::Mask(MASK_CHANNELS))
        .with_resolved()
        .with_sampled()
}

/// GPU uniform for screen-space highlight outline rendering.
/// Must match the layout in `outline_screenspace.wesl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OutlineUniform {
    pub primary_color: [f32; 4],
    pub secondary_color: [f32; 4],
    /// `(primary_width, secondary_width, screen_width, screen_height)`, in pixels.
    /// A width of zero disables that tier.
    pub params: [f32; 4],
}

/// Creates a pipeline that renders highlighted geometry of one topology into
/// one channel of the mask texture. `write_mask` selects the tier's channel.
fn build_mask_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    sample_count: u32,
    write_mask: wgpu::ColorWrites,
    topology: wgpu::PrimitiveTopology,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_mask"),
            buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_mask"),
            targets: &[Some(wgpu::ColorTargetState {
                format: MASK_CHANNELS.format(),
                blend: None,
                write_mask,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            // Deliberately unculled: double-sided materials draw their back faces
            // in the scene pass, and culling here would leave an open/sheet body
            // viewed from behind with an empty mask and no outline.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: GpuTexture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
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

/// One tier's mask pipelines, one per primitive kind, all writing the tier's
/// channel.
struct TierPipelines {
    triangles: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
    points: wgpu::RenderPipeline,
}

impl TierPipelines {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        sample_count: u32,
        write_mask: wgpu::ColorWrites,
        tier: &str,
    ) -> Self {
        let make = |topology, kind: &str| {
            let label = format!("Outline Mask {kind} Pipeline ({tier})");
            build_mask_pipeline(device, layout, shader, sample_count, write_mask, topology, &label)
        };
        Self {
            triangles: make(wgpu::PrimitiveTopology::TriangleList, "Triangle"),
            lines: make(wgpu::PrimitiveTopology::LineList, "Line"),
            points: make(wgpu::PrimitiveTopology::PointList, "Point"),
        }
    }

    fn get(&self, primitive_type: PrimitiveType) -> &wgpu::RenderPipeline {
        match primitive_type {
            PrimitiveType::TriangleList => &self.triangles,
            PrimitiveType::LineList => &self.lines,
            PrimitiveType::PointList => &self.points,
        }
    }
}

/// Renders highlighted geometry — faces, lines and points — into the two
/// channels of the [`aux::OUTLINE_MASK`](super::aux::OUTLINE_MASK) attachment
/// (red = primary tier, green = secondary), depth-tested against the scene
/// depth buffer so occluded geometry is not outlined. A node without faces,
/// such as a wire, is thereby outlined as a stroke along its lines.
///
/// Belongs between the face and line geometry passes: the mask must depth-test
/// against faces only, since lines write unbiased depth that would punch holes
/// in it.
pub struct OutlineMaskPass {
    primary: TierPipelines,
    secondary: TierPipelines,
}

impl OutlineMaskPass {
    #[must_use]
    pub fn new(builder: &mut PassBuilder<'_>) -> Self {
        let sample_count = builder.sample_count();
        let pipeline_layout = builder.device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Outline Mask Pipeline Layout"),
                bind_group_layouts: &[&builder.layouts().camera],
                push_constant_ranges: &[],
            },
        );
        let (shaders, device) = builder.shaders();
        let shader = shaders
            .generate_outline_mask_shader(device)
            .expect("Failed to generate outline mask shader");

        let tier = |write_mask, name| {
            TierPipelines::new(device, &pipeline_layout, &shader, sample_count, write_mask, name)
        };
        Self {
            primary: tier(wgpu::ColorWrites::RED, "primary"),
            secondary: tier(wgpu::ColorWrites::GREEN, "secondary"),
        }
    }
}

/// Draws one tier's highlighted batches, each through the pipeline for its
/// primitive kind.
fn draw_tier(
    gpu: &Gpu,
    rp: &mut wgpu::RenderPass<'_>,
    frame: &SceneFrame<'_>,
    pipelines: &TierPipelines,
    batches: &[DrawBatch],
) {
    for batch in batches {
        rp.set_pipeline(pipelines.get(batch.primitive_type));
        frame.draw_batch(gpu, rp, batch);
    }
}

impl Pass<SceneFrames> for OutlineMaskPass {
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::depth().with_aux(mask_target())
    }

    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        frame.draw.has_node_highlights()
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let Some(mask) = targets.aux(super::aux::OUTLINE_MASK) else {
            return;
        };
        let (attachment, resolve_target) = mask.attachment();

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Highlight Mask Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attachment,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    // Only the resolved texture is ever read.
                    store: if resolve_target.is_some() {
                        wgpu::StoreOp::Discard
                    } else {
                        wgpu::StoreOp::Store
                    },
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
        rp.set_bind_group(abi::GROUP_CAMERA, frame.bindings.camera, &[]);

        if frame.draw.highlight_config().is_some() {
            draw_tier(
                gpu,
                &mut rp,
                frame,
                &self.primary,
                frame.draw.highlighted_batches(),
            );
        }
        if frame.draw.secondary_highlight_config().is_some() {
            draw_tier(
                gpu,
                &mut rp,
                frame,
                &self.secondary,
                frame.draw.secondary_highlighted_batches(),
            );
        }
    }
}

/// Reads the [`aux::OUTLINE_MASK`](super::aux::OUTLINE_MASK) attachment and
/// composites both tiers' outlines over the scene in a single fullscreen pass.
///
/// Belongs after lines and points, so nothing cuts the outline band.
pub struct OutlineCompositePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// Rebuilt whenever the mask attachment is reallocated.
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
}

impl OutlineCompositePass {
    #[must_use]
    pub fn new(builder: &mut PassBuilder<'_>) -> Self {
        use wgpu::util::DeviceExt;

        let device = builder.device();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Outline Uniform Buffer"),
            contents: bytemuck::cast_slice(&[OutlineUniform {
                primary_color: [1.0, 0.6, 0.0, 1.0],
                secondary_color: [0.7, 0.35, 0.0, 1.0],
                params: [0.0; 4],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline Composite Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let format = builder.format();
        let sample_count = builder.sample_count();
        let composite_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Outline Composite Pipeline Layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
        let (shaders, device) = builder.shaders();
        let shader = shaders
            .generate_outline_screenspace_shader(device)
            .expect("Failed to generate outline screenspace shader");

        // Overrides are resolved per pipeline stage, and only the stage that
        // references one must supply it — today that is the fragment shader alone.
        // Supplying both keeps this correct if the vertex shader ever grows a use.
        let overrides = [("max_search_radius", max_search_radius())];
        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &overrides,
            ..Default::default()
        };
        let pipeline = builder.device().create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Outline Composite Pipeline"),
                layout: Some(&composite_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: compilation_options.clone(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_outline"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                // The mask is single-sampled, but the color target is still the
                // scene's MSAA attachment.
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            },
        );

        Self { pipeline, layout, bind_group: None, uniform_buffer }
    }

    /// Point the bind group at the current mask attachment, rebuilding it if
    /// the attachment has been reallocated since.
    fn bind_group<'a>(
        &'a mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
    ) -> Option<&'a wgpu::BindGroup> {
        if self.bind_group.is_none() {
            let mask = targets.aux(super::aux::OUTLINE_MASK)?;
            self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Outline Composite Bind Group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(mask.sampled_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        self.bind_group.as_ref()
    }
}

impl Pass<SceneFrames> for OutlineCompositePass {
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::none().with_aux(mask_target())
    }

    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        // The mask pass's condition too, so the composite never reads a stale mask.
        frame.draw.has_node_highlights()
    }

    fn resize(&mut self, _gpu: &Gpu, _targets: &FrameTargets) {
        // The mask attachment has been reallocated; rebuild against the new one
        // lazily, so a resize costs nothing for a frame that draws no outline.
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
        let (screen_width, screen_height) = targets.size();
        let width_of = |cfg: Option<&HighlightConfig>| {
            cfg.map_or(0.0, |c| c.width_pixels.clamp(0.0, MAX_OUTLINE_WIDTH))
        };
        let color_of = |cfg: Option<&HighlightConfig>| cfg.map_or([0.0; 4], |c| c.color);
        let primary = frame.draw.highlight_config();
        let secondary = frame.draw.secondary_highlight_config();
        gpu.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[OutlineUniform {
                primary_color: color_of(primary),
                secondary_color: color_of(secondary),
                params: [
                    width_of(primary),
                    width_of(secondary),
                    screen_width as f32,
                    screen_height as f32,
                ],
            }]),
        );

        let (color_view, resolve_target) = targets.color_views(view);
        // Build the bind group before taking any shared borrow of `self`.
        if self.bind_group(gpu, targets).is_none() {
            return;
        }
        let (Some(bind_group), pipeline) = (self.bind_group.as_ref(), &self.pipeline) else {
            return;
        };

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Outline Composite Pass"),
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
        rp.set_pipeline(pipeline);
        rp.set_bind_group(0, bind_group, &[]);
        rp.draw(0..3, 0..1);
    }
}
