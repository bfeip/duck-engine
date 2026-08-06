use std::cell::RefCell;
use std::rc::Rc;

use crate::abi;
use crate::render_core::{FrameTargets, Gpu, GpuTexture, MaskChannels};
use crate::scene::resource::PrimitiveType;

use super::super::batching::DrawBatch;
use super::super::mesh::{instance_buffer_layout, vertex_buffer_layout};
use super::super::pass_context::{SceneFrame, SceneRenderPass};

/// One channel per highlight tier: red for primary, green for secondary.
const MASK_CHANNELS: MaskChannels = MaskChannels::Two;

/// Widest outline the composite shader will search for, in pixels.
const MAX_OUTLINE_WIDTH: f32 = 4.0;

/// The shader searches one texel past the widest band it must draw.
fn max_search_radius() -> f64 {
    f64::from(MAX_OUTLINE_WIDTH.ceil() + 1.0)
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

/// Creates a pipeline that renders highlighted geometry into one channel of the
/// mask texture. `write_mask` selects the tier's channel.
fn build_mask_pipeline(
    device: &wgpu::Device,
    camera_bgl: &wgpu::BindGroupLayout,
    shader_generator: &mut crate::shaders::ShaderGenerator,
    sample_count: u32,
    write_mask: wgpu::ColorWrites,
    label: &str,
) -> wgpu::RenderPipeline {
    let shader = shader_generator
        .generate_outline_mask_shader(device)
        .expect("Failed to generate outline mask shader");
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Outline Mask Pipeline Layout"),
        bind_group_layouts: &[camera_bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_mask"),
            buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_mask"),
            targets: &[Some(wgpu::ColorTargetState {
                format: MASK_CHANNELS.format(),
                blend: None,
                write_mask,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
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

fn make_composite_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    mask_view: &wgpu::TextureView,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Outline Composite Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(mask_view),
            },
            wgpu::BindGroupEntry { binding: 2, resource: uniform_buffer.as_entire_binding() },
        ],
    })
}

/// The mask attachment pair.
struct MaskTextures {
    /// `Some` only when `sample_count > 1`.
    msaa: Option<GpuTexture>,
    resolved: GpuTexture,
}

impl MaskTextures {
    fn new(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> Self {
        Self {
            msaa: (sample_count > 1).then(|| {
                GpuTexture::mask(device, width, height, MASK_CHANNELS, sample_count, "Outline Mask (MSAA)")
            }),
            resolved: GpuTexture::mask(device, width, height, MASK_CHANNELS, 1, "Outline Mask (resolved)"),
        }
    }

    /// `(attachment, resolve_target)` for the mask render pass.
    fn attachment(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        match &self.msaa {
            Some(msaa) => (&msaa.view, Some(&self.resolved.view)),
            None => (&self.resolved.view, None),
        }
    }
}

/// GPU resources shared by [`OutlineMaskPass`] and [`OutlineCompositePass`].
pub(crate) struct OutlineResources {
    primary_mask_pipeline: wgpu::RenderPipeline,
    secondary_mask_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    masks: MaskTextures,
}

impl OutlineResources {
    /// True when at least one tier has triangle geometry to mask.
    fn has_maskable_geometry(frame: &SceneFrame<'_>) -> bool {
        frame
            .draw
            .highlighted_batches()
            .iter()
            .chain(frame.draw.secondary_highlighted_batches())
            .any(|b| b.primitive_type == PrimitiveType::TriangleList)
    }

    fn rebuild_masks(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        let (w, h) = targets.size();
        self.masks = MaskTextures::new(&gpu.device, w, h, targets.sample_count());
        self.composite_bind_group = make_composite_bind_group(
            &gpu.device,
            &self.composite_bgl,
            &self.masks.resolved.view,
            &self.uniform_buffer,
        );
    }
}

/// Renders highlighted triangle geometry into the two channels of the mask
/// texture (red = primary tier, green = secondary), depth-tested against the
/// scene depth buffer so occluded geometry is not outlined.
pub(crate) struct OutlineMaskPass(Rc<RefCell<OutlineResources>>);

/// Reads the mask and composites both tiers' outlines over the scene in a single
/// fullscreen pass. Scheduled after lines and points so nothing cuts the band.
pub(crate) struct OutlineCompositePass(Rc<RefCell<OutlineResources>>);

/// Builds the outline mask and composite passes over one shared set of resources.
pub(crate) fn outline_passes(
    device: &wgpu::Device,
    config: crate::render_core::TargetConfig,
    camera_bgl: &wgpu::BindGroupLayout,
    shader_generator: &mut crate::shaders::ShaderGenerator,
) -> (OutlineMaskPass, OutlineCompositePass) {
    use wgpu::util::DeviceExt;

    let (width, height) = config.size;
    let sample_count = config.sample_count;

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Outline Uniform Buffer"),
        contents: bytemuck::cast_slice(&[OutlineUniform {
            primary_color: [1.0, 0.6, 0.0, 1.0],
            secondary_color: [0.7, 0.35, 0.0, 1.0],
            params: [2.0, 2.0, width as f32, height as f32],
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let primary_mask_pipeline = build_mask_pipeline(
        device,
        camera_bgl,
        shader_generator,
        sample_count,
        wgpu::ColorWrites::RED,
        "Outline Mask Pipeline (primary)",
    );
    let secondary_mask_pipeline = build_mask_pipeline(
        device,
        camera_bgl,
        shader_generator,
        sample_count,
        wgpu::ColorWrites::GREEN,
        "Outline Mask Pipeline (secondary)",
    );

    let masks = MaskTextures::new(device, width, height, sample_count);

    let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let composite_bind_group =
        make_composite_bind_group(device, &composite_bgl, &masks.resolved.view, &uniform_buffer);

    let shader = shader_generator
        .generate_outline_screenspace_shader(device)
        .expect("Failed to generate outline screenspace shader");
    let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Outline Composite Pipeline Layout"),
        bind_group_layouts: &[&composite_bgl],
        push_constant_ranges: &[],
    });
    // Overrides are resolved per pipeline stage, and only the stage that
    // references one must supply it — today that is the fragment shader alone.
    // Supplying both keeps this correct if the vertex shader ever grows a use.
    let overrides = [("max_search_radius", max_search_radius())];
    let compilation_options = wgpu::PipelineCompilationOptions {
        constants: &overrides,
        ..Default::default()
    };
    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                format: config.format,
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
        // The mask is single-sampled, but the color target is still the scene's
        // MSAA attachment.
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let shared = Rc::new(RefCell::new(OutlineResources {
        primary_mask_pipeline,
        secondary_mask_pipeline,
        composite_pipeline,
        composite_bgl,
        composite_bind_group,
        uniform_buffer,
        masks,
    }));

    (OutlineMaskPass(Rc::clone(&shared)), OutlineCompositePass(shared))
}

/// Draws one tier's highlighted triangles through `pipeline`.
fn draw_tier(
    gpu: &Gpu,
    rp: &mut wgpu::RenderPass<'_>,
    frame: &SceneFrame<'_>,
    pipeline: &wgpu::RenderPipeline,
    batches: &[DrawBatch],
) {
    let mut bound = false;
    for batch in batches {
        if batch.primitive_type != PrimitiveType::TriangleList {
            continue;
        }
        let Some(gpu_mesh) = frame.gpu_meshes.get(batch.mesh_id) else {
            continue;
        };
        if !bound {
            rp.set_pipeline(pipeline);
            bound = true;
        }
        gpu_mesh.draw_instances(
            &gpu.device,
            rp,
            batch.primitive_type,
            &batch.instances,
            batch.index_count,
        );
    }
}

impl SceneRenderPass for OutlineMaskPass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        OutlineResources::has_maskable_geometry(frame)
    }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        self.0.borrow_mut().rebuild_masks(gpu, targets);
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let res = self.0.borrow();

        let (sw, sh) = targets.size();
        let width_of = |cfg: Option<&crate::highlight_query::HighlightConfig>| {
            cfg.map_or(0.0, |c| c.width_pixels.clamp(0.0, MAX_OUTLINE_WIDTH))
        };
        let color_of = |cfg: Option<&crate::highlight_query::HighlightConfig>| {
            cfg.map_or([0.0; 4], |c| c.color)
        };
        let primary = frame.draw.highlight_config();
        let secondary = frame.draw.secondary_highlight_config();
        gpu.queue.write_buffer(
            &res.uniform_buffer,
            0,
            bytemuck::cast_slice(&[OutlineUniform {
                primary_color: color_of(primary),
                secondary_color: color_of(secondary),
                params: [width_of(primary), width_of(secondary), sw as f32, sh as f32],
            }]),
        );

        let (attachment, resolve_target) = res.masks.attachment();
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

        if primary.is_some() {
            draw_tier(
                gpu,
                &mut rp,
                frame,
                &res.primary_mask_pipeline,
                frame.draw.highlighted_batches(),
            );
        }
        if secondary.is_some() {
            draw_tier(
                gpu,
                &mut rp,
                frame,
                &res.secondary_mask_pipeline,
                frame.draw.secondary_highlighted_batches(),
            );
        }
    }
}

impl SceneRenderPass for OutlineCompositePass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        OutlineResources::has_maskable_geometry(frame)
    }

    // Resources are rebuilt by OutlineMaskPass, which shares them.

    fn execute(
        &mut self,
        _gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _frame: &mut SceneFrame<'_>,
    ) {
        let res = self.0.borrow();
        let (color_view, resolve_target) = targets.color_views(view);

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
        rp.set_pipeline(&res.composite_pipeline);
        rp.set_bind_group(0, &res.composite_bind_group, &[]);
        rp.draw(0..3, 0..1);
    }
}
