//! Composites per-view color textures onto the viewer's final target.

use crate::{Viewer, view::PixelRect};

/// The format every view renders at, and the format of an [`SrgbStandIn`].
/// Views write linear color, which the format encodes on store and decodes
/// again when the compositor samples it.
pub(crate) const VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The composite fragment entry point for a final target of `format`.
///
/// View textures hold linear color. An sRGB target encodes on write; any other
/// format needs the encode applied in the shader.
fn composite_entry_point(target_format: wgpu::TextureFormat) -> &'static str {
    if target_format.is_srgb() { "fs_main" } else { "fs_encode" }
}

/// The sRGB format a texture of `format` can be viewed at: its own when it is
/// sRGB, else its sRGB variant when the texture can be reinterpreted.
pub(crate) fn srgb_view_format(
    format: wgpu::TextureFormat,
    can_reinterpret: bool,
) -> Option<wgpu::TextureFormat> {
    let srgb = format.add_srgb_suffix();
    (srgb.is_srgb() && (srgb == format || can_reinterpret)).then_some(srgb)
}

/// How the viewer composites into a destination texture: always through an
/// sRGB view, so the clear and the blending of views happen in linear space.
pub(crate) enum SrgbTarget {
    /// The destination itself, viewed at this sRGB format.
    View(wgpu::TextureFormat),
    /// The destination has no sRGB view.
    StandIn(SrgbStandIn),
}

impl SrgbTarget {
    /// The target for a destination of `format`. `can_reinterpret` says whether
    /// the destination was created to allow its sRGB variant as a view format.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        can_reinterpret: bool,
        width: u32,
        height: u32,
    ) -> Self {
        match srgb_view_format(format, can_reinterpret) {
            Some(srgb) => Self::View(srgb),
            None => Self::StandIn(SrgbStandIn::new(device, format, width, height)),
        }
    }

    /// The format the compositor writes, for [`Viewer::from_gpu`].
    pub fn composite_format(&self) -> wgpu::TextureFormat {
        match self {
            Self::View(format) => *format,
            Self::StandIn(_) => VIEW_FORMAT,
        }
    }

    /// Follow a resized destination. Ignores zero dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if let Self::StandIn(stand_in) = self {
            stand_in.resize(device, width, height);
        }
    }

    /// Render `viewer`'s views and composite them into `destination`. The
    /// encoder is not submitted.
    pub fn render(
        &self,
        viewer: &mut Viewer,
        destination: &wgpu::Texture,
        encoder: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<()> {
        match self {
            Self::View(format) => {
                let view = destination.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(*format),
                    ..Default::default()
                });
                viewer.render_to_target(&view, encoder)
            }
            Self::StandIn(stand_in) => {
                viewer.render_to_target(&stand_in.view, encoder)?;
                let view = destination.create_view(&wgpu::TextureViewDescriptor::default());
                stand_in.encode_into(encoder, &view);
                Ok(())
            }
        }
    }
}

/// A [`VIEW_FORMAT`] texture composited into on behalf of a destination with
/// no sRGB view, then re-encoded into it.
///
/// The re-encode samples the stand-in (decoding to linear) and encodes in the
/// shader, which is byte-identical to what an sRGB destination would hold.
pub(crate) struct SrgbStandIn {
    blit: Compositor,
    view: wgpu::TextureView,
    source: wgpu::BindGroup,
    size: (u32, u32),
}

impl SrgbStandIn {
    fn new(
        device: &wgpu::Device,
        destination_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let blit = Compositor::blit(device, destination_format);
        let (view, source) = Self::create_texture(device, &blit, width, height);
        Self { blit, view, source, size: (width.max(1), height.max(1)) }
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        (self.view, self.source) = Self::create_texture(device, &self.blit, width, height);
        self.size = (width.max(1), height.max(1));
    }

    /// The stand-in texture's render view, and the blit's bind group sampling it.
    fn create_texture(
        device: &wgpu::Device,
        blit: &Compositor,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sRGB Stand-In"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VIEW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let source = blit.bind(device, &view);
        (view, source)
    }

    /// Re-encode the composited frame into `destination`.
    fn encode_into(&self, encoder: &mut wgpu::CommandEncoder, destination: &wgpu::TextureView) {
        let (width, height) = self.size;
        self.blit.composite(
            encoder,
            destination,
            wgpu::Color::TRANSPARENT,
            std::iter::once((&self.source, PixelRect { x: 0, y: 0, width, height })),
        );
    }
}

/// Blits view textures onto the target in stack order with premultiplied-alpha
/// blending, so a view cleared to zero alpha overlays whatever is beneath it.
///
/// View textures hold premultiplied color: MSAA resolves average coverage into
/// color and alpha alike, and blending within a view keeps it so.
pub(crate) struct Compositor {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl Compositor {
    /// Stacks view textures with premultiplied-alpha blending.
    pub fn blend(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self::with_blend(device, format, Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING))
    }

    /// Copies one texture over the whole target, replacing it. Used to re-encode
    /// a finished frame into a destination with no sRGB view, where blending
    /// the source against the target would corrupt it.
    pub fn blit(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self::with_blend(device, format, None)
    }

    fn with_blend(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("View Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("View Composite Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("View Composite Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("View Composite Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(composite_entry_point(format)),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("View Composite Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self { pipeline, layout, sampler }
    }

    /// Create the bind group for one view texture.
    pub fn bind(&self, device: &wgpu::Device, src: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Composite Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Clear `target` and draw each layer's texture into its rect, in order
    /// (later layers on top).
    pub fn composite<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clear: wgpu::Color,
        layers: impl Iterator<Item = (&'a wgpu::BindGroup, PixelRect)>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("View Composite Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        for (bind_group, rect) in layers {
            pass.set_viewport(
                rect.x as f32,
                rect.y as f32,
                rect.width as f32,
                rect.height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::TextureFormat;

    #[test]
    fn srgb_targets_let_the_hardware_encode() {
        assert_eq!(composite_entry_point(TextureFormat::Rgba8UnormSrgb), "fs_main");
        assert_eq!(composite_entry_point(TextureFormat::Bgra8UnormSrgb), "fs_main");
    }

    #[test]
    fn non_srgb_targets_encode_in_the_shader() {
        assert_eq!(composite_entry_point(TextureFormat::Rgba8Unorm), "fs_encode");
        assert_eq!(composite_entry_point(TextureFormat::Bgra8Unorm), "fs_encode");
        assert_eq!(composite_entry_point(TextureFormat::Rgba16Float), "fs_encode");
    }

    #[test]
    fn srgb_formats_view_as_themselves() {
        for can_reinterpret in [false, true] {
            assert_eq!(
                srgb_view_format(TextureFormat::Bgra8UnormSrgb, can_reinterpret),
                Some(TextureFormat::Bgra8UnormSrgb)
            );
        }
    }

    #[test]
    fn reinterpretable_formats_view_as_their_srgb_variant() {
        assert_eq!(
            srgb_view_format(TextureFormat::Rgba8Unorm, true),
            Some(TextureFormat::Rgba8UnormSrgb)
        );
        assert_eq!(
            srgb_view_format(TextureFormat::Bgra8Unorm, true),
            Some(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn formats_needing_a_stand_in_have_no_srgb_view() {
        assert_eq!(srgb_view_format(TextureFormat::Rgba8Unorm, false), None);
        assert_eq!(srgb_view_format(TextureFormat::Rgba16Float, true), None);
    }
}
