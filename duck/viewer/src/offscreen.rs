use std::ops::{Deref, DerefMut};

use crate::{
    Viewer,
    compositor::Compositor,
    renderer::{Gpu, GpuOptions},
    view::PixelRect,
};

/// A [`Viewer`] that renders into an owned offscreen texture instead of a
/// surface. The texture is sampleable (`TEXTURE_BINDING`), so consumers can
/// display it inside a UI panel (e.g. `egui::Image` via
/// `egui_wgpu::Renderer::register_native_texture`) or read it back to an image.
pub struct OffscreenViewer {
    color_format: wgpu::TextureFormat,
    color_texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    sample_view: wgpu::TextureView,
    /// Set where the color target cannot carry a second, non-sRGB view.
    encode: Option<GammaEncode>,
    core: Viewer,
}

/// The gamma-encoded copy of the frame that samplers read, for devices without
/// `VIEW_FORMATS`.
///
/// The frame itself stays sRGB so views still composite in linear space; this
/// pass samples it (decoding to linear) and writes the encode into a non-sRGB
/// texture, which is byte-identical to what the sRGB target holds.
struct GammaEncode {
    blit: Compositor,
    texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    source: wgpu::BindGroup,
}

impl GammaEncode {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        source_view: &wgpu::TextureView,
    ) -> Self {
        let format = format.remove_srgb_suffix();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Sample Target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let target_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // A compositor over a non-sRGB format encodes in the shader, which is
        // exactly the conversion needed here.
        let blit = Compositor::blit(device, format);
        let source = blit.bind(device, source_view);
        Self { blit, texture, target_view, source }
    }

    /// Re-encode the finished frame into the sampleable copy.
    fn run(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        self.blit.composite(
            encoder,
            &self.target_view,
            wgpu::Color::TRANSPARENT,
            std::iter::once((&self.source, PixelRect { x: 0, y: 0, width, height })),
        );
    }
}

impl Deref for OffscreenViewer {
    type Target = Viewer;
    fn deref(&self) -> &Viewer {
        &self.core
    }
}

impl DerefMut for OffscreenViewer {
    fn deref_mut(&mut self) -> &mut Viewer {
        &mut self.core
    }
}

impl OffscreenViewer {
    /// Build an offscreen viewer around an existing GPU context.
    ///
    /// Use this to share the device/queue with the rest of an application (e.g.
    /// the device that owns the window surface and the egui renderer) so the
    /// rendered texture can be sampled by that same device. Views render at
    /// `sample_count` internally; the owned color texture is single-sampled.
    pub fn from_gpu(
        gpu: Gpu,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        let (color_texture, render_view, sample_view, encode) = Self::create_color(
            &gpu.device,
            color_format,
            width,
            height,
            gpu.capabilities().has_view_formats,
        );
        let core = Viewer::from_gpu(gpu, color_format, width, height, sample_count);
        Self {
            color_format,
            color_texture,
            render_view,
            sample_view,
            encode,
            core,
        }
    }

    /// Create an offscreen viewer with its own headless GPU context.
    ///
    /// Convenience for thumbnails / server-side rendering where no surface or
    /// external device is involved. Uses `Rgba8UnormSrgb` and no MSAA.
    pub async fn headless(width: u32, height: u32, options: GpuOptions) -> anyhow::Result<Self> {
        let gpu = Gpu::headless(options).await?;
        Ok(Self::from_gpu(gpu, wgpu::TextureFormat::Rgba8UnormSrgb, width, height, 1))
    }

    /// The color target, its render view, the view samplers read it through,
    /// and — where the device has no `VIEW_FORMATS` — the encode pass that
    /// produces that view's texture.
    fn create_color(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        has_view_formats: bool,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView, Option<GammaEncode>) {
        // The sample view uses the non-sRGB variant so samplers read the raw
        // (gamma-encoded) bytes; for a non-sRGB `format` this is the same format.
        let sample_format = format.remove_srgb_suffix();
        let reinterpretable = has_view_formats || sample_format == format;
        let extra_view_format = [sample_format];
        let view_formats: &[wgpu::TextureFormat] =
            if reinterpretable && sample_format != format { &extra_view_format } else { &[] };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Color Target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats,
        });
        let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        if reinterpretable {
            let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(sample_format),
                ..Default::default()
            });
            return (texture, render_view, sample_view, None);
        }

        let encode = GammaEncode::new(device, format, width, height, &render_view);
        let sample_view = encode.texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, render_view, sample_view, Some(encode))
    }

    /// Resize the offscreen target, recreating the color texture.
    pub fn resize(&mut self, width: u32, height: u32) {
        let has_view_formats = self.core.context().capabilities().has_view_formats;
        let (texture, render_view, sample_view, encode) = Self::create_color(
            self.core.context().device(),
            self.color_format,
            width,
            height,
            has_view_formats,
        );
        self.color_texture = texture;
        self.render_view = render_view;
        self.sample_view = sample_view;
        self.encode = encode;
        self.core.resize(width, height);
    }

    /// Render all views into the owned offscreen texture and submit. Does not
    /// present (there is no surface); read the result via [`texture_view`] or
    /// [`Viewer::render_view_to_image`].
    ///
    /// [`texture_view`]: Self::texture_view
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        let mut encoder = self.core.context().device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Offscreen Render Encoder") },
        );
        // Disjoint field borrows: `&self.render_view` alongside `&mut self.core`.
        self.core.render_to_target(&self.render_view, &mut encoder)?;
        if let Some(encode) = &self.encode {
            let (width, height) = self.core.size();
            encode.run(&mut encoder, width, height);
        }
        self.core.context().queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// The offscreen color texture being rendered into.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.color_texture
    }

    /// The texture behind [`texture_view`](Self::texture_view): the color
    /// target itself, or its gamma-encoded copy where the device cannot
    /// reinterpret formats.
    pub fn sample_texture(&self) -> &wgpu::Texture {
        self.encode.as_ref().map_or(&self.color_texture, |encode| &encode.texture)
    }

    /// A non-sRGB view of the offscreen color texture, for sampling (e.g. egui).
    /// See the type docs for why this is not the sRGB render view.
    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.sample_view
    }
}
