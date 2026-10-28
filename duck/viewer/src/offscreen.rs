use std::ops::{Deref, DerefMut};

use crate::{
    Viewer,
    compositor::SrgbTarget,
    renderer::{Gpu, GpuOptions},
};

/// The format of an [`OffscreenViewer`]'s texture. Plain bytes holding the
/// gamma-encoded frame, which is what samplers such as egui's expect.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A [`Viewer`] that renders into an owned offscreen texture instead of a
/// surface. The texture is sampleable (`TEXTURE_BINDING`), so consumers can
/// display it inside a UI panel (e.g. `egui::Image` via
/// `egui_wgpu::Renderer::register_native_texture`) or copy it out.
///
/// The texture is `Rgba8Unorm` and holds display-ready, gamma-encoded color:
/// sampling it yields the bytes as they would appear on screen.
pub struct OffscreenViewer {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    target: SrgbTarget,
    core: Viewer,
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
    /// `sample_count` internally; the owned texture is single-sampled.
    pub fn from_gpu(gpu: Gpu, width: u32, height: u32, sample_count: u32) -> Self {
        let can_reinterpret = gpu.capabilities().has_view_formats;
        let (texture, view) = Self::create_texture(&gpu.device, width, height, can_reinterpret);
        let target = SrgbTarget::new(&gpu.device, FORMAT, can_reinterpret, width, height);
        let core = Viewer::from_gpu(gpu, target.composite_format(), width, height, sample_count);
        Self { texture, view, target, core }
    }

    /// Create an offscreen viewer with its own headless GPU context.
    ///
    /// Convenience for thumbnails / server-side rendering where no surface or
    /// external device is involved. Uses no MSAA.
    pub async fn headless(width: u32, height: u32, options: GpuOptions) -> anyhow::Result<Self> {
        let gpu = Gpu::headless(options).await?;
        Ok(Self::from_gpu(gpu, width, height, 1))
    }

    /// The texture and its default view. Where the device can reinterpret
    /// formats, the texture also allows the sRGB view the compositor writes.
    fn create_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        can_reinterpret: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let srgb = [FORMAT.add_srgb_suffix()];
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
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: if can_reinterpret { &srgb } else { &[] },
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Resize the offscreen target, recreating the texture. Ignores zero
    /// dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let device = self.core.context().device();
        let can_reinterpret = self.core.context().capabilities().has_view_formats;
        (self.texture, self.view) = Self::create_texture(device, width, height, can_reinterpret);
        self.target.resize(device, width, height);
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
        self.target.render(&mut self.core, &self.texture, &mut encoder)?;
        self.core.context().queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// The offscreen texture the frame lands in.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// A view of [`texture`](Self::texture) for sampling (e.g. egui).
    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
