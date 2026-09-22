/// How many independent coverage channels a mask texture carries.
///
/// Determines the backing texture format for the mask, discoverable via
/// [`Self::format`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskChannels {
    One,
    Two,
    // Rgb8Unorm does not exist, thus no `Three`
    Four,
}

impl MaskChannels {
    pub const ALL: [Self; 3] = [Self::One, Self::Two, Self::Four];

    /// The texture format backing this channel count.
    #[must_use]
    pub const fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::One => wgpu::TextureFormat::R8Unorm,
            Self::Two => wgpu::TextureFormat::Rg8Unorm,
            Self::Four => wgpu::TextureFormat::Rgba8Unorm,
        }
    }
}

/// A GPU texture bundled with its view and sampler.
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl GpuTexture {
    /// Depth-stencil texture format used for depth and stencil buffers.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// The sampler an attachment carries unless it needs something more:
    /// clamped to the edge, linear, no mips.
    fn clamped_linear(label: &str) -> wgpu::SamplerDescriptor<'_> {
        wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }
    }

    /// A 2D render-attachment texture, its default view, and `sampler`.
    ///
    /// Every constructor below is this plus the sampler its kind needs; they
    /// differ in nothing else. Zero dimensions are clamped to 1, so a minimized
    /// window does not fail allocation.
    fn attachment_with(
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        sample_count: u32,
        sampled: bool,
        label: &str,
        sampler: wgpu::Sampler,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: if sampled {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            },
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, sampler }
    }

    /// Create a depth texture at the given pixel dimensions.
    ///
    /// `sampled` makes the texture bindable, for passes that read depth. Only
    /// backends where [`GpuCapabilities::samples_depth_textures`] holds can do
    /// anything with it.
    ///
    /// Carries a comparison sampler, which is what distinguishes it from
    /// [`attachment`](Self::attachment).
    #[must_use]
    pub fn depth(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        sampled: bool,
        label: &str,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_max_clamp: 100.0,
            ..Self::clamped_linear(label)
        });
        Self::attachment_with(
            device,
            (width, height),
            Self::DEPTH_FORMAT,
            sample_count,
            sampled,
            label,
            sampler,
        )
    }

    /// Create a render-attachment texture in the given format, with a clamped
    /// linear sampler.
    ///
    /// `sampled` adds `TEXTURE_BINDING`. Do not combine it with a sample count
    /// above 1: wgpu's GL backend can only multisample a renderbuffer, which is
    /// not bindable. A multisampled attachment that will be read is therefore a
    /// resolve source only — resolve it into a single-sampled texture and
    /// sample that instead.
    #[must_use]
    pub fn attachment(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        sampled: bool,
        label: &str,
    ) -> Self {
        let sampler = device.create_sampler(&Self::clamped_linear(label));
        Self::attachment_with(
            device,
            (width, height),
            format,
            sample_count,
            sampled,
            label,
            sampler,
        )
    }

    /// Create a write-only color attachment texture in the given format.
    ///
    /// Named for the role — a render target nothing samples, such as the
    /// multisampled attachment that resolves into a surface texture — so call
    /// sites do not carry a bare `false`.
    #[must_use]
    pub fn color_attachment(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        label: &str,
    ) -> Self {
        Self::attachment(device, width, height, format, sample_count, false, label)
    }
}
