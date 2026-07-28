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

    /// Create a depth texture at the given pixel dimensions.
    #[must_use] 
    pub fn depth(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });
        Self { texture, view, sampler }
    }

    /// Create a mask texture at the given dimensions, with `channels`
    /// independent coverage channels.
    #[must_use]
    pub fn mask(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        channels: MaskChannels,
        sample_count: u32,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: channels.format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mask Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self { texture, view, sampler }
    }

    /// Create a color attachment texture in the given format.
    ///
    /// Used as the multisampled color render target that resolves to the
    /// swapchain surface texture.
    #[must_use] 
    pub fn color_attachment(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        Self { texture, view, sampler }
    }
}
