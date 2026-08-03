//! Cubemap texture infrastructure for IBL.
//!
//! Cubemaps are used to store environment maps, irradiance maps, and pre-filtered
//! specular maps for image-based lighting.

/// GPU resources for a cubemap texture.
///
/// Cubemaps consist of 6 square faces representing the +X, -X, +Y, -Y, +Z, -Z directions.
/// The `view` is a cube view suitable for sampling in shaders.
#[derive(Debug)]
pub struct GpuCubemap {
    /// The underlying wgpu texture (2D array with 6 layers).
    pub texture: wgpu::Texture,
    /// Cube view for sampling the entire cubemap.
    pub view: wgpu::TextureView,
    /// Sampler configured for cubemap sampling.
    pub sampler: wgpu::Sampler,
}

impl GpuCubemap {
    /// Create a new cubemap texture.
    ///
    /// # Arguments
    /// * `device` - The wgpu device
    /// * `size` - Width and height of each face in pixels
    /// * `format` - Texture format (typically Rgba16Float for HDR)
    /// * `mip_levels` - Number of mip levels (1 for no mipmaps)
    /// * `usage` - Texture usage flags
    /// * `label` - Debug label for the texture
    pub fn new(
        device: &wgpu::Device,
        size: u32,
        format: wgpu::TextureFormat,
        mip_levels: u32,
        usage: wgpu::TextureUsages,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        // Create cube view for sampling
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{} Cube View", label)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });

        // Create sampler with linear filtering for smooth cubemap sampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} Sampler", label)),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: if mip_levels > 1 {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    /// Create a 2D array view suitable for compute shader storage texture binding.
    ///
    /// This creates a view of a single mip level as a 2D array with 6 layers,
    /// which can be bound as a storage texture for compute shaders to write to.
    pub fn create_storage_view(
        &self,
        mip_level: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("Cubemap Storage View Mip {}", mip_level)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        })
    }
}
