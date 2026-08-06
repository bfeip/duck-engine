//! Upload of scene textures to the GPU.
//!
//! Mirrors [`mesh`](super::mesh) as the parallel "turn a scene resource into GPU
//! resources" helper, producing a [`GpuTexture`] the renderer keeps
//! generation-synced in a `GenCache`.

use anyhow::Result;
use image::{imageops::FilterType, GenericImageView};

use crate::render_core::GpuTexture;
use crate::scene::resource::Texture;

/// Create GPU resources for a texture.
pub(crate) fn create_texture_gpu_resources(
    texture: &Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<GpuTexture> {
    let texture_id = texture.id();
    let img = texture.get_image()?;
    let dimensions = img.dimensions();

    // Resize if texture exceeds device limits (e.g. WebGL max 2048px)
    let max_dim = device.limits().max_texture_dimension_2d;
    let img = if dimensions.0 > max_dim || dimensions.1 > max_dim {
        log::info!(
            "Texture {} ({}x{}) exceeds max dimension {}; resizing",
            texture_id,
            dimensions.0,
            dimensions.1,
            max_dim,
        );
        std::borrow::Cow::Owned(img.resize(max_dim, max_dim, FilterType::Triangle))
    } else {
        std::borrow::Cow::Borrowed(img)
    };

    let rgba = img.to_rgba8();
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let wgpu_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture: &wgpu_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        size,
    );

    let view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Ok(GpuTexture {
        texture: wgpu_texture,
        view,
        sampler,
    })
}
