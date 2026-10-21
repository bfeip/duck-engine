//! The BRDF integration lookup table of the split-sum approximation.
//!
//! Indexed by N·V and roughness; holds the scale and bias applied to F0.
//! Baked by `build.rs`.

use crate::render_core::{Gpu, GpuTexture};
use crate::BRDF_LUT_SIZE;

/// `BRDF_LUT_SIZE`² texels of RGBA f32, little-endian.
const BAKED_BRDF_LUT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/brdf_lut.bin"));

/// Upload the baked BRDF LUT as an Rgba16Float texture.
pub fn brdf_lut(gpu: &Gpu) -> GpuTexture {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BRDF LUT Texture"),
        size: wgpu::Extent3d {
            width: BRDF_LUT_SIZE,
            height: BRDF_LUT_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Convert f32 RGBA to f16 RGBA. The baked data is a byte blob from
    // `include_bytes!`, so it carries no alignment and must be decoded as
    // the little-endian f32s build.rs wrote rather than cast.
    let pixel_count = (BRDF_LUT_SIZE * BRDF_LUT_SIZE) as usize;
    let mut f16_data = vec![0u16; pixel_count * 4];
    for (channel, bytes) in f16_data.iter_mut().zip(BAKED_BRDF_LUT.chunks_exact(4)) {
        let value = f32::from_le_bytes(bytes.try_into().expect("chunks_exact(4) yields 4 bytes"));
        *channel = half::f16::from_f32(value).to_bits();
    }

    let bytes_per_pixel = 8; // Rgba16Float = 4 * 2 bytes
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&f16_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BRDF_LUT_SIZE * bytes_per_pixel),
            rows_per_image: Some(BRDF_LUT_SIZE),
        },
        wgpu::Extent3d {
            width: BRDF_LUT_SIZE,
            height: BRDF_LUT_SIZE,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("BRDF LUT Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    GpuTexture { texture, view, sampler }
}
