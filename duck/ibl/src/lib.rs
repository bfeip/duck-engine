//! Image-based lighting: equirectangular HDR environments baked into the
//! textures split-sum PBR shading samples.
//!
//! [`IblMaps::from_equirect`] bakes an [`HdrImage`] into a diffuse irradiance
//! cubemap and a roughness-prefiltered specular cubemap, and
//! [`IblMaps::upload`] turns them into [`IblTextures`]. [`brdf_lut`] uploads
//! the BRDF lookup table that goes with them, baked at build time.
//! [`load_hdr_from_path`] and its siblings read Radiance HDR files.

#[cfg(test)]
mod brdf;
mod brdf_lut;
mod cube;
mod equirect;
mod hdr;
mod irradiance;
mod prefilter;
mod sampling;

/// The scene-agnostic rendering core this crate is built on.
pub use duck_engine_render_core as render_core;

pub use brdf_lut::brdf_lut;
pub use cube::{CubeImage, Rgb};
pub use hdr::{load_hdr_from_bytes, load_hdr_from_path, load_hdr_from_reader, HdrImage};

use render_core::{Gpu, GpuTexture};

/// Size of the irradiance cubemap (per face). Low resolution since irradiance is low-frequency.
pub const IRRADIANCE_CUBEMAP_SIZE: u32 = 32;

/// Size of the pre-filtered environment cubemap base level (per face).
pub const PREFILTERED_CUBEMAP_SIZE: u32 = 128;

/// Number of mip levels for the pre-filtered cubemap (roughness levels).
pub const PREFILTERED_MIP_LEVELS: u32 = 5;

/// Size of the BRDF integration LUT.
pub const BRDF_LUT_SIZE: u32 = 512;

/// The cubemaps PBR shading samples, baked from one environment.
#[derive(Debug, Clone)]
pub struct IblMaps {
    /// Diffuse irradiance over π, [`IRRADIANCE_CUBEMAP_SIZE`]² faces.
    pub irradiance: CubeImage,
    /// Specular radiance, [`PREFILTERED_CUBEMAP_SIZE`]² faces, one level per
    /// roughness step over [`PREFILTERED_MIP_LEVELS`].
    pub prefiltered: CubeImage,
}

impl IblMaps {
    /// Bake the maps for an equirectangular environment.
    pub fn from_equirect(hdr: &HdrImage) -> Self {
        let environment =
            equirect::cube_from_equirect(hdr, PREFILTERED_CUBEMAP_SIZE).with_mips();
        Self {
            irradiance: irradiance::irradiance(&environment, IRRADIANCE_CUBEMAP_SIZE),
            prefiltered: prefilter::prefilter(&environment, PREFILTERED_MIP_LEVELS),
        }
    }

    /// Upload the maps as the textures shading samples.
    pub fn upload(&self, gpu: &Gpu) -> IblTextures {
        IblTextures {
            irradiance: upload_cube(gpu, &self.irradiance, "Irradiance Cubemap"),
            prefiltered: upload_cube(gpu, &self.prefiltered, "Prefiltered Cubemap"),
        }
    }
}

/// The uploaded cubemaps of one environment.
pub struct IblTextures {
    /// Diffuse irradiance over π, [`IRRADIANCE_CUBEMAP_SIZE`]² faces.
    pub irradiance: GpuTexture,
    /// Specular radiance, [`PREFILTERED_CUBEMAP_SIZE`]² faces, one mip per
    /// roughness step over [`PREFILTERED_MIP_LEVELS`].
    pub prefiltered: GpuTexture,
}

/// `cube` as an Rgba16Float cube texture with all its levels, sampled linearly.
fn upload_cube(gpu: &Gpu, cube: &CubeImage, label: &str) -> GpuTexture {
    let format = wgpu::TextureFormat::Rgba16Float;
    let mip_levels = cube.level_count();
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: cube.size(),
            height: cube.size(),
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for level in 0..mip_levels {
        let size = cube.level_size(level);
        let texels: Vec<[u16; 4]> = cube
            .level(level)
            .iter()
            .map(|&[r, g, b]| [r, g, b, 1.0].map(|c| half::f16::from_f32(c).to_bits()))
            .collect();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 8), // Rgba16Float = 4 * 2 bytes
                rows_per_image: Some(size),
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
        );
    }

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

    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
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

    GpuTexture { texture, view, sampler }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-2;

    #[test]
    fn uniform_environment_bakes_uniform_maps() {
        let hdr = HdrImage { width: 64, height: 32, data: [0.5, 1.0, 2.0].repeat(64 * 32) };
        let maps = IblMaps::from_equirect(&hdr);

        assert_eq!(maps.irradiance.size(), IRRADIANCE_CUBEMAP_SIZE);
        assert_eq!(maps.irradiance.level_count(), 1);
        assert_eq!(maps.prefiltered.size(), PREFILTERED_CUBEMAP_SIZE);
        assert_eq!(maps.prefiltered.level_count(), PREFILTERED_MIP_LEVELS);

        let levels = std::iter::once(maps.irradiance.level(0))
            .chain((0..PREFILTERED_MIP_LEVELS).map(|l| maps.prefiltered.level(l)));
        for texels in levels {
            for texel in texels {
                assert!(
                    (0..3).all(|c| (texel[c] / [0.5, 1.0, 2.0][c] - 1.0).abs() < EPSILON),
                    "{texel:?}"
                );
            }
        }
    }
}
