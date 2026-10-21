//! Image-based lighting: equirectangular HDR environments baked into the
//! textures split-sum PBR shading samples.
//!
//! [`IblBaker`] bakes an [`HdrImage`] into [`IblTextures`] — a diffuse
//! irradiance cubemap and a roughness-prefiltered specular cubemap — in compute
//! shaders where the adapter has them, and on the CPU ([`IblMaps`]) otherwise.
//! [`brdf_lut`] uploads the BRDF lookup table that goes with them, baked at
//! build time. [`load_hdr_from_path`] and its siblings read Radiance HDR files.

mod brdf_lut;
mod compute;
mod cpu;
mod hdr;

/// The scene-agnostic rendering core this crate is built on.
pub use duck_engine_render_core as render_core;

pub use brdf_lut::brdf_lut;
pub use compute::ComputeBaker;
pub use cpu::{CubeImage, IblMaps, Rgb};
pub use hdr::{load_hdr_from_bytes, load_hdr_from_path, load_hdr_from_reader, HdrImage};

use render_core::{Gpu, GpuCapabilities, GpuTexture};

/// Size of the irradiance cubemap (per face). Low resolution since irradiance is low-frequency.
pub const IRRADIANCE_CUBEMAP_SIZE: u32 = 32;

/// Size of the pre-filtered environment cubemap base level (per face).
pub const PREFILTERED_CUBEMAP_SIZE: u32 = 128;

/// Number of mip levels for the pre-filtered cubemap (roughness levels).
pub const PREFILTERED_MIP_LEVELS: u32 = 5;

/// Size of the BRDF integration LUT.
pub const BRDF_LUT_SIZE: u32 = 512;

/// Texel format of every IBL cubemap.
const CUBEMAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The baked cubemaps of one environment.
pub struct IblTextures {
    /// Diffuse irradiance over π, [`IRRADIANCE_CUBEMAP_SIZE`]² faces.
    pub irradiance: GpuTexture,
    /// Specular radiance, [`PREFILTERED_CUBEMAP_SIZE`]² faces, one mip per
    /// roughness step over [`PREFILTERED_MIP_LEVELS`].
    pub prefiltered: GpuTexture,
}

/// Bakes environments into [`IblTextures`].
pub enum IblBaker {
    /// In compute shaders.
    Compute(ComputeBaker),
    /// On the CPU, through [`IblMaps`].
    Cpu,
}

impl IblBaker {
    /// The compute baker where `capabilities` allow it, the CPU one otherwise.
    pub fn new(gpu: &Gpu, capabilities: GpuCapabilities) -> Self {
        if capabilities.has_compute {
            Self::Compute(ComputeBaker::new(gpu))
        } else {
            Self::Cpu
        }
    }

    /// Bake the textures for an equirectangular environment.
    pub fn bake(&self, gpu: &Gpu, hdr: &HdrImage) -> IblTextures {
        match self {
            Self::Compute(baker) => baker.bake(gpu, hdr),
            Self::Cpu => IblMaps::from_equirect(hdr).upload(gpu),
        }
    }
}

/// A cubemap texture with `size`² faces and `mip_levels` levels, viewed as a
/// cube and sampled linearly.
fn cube_texture(
    device: &wgpu::Device,
    size: u32,
    mip_levels: u32,
    usage: wgpu::TextureUsages,
    label: &str,
) -> GpuTexture {
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
        format: CUBEMAP_FORMAT,
        usage,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{} Cube View", label)),
        format: Some(CUBEMAP_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_levels),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });

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

    GpuTexture { texture, view, sampler }
}
