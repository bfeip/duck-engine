//! Image-based lighting: environment maps as light sources for PBR shading.
//!
//! An [`EnvironmentMap`] is a scene resource naming an equirectangular HDR
//! source ([`EnvironmentSource`]) plus an intensity. When a scene's active
//! environment map is prepared for rendering, its HDR image is baked into the
//! textures PBR shading samples — an irradiance cubemap, a
//! roughness-prefiltered cubemap, and a BRDF lookup table — and lit materials
//! pick it up automatically. Baking runs in compute shaders where the adapter
//! has them, and on the CPU ([`IblMaps`]) otherwise.
//!
//! This module also exposes the Radiance HDR loader ([`HdrImage`],
//! [`load_hdr_from_path`], [`load_hdr_from_bytes`], [`load_hdr_from_reader`])
//! and the sizes of the generated textures.

mod brdf_lut;
mod compute;
mod cpu;
mod hdr_loader;

pub use cpu::{CubeImage, IblMaps, Rgb};
pub use hdr_loader::{load_hdr_from_bytes, load_hdr_from_path, load_hdr_from_reader, HdrImage};

use brdf_lut::brdf_lut;
use compute::ComputeBaker;

use wgpu::util::DeviceExt;

use crate::render_core::GpuTexture;

// Re-export EnvironmentMap types from scene crate
pub use duck_engine_scene::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};

/// GPU uniform for environment map parameters (intensity, etc.).
///
/// Bound at binding 6 in the IBL bind group, updated per-frame without
/// retriggering expensive IBL texture processing.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvironmentUniform {
    intensity: f32,
    _padding: [u32; 3],
}

/// Size of the environment cubemap (per face) the compute baker resamples the
/// HDR onto.
pub const ENVIRONMENT_CUBEMAP_SIZE: u32 = 512;

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

/// The baked cubemaps of one environment.
pub(crate) struct IblTextures {
    /// Diffuse irradiance over π, [`IRRADIANCE_CUBEMAP_SIZE`]² faces.
    pub irradiance: GpuTexture,
    /// Specular radiance, [`PREFILTERED_CUBEMAP_SIZE`]² faces, one mip per
    /// roughness step over [`PREFILTERED_MIP_LEVELS`].
    pub prefiltered: GpuTexture,
}

/// Bakes environments into [`IblTextures`]: in compute shaders where the
/// adapter has them, on the CPU otherwise.
pub(crate) enum IblBaker {
    Compute(ComputeBaker),
    Cpu,
}

impl IblBaker {
    pub fn new(device: &wgpu::Device, has_compute: bool) -> Self {
        if has_compute {
            Self::Compute(ComputeBaker::new(device))
        } else {
            Self::Cpu
        }
    }

    /// Bake the textures for an equirectangular environment.
    pub fn bake(&self, device: &wgpu::Device, queue: &wgpu::Queue, hdr: &HdrImage) -> IblTextures {
        match self {
            Self::Compute(baker) => baker.bake(device, queue, hdr),
            Self::Cpu => IblMaps::from_equirect(hdr).upload(device, queue),
        }
    }
}

/// Processed GPU resources for an environment map.
#[derive(Debug)]
pub(crate) struct ProcessedEnvironment {
    /// Uniform buffer for environment parameters (intensity).
    pub params_buffer: wgpu::Buffer,
    /// Bind group for sampling in fragment shader.
    pub bind_group: wgpu::BindGroup,
}

/// GPU resources for Image Based Lighting.
///
/// Bakes each environment map once, with the shared BRDF lookup table, into
/// the bind group shading samples.
pub(crate) struct IblResources {
    baker: IblBaker,
    /// Shared BRDF lookup table, baked at build time.
    brdf_lut: GpuTexture,
    /// Cheap (ref-counted) clone of the IBL bind group layout, whose canonical
    /// owner is the renderer's `BindGroupLayouts` registry. Kept here so
    /// per-environment bind groups can be built without threading the layout
    /// through `process_environment`.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Processed environment maps by ID, with the generation they were synced at.
    processed_environments: std::collections::HashMap<EnvironmentMapId, (ProcessedEnvironment, u64)>,
}

/// Create the bind group layout for IBL sampling in fragment shaders.
/// 
/// Always needed for pipeline layout creation even when IBL is unavailable
/// (no compute shaders), so the layout exists independently of whether any
/// environment is processed.
pub(crate) fn ibl_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("IBL Bind Group Layout"),
        entries: &[
            // Irradiance cubemap
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            // Irradiance sampler
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Pre-filtered cubemap
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            // Pre-filtered sampler
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // BRDF LUT
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // BRDF LUT sampler
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Environment parameters (intensity, etc.)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

impl IblResources {
    /// Create new IBL resources, baking in compute shaders when `has_compute`
    /// and on the CPU otherwise.
    ///
    /// `layout` is the canonical IBL bind group layout owned by the renderer's
    /// `BindGroupLayouts` registry; a cheap clone is retained for building
    /// per-environment bind groups.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        has_compute: bool,
    ) -> Self {
        if !has_compute {
            log::info!("Compute shaders unavailable — baking IBL environment maps on the CPU.");
        }
        Self {
            baker: IblBaker::new(device, has_compute),
            brdf_lut: brdf_lut(device, queue),
            bind_group_layout: layout.clone(),
            processed_environments: std::collections::HashMap::new(),
        }
    }

    /// Process an environment map, generating all required GPU resources.
    ///
    /// If the environment has already been processed (same ID), only updates
    /// the params buffer (intensity). Full texture reprocessing is skipped
    /// since the source cannot change on an existing environment map.
    pub fn process_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        env_map: &EnvironmentMap,
    ) -> anyhow::Result<()> {
        // Already processed — just update params (intensity, etc.)
        if self.processed_environments.contains_key(&env_map.id) {
            self.update_env_params(queue, env_map);
            return Ok(());
        }

        let hdr_image = match env_map.source() {
            EnvironmentSource::EquirectangularPath(path) => load_hdr_from_path(path)?,
            EnvironmentSource::EquirectangularHdr(bytes) => load_hdr_from_bytes(bytes)?,
        };
        let textures = self.baker.bake(device, queue, &hdr_image);
        self.store_processed(device, env_map, &textures);
        Ok(())
    }

    /// Create the bind group and store the processed environment.
    fn store_processed(
        &mut self,
        device: &wgpu::Device,
        env_map: &EnvironmentMap,
        textures: &IblTextures,
    ) {
        // Create uniform buffer for environment parameters
        let env_uniform = EnvironmentUniform {
            intensity: env_map.intensity(),
            _padding: [0; 3],
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("IBL Params Buffer Env {}", env_map.id)),
            contents: bytemuck::cast_slice(&[env_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group for fragment shader sampling
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("IBL Bind Group Env {}", env_map.id)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&textures.irradiance.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&textures.irradiance.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&textures.prefiltered.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&textures.prefiltered.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.brdf_lut.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.brdf_lut.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Store processed environment
        self.processed_environments.insert(
            env_map.id,
            (
                ProcessedEnvironment {
                    params_buffer,
                    bind_group,
                },
                env_map.generation(),
            ),
        );
    }

    /// Update the environment params uniform buffer (intensity) without reprocessing textures.
    fn update_env_params(&self, queue: &wgpu::Queue, env_map: &EnvironmentMap) {
        if let Some((processed, _)) = self.processed_environments.get(&env_map.id) {
            let env_uniform = EnvironmentUniform {
                intensity: env_map.intensity(),
                _padding: [0; 3],
            };
            queue.write_buffer(
                &processed.params_buffer,
                0,
                bytemuck::cast_slice(&[env_uniform]),
            );
        }
    }

    /// Get the processed environment for an ID, if available.
    pub fn get_processed(&self, id: EnvironmentMapId) -> Option<&ProcessedEnvironment> {
        self.processed_environments.get(&id).map(|(env, _)| env)
    }
}
