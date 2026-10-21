//! Environment maps bound for shading: one baked IBL bind group per map.

use wgpu::util::DeviceExt;

use crate::ibl::{brdf_lut, load_hdr_from_bytes, load_hdr_from_path, IblMaps, IblTextures};
use crate::render_core::{Gpu, GpuTexture};
use crate::scene::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};

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

impl IblResources {
    /// Create new IBL resources.
    ///
    /// `layout` is the canonical IBL bind group layout owned by the renderer's
    /// `BindGroupLayouts` registry; a cheap clone is retained for building
    /// per-environment bind groups.
    pub fn new(gpu: &Gpu, layout: &wgpu::BindGroupLayout) -> Self {
        Self {
            brdf_lut: brdf_lut(gpu),
            bind_group_layout: layout.clone(),
            processed_environments: std::collections::HashMap::new(),
        }
    }

    /// Process an environment map, generating all required GPU resources.
    ///
    /// If the environment has already been processed (same ID), only updates
    /// the params buffer (intensity). Full texture reprocessing is skipped
    /// since the source cannot change on an existing environment map.
    pub fn process_environment(&mut self, gpu: &Gpu, env_map: &EnvironmentMap) -> anyhow::Result<()> {
        // Already processed — just update params (intensity, etc.)
        if self.processed_environments.contains_key(&env_map.id) {
            self.update_env_params(&gpu.queue, env_map);
            return Ok(());
        }

        let hdr_image = match env_map.source() {
            EnvironmentSource::EquirectangularPath(path) => load_hdr_from_path(path)?,
            EnvironmentSource::EquirectangularHdr(bytes) => load_hdr_from_bytes(bytes)?,
        };
        let textures = IblMaps::from_equirect(&hdr_image).upload(gpu);
        self.store_processed(&gpu.device, env_map, &textures);
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

    /// Evict the baked textures and bind group of a removed environment map.
    pub fn remove(&mut self, id: EnvironmentMapId) {
        self.processed_environments.remove(&id);
    }

    /// Evict every processed environment map.
    pub fn clear(&mut self) {
        self.processed_environments.clear();
    }

    /// Get the processed environment for an ID, if available.
    pub fn get_processed(&self, id: EnvironmentMapId) -> Option<&ProcessedEnvironment> {
        self.processed_environments.get(&id).map(|(env, _)| env)
    }
}
