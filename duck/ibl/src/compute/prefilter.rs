//! Pre-filtered environment map generation for specular IBL.
//!
//! Uses GGX importance sampling to generate a pre-filtered environment cubemap
//! where each mip level represents a different roughness value.

use super::storage_view;
use crate::{cube_texture, CUBEMAP_FORMAT, PREFILTERED_CUBEMAP_SIZE, PREFILTERED_MIP_LEVELS};
use crate::render_core::GpuTexture;

/// Shader source for pre-filter convolution.
const PREFILTER_SHADER: &str = include_str!("../shaders/prefilter.wgsl");

/// Uniform buffer for pre-filter parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefilterParams {
    roughness: f32,
    _padding: [f32; 3],
}

/// Pipeline for generating pre-filtered environment cubemaps.
pub struct PrefilterPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl PrefilterPipeline {
    /// Create a new pre-filter generation pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Prefilter Shader"),
            source: wgpu::ShaderSource::Wgsl(PREFILTER_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Prefilter Bind Group Layout"),
            entries: &[
                // Input environment cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler for environment cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Output pre-filtered storage texture (single mip level)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: CUBEMAP_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                // Roughness parameter
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Prefilter Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Prefilter Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Generate a pre-filtered cubemap from an environment cubemap.
    ///
    /// Each mip level is generated with increasing roughness:
    /// - Mip 0: roughness = 0.0 (mirror reflection)
    /// - Mip N-1: roughness = 1.0 (fully diffuse)
    pub fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environment: &GpuTexture,
    ) -> GpuTexture {
        // Create output pre-filtered cubemap with mip chain
        let prefiltered = cube_texture(
            device,
            PREFILTERED_CUBEMAP_SIZE,
            PREFILTERED_MIP_LEVELS,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "Prefiltered Cubemap",
        );

        // Create uniform buffer for roughness parameter
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefilter Params Buffer"),
            size: std::mem::size_of::<PrefilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Generate each mip level
        for mip in 0..PREFILTERED_MIP_LEVELS {
            let roughness = mip as f32 / (PREFILTERED_MIP_LEVELS - 1) as f32;
            let mip_size = PREFILTERED_CUBEMAP_SIZE >> mip;

            // Update roughness parameter
            let params = PrefilterParams {
                roughness,
                _padding: [0.0; 3],
            };
            queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

            // Create storage view for this mip level
            let output_view = storage_view(&prefiltered, mip);

            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Prefilter Bind Group Mip {}", mip)),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&environment.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&environment.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch compute shader for this mip level
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Prefilter Encoder Mip {}", mip)),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Prefilter Pass Mip {}", mip)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = mip_size.div_ceil(8);
                let workgroups_y = mip_size.div_ceil(8);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 6);
            }

            queue.submit(std::iter::once(encoder.finish()));
        }

        prefiltered
    }
}
