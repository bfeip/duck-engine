//! Irradiance map generation using a compute shader.
//!
//! The irradiance map stores the diffuse component of IBL lighting by
//! convolving the environment cubemap with a cosine-weighted hemisphere.

use super::storage_view;
use crate::{cube_texture, CUBEMAP_FORMAT, IRRADIANCE_CUBEMAP_SIZE};
use crate::render_core::GpuTexture;

/// Shader source for irradiance convolution.
const IRRADIANCE_SHADER: &str = include_str!("../shaders/irradiance.wgsl");

/// Pipeline for generating irradiance cubemaps.
pub struct IrradiancePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl IrradiancePipeline {
    /// Create a new irradiance generation pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Irradiance Shader"),
            source: wgpu::ShaderSource::Wgsl(IRRADIANCE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Irradiance Bind Group Layout"),
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
                // Output irradiance storage texture
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Irradiance Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Irradiance Pipeline"),
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

    /// Generate an irradiance cubemap from an environment cubemap.
    pub fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environment: &GpuTexture,
    ) -> GpuTexture {
        // Create output irradiance cubemap (small since irradiance is low-frequency)
        let irradiance = cube_texture(
            device,
            IRRADIANCE_CUBEMAP_SIZE,
            1,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "Irradiance Cubemap",
        );

        // Create storage view for compute output
        let output_view = storage_view(&irradiance, 0);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Irradiance Bind Group"),
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
            ],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Irradiance Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Irradiance Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: workgroup size is 8x8, for each of 6 faces
            let workgroups_x = IRRADIANCE_CUBEMAP_SIZE.div_ceil(8);
            let workgroups_y = IRRADIANCE_CUBEMAP_SIZE.div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 6);
        }

        queue.submit(std::iter::once(encoder.finish()));

        irradiance
    }
}
