//! Equirectangular to cubemap conversion using a compute shader.

use super::{storage_view, ENVIRONMENT_CUBEMAP_SIZE};
use crate::{cube_texture, HdrImage, CUBEMAP_FORMAT};
use crate::render_core::GpuTexture;

/// Shader source for equirect to cube conversion.
const EQUIRECT_TO_CUBE_SHADER: &str = include_str!("../shaders/equirect_to_cube.wgsl");

/// Resources for converting equirectangular images to cubemaps.
pub struct EquirectToCubePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl EquirectToCubePipeline {
    /// Create a new equirect-to-cube pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Equirect to Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(EQUIRECT_TO_CUBE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Equirect to Cube Bind Group Layout"),
            entries: &[
                // Input equirectangular texture
                // Use non-filterable since Rgba32Float doesn't support filtering
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler for equirectangular texture (non-filtering)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Output cubemap storage texture
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
            label: Some("Equirect to Cube Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Equirect to Cube Pipeline"),
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

    /// Convert an equirectangular HDR image to a cubemap.
    ///
    /// Returns the environment cubemap.
    pub fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hdr_image: &HdrImage,
    ) -> GpuTexture {
        // Create the input 2D texture from HDR data
        let input_texture = self.create_input_texture(device, queue, hdr_image);
        let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Use nearest filtering since Rgba32Float doesn't support linear filtering
        let input_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Equirect Input Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create output cubemap
        let output_cubemap = cube_texture(
            device,
            ENVIRONMENT_CUBEMAP_SIZE,
            1, // No mipmaps for the environment cubemap itself
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            "Environment Cubemap",
        );

        // Create storage view for compute shader output
        let output_view = storage_view(&output_cubemap, 0);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Equirect to Cube Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&input_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Equirect to Cube Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Equirect to Cube Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: one workgroup per 16x16 pixels, for each of 6 faces
            let workgroups_x = ENVIRONMENT_CUBEMAP_SIZE.div_ceil(16);
            let workgroups_y = ENVIRONMENT_CUBEMAP_SIZE.div_ceil(16);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 6);
        }

        queue.submit(std::iter::once(encoder.finish()));

        output_cubemap
    }

    /// Create a 2D texture from HDR image data.
    fn create_input_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hdr_image: &HdrImage,
    ) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: hdr_image.width,
            height: hdr_image.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Equirect HDR Input"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float, // HDR requires float format
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Convert RGB f32 data to RGBA f32 (add alpha channel)
        let mut rgba_data = Vec::with_capacity(hdr_image.data.len() / 3 * 4);
        for chunk in hdr_image.data.chunks(3) {
            rgba_data.push(chunk[0]);
            rgba_data.push(chunk[1]);
            rgba_data.push(chunk[2]);
            rgba_data.push(1.0);
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(hdr_image.width * 4 * 4), // 4 channels * 4 bytes per f32
                rows_per_image: Some(hdr_image.height),
            },
            size,
        );

        texture
    }
}
