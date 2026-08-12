//! The renderer's shared bind group layout registry.
//!
//! A bind group layout is a *schema*: exactly one per kind, created once and
//! shared by every pipeline layout and every conforming bind group.

use crate::ibl::ibl_bind_group_layout;

/// Canonical owner of every bind group layout used by the renderer.
///
/// A bind group layout is a *schema*: exactly one per kind, created once and
/// shared by every pipeline layout and every conforming bind group. Keeping them
/// all here (rather than bundled into the resource structs that happen to create
/// the first bind group of each kind) lets multiple instances — e.g. one camera
/// binding per view — share a single layout, and gives pipeline-layout
/// construction one place to borrow from.
pub(crate) struct BindGroupLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub light: wgpu::BindGroupLayout,
    /// Color material layout for the standalone flat-color overlay shader
    /// (a single `vec4` uniform). Surface materials use the dynamically-derived
    /// layouts in [`MaterialLayoutCache`](super::pipeline::MaterialLayoutCache) instead.
    pub color: wgpu::BindGroupLayout,
    pub ibl: wgpu::BindGroupLayout,
}

impl BindGroupLayouts {
    /// Create all bind group layouts.
    pub fn new(device: &wgpu::Device) -> BindGroupLayouts {
        let camera = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Visible in both VERTEX (for view_proj) and FRAGMENT (for eye_position in PBR)
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let light = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Light bind group layout"),
        });

        let color = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Material Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let ibl = ibl_bind_group_layout(device);

        BindGroupLayouts { camera, light, color, ibl }
    }
}
