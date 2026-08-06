//! Mesh geometry on the GPU: vertex/index buffers, the instance data format,
//! and the draw helpers that issue instanced draw calls.
//!
//! A [`Mesh`] is uploaded once into a [`MeshGpuResources`] (one vertex buffer plus
//! a per-primitive-type index buffer); the renderer keeps those generation-synced
//! in a `GenCache`. [`GpuInstance`] is the per-instance vertex data, and
//! [`vertex_buffer_layout`]/[`instance_buffer_layout`] describe the two vertex
//! buffers a surface pipeline binds.

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::renderer::batching::InstanceTransform;
use crate::scene::resource::{Mesh, MeshIndex, PrimitiveType, Vertex};

/// GPU-ready instance data for instanced rendering.
///
/// Contains a world transform matrix and normal matrix packed for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstance {
    pub transform: [[f32; 4]; 4],
    pub normal_mat: [[f32; 3]; 3],
}

/// GPU resources for a mesh (vertex and index buffers).
pub(crate) struct MeshGpuResources {
    pub vertex_buffer: wgpu::Buffer,
    pub triangle_index_buffer: wgpu::Buffer,
    pub line_index_buffer: wgpu::Buffer,
    pub point_index_buffer: wgpu::Buffer,
}

impl MeshGpuResources {
    /// Upload a [`Mesh`] to the GPU: one vertex buffer plus a per-primitive-type
    /// index buffer.
    pub(crate) fn new(mesh: &Mesh, device: &wgpu::Device) -> MeshGpuResources {
        let vertices = mesh.vertices();
        let primitives = mesh.primitives();

        // Create vertex buffer
        let vertex_buffer = if vertices.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh Vertex Buffer"),
                size: 0,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        };

        // Helper to create index buffer for a specific primitive type
        let create_index_buffer = |prim_type: PrimitiveType, label: &str| -> wgpu::Buffer {
            let indices: Vec<MeshIndex> = primitives
                .iter()
                .filter(|p| p.primitive_type == prim_type)
                .flat_map(|p| p.indices.iter().copied())
                .collect();

            if indices.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 0,
                    usage: wgpu::BufferUsages::INDEX,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
            }
        };

        let triangle_index_buffer = create_index_buffer(PrimitiveType::TriangleList, "Triangle Index Buffer");
        let line_index_buffer = create_index_buffer(PrimitiveType::LineList, "Line Index Buffer");
        let point_index_buffer = create_index_buffer(PrimitiveType::PointList, "Point Index Buffer");

        MeshGpuResources {
            vertex_buffer,
            triangle_index_buffer,
            line_index_buffer,
            point_index_buffer,
        }
    }

    /// Draw `instance_transforms` instances of this mesh's `primitive_type`
    /// geometry. Creates an instance buffer and issues one instanced draw call.
    pub(crate) fn draw_instances(
        &self,
        device: &wgpu::Device,
        render_pass: &mut wgpu::RenderPass,
        primitive_type: PrimitiveType,
        instance_transforms: &[InstanceTransform],
        index_count: u32,
    ) {
        // Skip drawing if there are no indices for this primitive type
        if index_count == 0 {
            return;
        }

        // Convert InstanceTransforms to GpuInstance for the GPU
        let instance_raws: Vec<GpuInstance> = instance_transforms
            .iter()
            .map(|inst_transform| GpuInstance {
                // Upload the effective (screen-space-adjusted) transform. For
                // ordinary geometry this equals the world transform.
                transform: inst_transform.effective_transform.into(),
                normal_mat: inst_transform.effective_normal_matrix.into(),
            })
            .collect();

        // Create instance buffer
        // TODO: Consider caching/reusing instance buffers for better performance
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_raws),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let n_instances = instance_transforms.len() as u32;
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer(primitive_type).slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..index_count, 0, 0..n_instances);
    }

    /// Draw a single sub-range of this mesh's index buffer for one instance.
    ///
    /// Used to render individual faces or edges (e.g. for highlight outlines).
    /// `first_index` and `index_count` are raw index counts (not triangle/segment counts).
    pub(crate) fn draw_subgeom(
        &self,
        device: &wgpu::Device,
        render_pass: &mut wgpu::RenderPass,
        primitive_type: PrimitiveType,
        instance_transform: &InstanceTransform,
        first_index: u32,
        index_count: u32,
    ) {
        if index_count == 0 {
            return;
        }

        let instance_raw = GpuInstance {
            transform: instance_transform.world_transform.into(),
            normal_mat: instance_transform.normal_matrix.into(),
        };

        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("SubGeom Instance Buffer"),
            contents: bytemuck::cast_slice(&[instance_raw]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer(primitive_type).slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(first_index..(first_index + index_count), 0, 0..1);
    }

    /// The index buffer for `primitive_type`.
    fn index_buffer(&self, primitive_type: PrimitiveType) -> &wgpu::Buffer {
        match primitive_type {
            PrimitiveType::TriangleList => &self.triangle_index_buffer,
            PrimitiveType::LineList => &self.line_index_buffer,
            PrimitiveType::PointList => &self.point_index_buffer,
        }
    }
}

// Vertex shader attribute locations
pub(crate) enum VertexShaderLocations {
    VertexPosition = 0,
    TextureCoords,
    VertexNormal,
    InstanceTransformRow0,
    InstanceTransformRow1,
    InstanceTransformRow2,
    InstanceTransformRow3,
    InstanceNormalRow0,
    InstanceNormalRow1,
    InstanceNormalRow2,
}

/// Returns the vertex buffer layout for [`Vertex`] structs.
///
/// Prefer [`Renderer::custom_pipeline_builder`] for building custom render pipelines —
/// it applies both buffer layouts automatically. These functions are a lower-level
/// escape hatch for cases where the builder's defaults don't fit. Callers that use
/// the layouts directly must also duplicate the engine's shader struct definitions,
/// which will silently break if the engine types change.
pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: VertexShaderLocations::VertexPosition as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: VertexShaderLocations::TextureCoords as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3 * 2]>() as wgpu::BufferAddress,
                shader_location: VertexShaderLocations::VertexNormal as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Returns the instance buffer layout for `GpuInstance` structs.
///
/// See [`vertex_buffer_layout`] — the same caveats apply.
pub fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    use VertexShaderLocations as VSL;

    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: VSL::InstanceTransformRow0 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 1]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow1 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 2]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow2 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 3]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow3 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 4]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow0 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; (4 * 4) + (3 * 1)]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow1 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; (4 * 4) + (3 * 2)]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow2 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

