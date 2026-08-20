//! Per-material GPU bind groups, generation-synced to one scene's materials.
//!
//! The pipelines and layouts these bind groups conform to live in the
//! [`MaterialPipelineCache`], which belongs to the render context and is shared
//! across scenes; this cache holds only the per-material uniforms and bind
//! groups, which are scene state. Passes reach it through
//! [`SceneFrame::materials`](super::pass_context::SceneFrame).
//!
//! A material binds exactly the textures it has: the bind-group layout is derived
//! from the same [`SurfaceConfig`] that drives the shader, so there are no
//! fallback textures and no texture-presence flags.

use anyhow::{anyhow, Result};
use bytemuck::bytes_of;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::render_core::{GenCache, GpuTexture};
use crate::scene::resource::{
    FaceMaterial, FaceMaterialId, GenericId, LineMaterial, LineMaterialId, PointMaterial,
    PointMaterialId, ResourceKind, TextureId,
};
use crate::scene::SceneData;
use crate::scene::common::RgbaColor;

use super::batching::BatchMaterial;
use super::pipeline::MaterialPipelineCache;
use super::surface_config::{MaterialTextureSlot, SurfaceConfig, TexturePresence};

/// Material parameters for the surface shader's group-2 uniform.
///
/// One layout serves every variant (lit and unlit). Which textures are bound is
/// decided by the variant's bind-group layout, not by this uniform — there are
/// no texture-presence flags. Unlit materials simply ignore the lighting-only
/// factors. Must match `MaterialParams` in `material.wesl` (32 bytes: a `vec4`
/// followed by four `f32`, already 16-byte aligned).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub alpha_cutoff: f32,
}

impl MaterialUniform {
    /// Build from a (lit or unlit) `FaceMaterial`.
    pub fn from_face_material(material: &FaceMaterial) -> Self {
        let base_color = material.base_color_factor();
        MaterialUniform {
            base_color_factor: [base_color.r, base_color.g, base_color.b, base_color.a],
            metallic_factor: material.metallic_factor(),
            roughness_factor: material.roughness_factor(),
            normal_scale: material.normal_scale(),
            alpha_cutoff: material.alpha_cutoff(),
        }
    }

    /// Build from a `LineMaterial` (unlit: only the color is used).
    pub fn from_line_material(material: &LineMaterial) -> Self {
        Self::unlit(material.color())
    }

    /// Build from a `PointMaterial` (unlit: only the color is used).
    pub fn from_point_material(material: &PointMaterial) -> Self {
        Self::unlit(material.color())
    }

    /// Shared constructor for unlit color materials. The lighting-only factors
    /// are unused by the unlit path.
    fn unlit(color: RgbaColor) -> Self {
        MaterialUniform {
            base_color_factor: [color.r, color.g, color.b, color.a],
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            alpha_cutoff: 0.0,
        }
    }
}

/// GPU resources for a material primitive type (bind group).
pub(crate) struct MaterialGpuResources {
    pub bind_group: wgpu::BindGroup,
    pub _buffer: Option<wgpu::Buffer>,
}

/// The uploaded bind group of every material in one scene.
///
/// Each material kind (face/line/point) has its own cache keyed by its typed id.
/// Callers reach the right cache through the typed accessors.
pub(crate) struct MaterialCache {
    face: GenCache<FaceMaterialId, MaterialGpuResources>,
    line: GenCache<LineMaterialId, MaterialGpuResources>,
    point: GenCache<PointMaterialId, MaterialGpuResources>,
}

impl MaterialCache {
    pub fn new() -> Self {
        Self {
            face: GenCache::new(),
            line: GenCache::new(),
            point: GenCache::new(),
        }
    }

    /// The cached bind group for a face material, if uploaded.
    pub fn face_material(&self, id: FaceMaterialId) -> Option<&MaterialGpuResources> {
        self.face.get(id)
    }

    /// The cached bind group for a line material, if uploaded.
    pub fn line_material(&self, id: LineMaterialId) -> Option<&MaterialGpuResources> {
        self.line.get(id)
    }

    /// The cached bind group for a point material, if uploaded.
    pub fn point_material(&self, id: PointMaterialId) -> Option<&MaterialGpuResources> {
        self.point.get(id)
    }

    /// The cached bind group for a batch's material, if uploaded.
    ///
    /// Dispatches on the typed [`BatchMaterial`] to the matching per-kind getter —
    /// the variant carries its own id kind, so there is no erased lookup.
    pub fn bind_group(&self, material: BatchMaterial) -> Option<&MaterialGpuResources> {
        match material {
            BatchMaterial::Face(id) => self.face_material(id),
            BatchMaterial::Line(id) => self.line_material(id),
            BatchMaterial::Point(id) => self.point_material(id),
        }
    }

    /// Drop all cached material bind groups (e.g. when the scene is replaced).
    pub fn clear(&mut self) {
        self.face.clear();
        self.line.clear();
        self.point.clear();
    }

    /// Evict the cached bind group of a removed material. No-op for
    /// non-material kinds.
    pub fn remove(&mut self, kind: ResourceKind, id: GenericId) {
        match kind {
            ResourceKind::FaceMaterial => {
                let _ = self.face.remove(id.cast());
            }
            ResourceKind::LineMaterial => {
                let _ = self.line.remove(id.cast());
            }
            ResourceKind::PointMaterial => {
                let _ = self.point.remove(id.cast());
            }
            _ => {}
        }
    }

    /// Upload any face/line/point materials whose generation has changed.
    ///
    /// `pipelines` supplies the group-2 bind-group layout each variant binds
    /// against. `textures` supplies already-uploaded texture GPU resources for
    /// PBR materials; the renderer prepares textures before calling this.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        pipelines: &mut MaterialPipelineCache,
        scene: &SceneData,
        textures: &GenCache<TextureId, GpuTexture>,
    ) -> Result<()> {
        let face_ids: Vec<FaceMaterialId> = scene
            .face_materials()
            .filter(|m| self.face.needs_upload(m.id, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in face_ids {
            let material = scene.get_face_material(id).unwrap();
            let generation = material.generation();
            // `props` (and thus texture presence) gates normal/metallic-roughness
            // to lit materials, matching the shader. IBL/depth-prepass don't
            // affect the group-2 layout, so any value works here.
            let cfg = SurfaceConfig::new(material.properties(), false, false);
            let resources = create_material(
                device,
                pipelines,
                cfg.textures(),
                MaterialUniform::from_face_material(material),
                |slot| match slot {
                    MaterialTextureSlot::BaseColor => material.base_color_texture(),
                    MaterialTextureSlot::Normal => material.normal_texture(),
                    MaterialTextureSlot::MetallicRoughness => material.metallic_roughness_texture(),
                },
                textures,
                "Face Material",
            )?;
            self.face.insert(id, resources, generation);
        }

        let line_ids: Vec<LineMaterialId> = scene
            .line_materials()
            .filter(|m| self.line.needs_upload(m.id, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in line_ids {
            let material = scene.get_line_material(id).unwrap();
            let cfg = SurfaceConfig::new(material.properties(), false, false);
            let resources = create_material(
                device,
                pipelines,
                cfg.textures(),
                MaterialUniform::from_line_material(material),
                |slot| match slot {
                    MaterialTextureSlot::BaseColor => material.base_color_texture(),
                    _ => None,
                },
                textures,
                "Line Material",
            )?;
            self.line.insert(id, resources, material.generation());
        }

        let point_ids: Vec<PointMaterialId> = scene
            .point_materials()
            .filter(|m| self.point.needs_upload(m.id, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in point_ids {
            let material = scene.get_point_material(id).unwrap();
            let cfg = SurfaceConfig::new(material.properties(), false, false);
            let resources = create_material(
                device,
                pipelines,
                cfg.textures(),
                MaterialUniform::from_point_material(material),
                |slot| match slot {
                    MaterialTextureSlot::BaseColor => material.base_color_texture(),
                    _ => None,
                },
                textures,
                "Point Material",
            )?;
            self.point.insert(id, resources, material.generation());
        }

        Ok(())
    }
}

/// Build a material bind group against the layout for `present`.
///
/// `resolve` maps each present channel to the texture id to bind; it is only
/// called for channels `present` marks (so it must return `Some` there). Each
/// channel's binding slots come from [`MaterialTextureSlot`], the same source
/// the layout uses.
fn create_material(
    device: &wgpu::Device,
    pipelines: &mut MaterialPipelineCache,
    present: TexturePresence,
    uniform: MaterialUniform,
    resolve: impl Fn(MaterialTextureSlot) -> Option<TextureId>,
    textures: &GenCache<TextureId, GpuTexture>,
    label: &str,
) -> Result<MaterialGpuResources> {
    let layout = pipelines.material_bind_group_layout(device, present).clone();

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }];
    for slot in present.slots() {
        let id = resolve(slot)
            .ok_or_else(|| anyhow!("{slot:?} texture marked present but unset"))?;
        let gpu = textures
            .get(id)
            .ok_or_else(|| anyhow!("{slot:?} texture {id} not found in GPU resources"))?;
        entries.push(wgpu::BindGroupEntry {
            binding: slot.texture_binding(),
            resource: wgpu::BindingResource::TextureView(&gpu.view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: slot.sampler_binding(),
            resource: wgpu::BindingResource::Sampler(&gpu.sampler),
        });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layout,
        entries: &entries,
    });

    Ok(MaterialGpuResources { bind_group, _buffer: Some(buffer) })
}
