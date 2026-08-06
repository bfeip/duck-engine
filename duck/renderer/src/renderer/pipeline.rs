use std::collections::HashMap;

use crate::render_core::{GpuTexture, PipelineCache};
use crate::scene::resource::{AlphaMode, PrimitiveType};
use crate::shaders::ShaderGenerator;

use super::bind_group_layouts::BindGroupLayouts;
use super::mesh::{instance_buffer_layout, vertex_buffer_layout};
use super::surface_config::{SurfaceConfig, TexturePresence};

/// Cached render pipelines keyed by surface variant + primitive topology.
///
/// Owns the [`MaterialLayoutCache`] (group-2 bind-group layouts and pipeline
/// layouts, derived per [`SurfaceConfig`](super::surface_config::SurfaceConfig))
/// and the WESL [`ShaderGenerator`]. Technique-specific pipelines with a fixed
/// configuration (outline, silhouette, hidden-line solid) are owned directly by
/// their respective passes.
pub struct MaterialPipelineCache {
    cache: PipelineCache<PipelineCacheKey>,
    layouts: MaterialLayoutCache,
    shader_generator: ShaderGenerator,
    sample_count: u32,
    surface_format: wgpu::TextureFormat,
}

impl MaterialPipelineCache {
    pub(super) fn new(
        layouts: MaterialLayoutCache,
        shader_generator: ShaderGenerator,
        sample_count: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            cache: PipelineCache::new(),
            layouts,
            shader_generator,
            sample_count,
            surface_format,
        }
    }

    pub(super) fn shader_generator_mut(&mut self) -> &mut ShaderGenerator {
        &mut self.shader_generator
    }

    /// The group-2 bind-group layout a material with the given textures binds
    /// against. Used when building per-material bind groups so layout and shader
    /// stay derived from the same [`TexturePresence`](super::surface_config::TexturePresence).
    pub(super) fn material_bind_group_layout(
        &mut self,
        device: &wgpu::Device,
        presence: super::surface_config::TexturePresence,
    ) -> &wgpu::BindGroupLayout {
        self.layouts.bind_group_layout(device, presence)
    }

    /// Discard all cached pipelines.
    ///
    /// Call this when `sample_count` or `surface_format` changes so pipelines
    /// are recreated with the new parameters on the next frame.
    #[allow(dead_code)]
    pub(super) fn invalidate(&mut self) {
        self.cache.invalidate();
    }

    pub(super) fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        cache_key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        let cfg = cache_key.surface.clone();
        let primitive_type = cache_key.primitive_type;
        self.cache.get_or_create(cache_key, || {
            // Layout and shader both derive from `cfg`, so they always agree on
            // which bindings exist. Clone the layout out so its borrow ends
            // before we borrow the shader generator (both fields of `self`).
            let pipeline_layout = self
                .layouts
                .pipeline_layout(device, cfg.textures(), cfg.has_ibl())
                .clone();

            let label = if cfg.lit() {
                if cfg.has_ibl() { "Surface Lit IBL Shader" } else { "Surface Lit Shader" }
            } else {
                "Surface Unlit Shader"
            };
            let shader = self
                .shader_generator
                .generate_surface_shader(device, &cfg.features(), label)
                .expect("Failed to generate surface shader");

            let topology = match primitive_type {
                PrimitiveType::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                PrimitiveType::LineList => wgpu::PrimitiveTopology::LineList,
                PrimitiveType::PointList => wgpu::PrimitiveTopology::PointList,
            };

            // Depth prepass: write depth only, no color output
            let (color_write_mask, depth_write_enabled) = if cfg.depth_prepass {
                (wgpu::ColorWrites::empty(), true)
            } else {
                (
                    wgpu::ColorWrites::ALL,
                    cfg.props.alpha_mode != AlphaMode::Blend,
                )
            };

            // Lines/points use LessEqual so they render correctly on coplanar triangles
            // (drawn after triangles, same depth buffer value). Blend materials also use
            // LessEqual so the main pass can render at the depth the prepass established.
            let is_blend = cfg.props.alpha_mode == AlphaMode::Blend;
            let is_non_triangle =
                matches!(primitive_type, PrimitiveType::LineList | PrimitiveType::PointList);
            let depth_compare = if !cfg.depth_prepass && (is_blend || is_non_triangle) {
                wgpu::CompareFunction::LessEqual
            } else {
                wgpu::CompareFunction::Less
            };

            // Push triangle faces slightly away from the camera so edge lines on curved
            // surfaces remain visible. Tessellated chords sit above the true curve at
            // chord midpoints, so without a bias the face depth genuinely wins over the
            // line. slope_scale accounts for grazing angles where chord error is worst.
            // clamp=0 avoids the DEPTH_BIAS_CLAMP feature requirement.
            let depth_bias = if primitive_type == PrimitiveType::TriangleList {
                wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                }
            } else {
                wgpu::DepthBiasState::default()
            };

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if cfg.depth_prepass {
                    "Depth Prepass Pipeline"
                } else {
                    "Render Pipeline"
                }),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.surface_format,
                        blend: Some(match cfg.props.alpha_mode {
                            AlphaMode::Blend => wgpu::BlendState::ALPHA_BLENDING,
                            _ => wgpu::BlendState::REPLACE,
                        }),
                        write_mask: color_write_mask,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Only cull for single-sided triangles, not for lines, points, or double-sided materials
                    cull_mode: if topology == wgpu::PrimitiveTopology::TriangleList
                        && !cfg.double_sided()
                    {
                        Some(wgpu::Face::Back)
                    } else {
                        None
                    },
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: GpuTexture::DEPTH_FORMAT,
                    depth_write_enabled,
                    depth_compare,
                    stencil: wgpu::StencilState::default(),
                    bias: depth_bias,
                }),
                multisample: wgpu::MultisampleState {
                    count: self.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        })
    }
}

/// Lazily-built, cached layouts for surface materials.
///
/// Both the group-2 material bind-group layout and the full pipeline layout are
/// derived from a material's [`TexturePresence`] (and, for the pipeline layout,
/// whether IBL is active) — the same value that drives the shader's `@if`
/// bindings. Deriving layout and shader from one source is what guarantees they
/// always agree, so a variant binds exactly the textures it declares.
pub(crate) struct MaterialLayoutCache {
    camera: wgpu::BindGroupLayout,
    light: wgpu::BindGroupLayout,
    ibl: wgpu::BindGroupLayout,
    bind_group_layouts: HashMap<TexturePresence, wgpu::BindGroupLayout>,
    pipeline_layouts: HashMap<(TexturePresence, bool), wgpu::PipelineLayout>,
}

impl MaterialLayoutCache {
    pub fn new(layouts: &BindGroupLayouts) -> MaterialLayoutCache {
        MaterialLayoutCache {
            camera: layouts.camera.clone(),
            light: layouts.light.clone(),
            ibl: layouts.ibl.clone(),
            bind_group_layouts: HashMap::new(),
            pipeline_layouts: HashMap::new(),
        }
    }

    /// The group-2 bind-group layout for a material with the given textures:
    /// the params uniform at binding 0 plus a texture+sampler pair at the fixed
    /// slot of each present texture (absent slots are simply omitted).
    pub fn bind_group_layout(
        &mut self,
        device: &wgpu::Device,
        presence: TexturePresence,
    ) -> &wgpu::BindGroupLayout {
        self.bind_group_layouts
            .entry(presence)
            .or_insert_with(|| build_material_bind_group_layout(device, presence))
    }

    /// The pipeline layout for a variant: `[camera, lights, material]`, plus the
    /// IBL group when `has_ibl`.
    pub fn pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        presence: TexturePresence,
        has_ibl: bool,
    ) -> &wgpu::PipelineLayout {
        // Materialize the bind-group layout first so the borrow ends before we
        // touch `pipeline_layouts`; the clone is a cheap Arc bump.
        let material = self.bind_group_layout(device, presence).clone();
        let camera = &self.camera;
        let light = &self.light;
        let ibl = &self.ibl;
        self.pipeline_layouts
            .entry((presence, has_ibl))
            .or_insert_with(|| {
                let mut bgls: Vec<&wgpu::BindGroupLayout> = vec![camera, light, &material];
                if has_ibl {
                    bgls.push(ibl);
                }
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Surface Material Pipeline Layout"),
                    bind_group_layouts: &bgls,
                    push_constant_ranges: &[],
                })
            })
    }
}

/// Build the group-2 bind-group layout for the given texture presence.
///
/// Binding 0 is always the material params uniform; each present channel adds a
/// texture+sampler pair at its fixed slot (from [`MaterialTextureSlot`](super::surface_config::MaterialTextureSlot)).
/// Binding numbers are explicit, so gaps left by absent textures are legal and
/// keep every texture at a stable slot.
fn build_material_bind_group_layout(
    device: &wgpu::Device,
    presence: TexturePresence,
) -> wgpu::BindGroupLayout {
    fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        }
    }
    fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }

    // Binding 0: material params uniform (always present).
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for slot in presence.slots() {
        entries.push(texture_entry(slot.texture_binding()));
        entries.push(sampler_entry(slot.sampler_binding()));
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surface Material Bind Group Layout"),
        entries: &entries,
    })
}

/// Cache key for render pipelines: the surface variant plus its primitive topology.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineCacheKey {
    pub surface: SurfaceConfig,
    pub primitive_type: PrimitiveType,
}
