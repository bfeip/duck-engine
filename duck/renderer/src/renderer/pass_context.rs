use crate::abi;
use crate::render_core::{FrameFamily, GenCache, Gpu, Pass, Workflow};
use crate::scene::resource::{AlphaMode, MeshId};
use crate::scene::{Projection, SceneData, SceneProperties};

use super::batching::{BatchMaterial, DrawBatch, DrawData};
use super::mesh::MeshGpuResources;
use super::material_cache::MaterialCache;
use super::pass::PrimitiveFilter;
use super::pipeline::{MaterialPipelineCache, PipelineCacheKey};
use super::scene_bindings::SceneBindingRefs;
use super::surface_config::SurfaceConfig;

/// Frame family for the standard scene renderer.
///
/// A type-level tag that ties the core dispatch machinery
/// ([`RenderHost`](crate::render_core::RenderHost),
/// [`Workflow`], [`Pass`]) to [`SceneFrame`] as its per-frame data type.
/// Uninhabited because it is never constructed — it exists only to name
/// `SceneFrame<'_>` at the type level. See [`FrameFamily`] for why this
/// indirection is needed.
pub enum SceneFrames {}

impl FrameFamily for SceneFrames {
    type Frame<'a> = SceneFrame<'a>;
}

/// A pass over the standard scene frame.
///
/// Convenience alias for boxing: write `Box<ScenePass>` rather than spelling
/// out the core trait plus frame family. A custom pass implements
/// [`Pass<SceneFrames>`](crate::render_core::Pass).
pub type ScenePass = dyn Pass<SceneFrames>;

/// An editable pass list over the standard scene frame.
///
/// The stock ones are built by [`workflow::shaded`](super::workflow::shaded)
/// and [`workflow::hidden_line`](super::workflow::hidden_line); either can then
/// be edited in place rather than rebuilt.
pub type SceneWorkflow = Workflow<SceneFrames>;

/// Per-frame data for the standard scene renderer.
///
/// Built once per frame by the renderer from disjoint field borrows of itself,
/// then handed to the active workflow. Holds everything a scene pass reads —
/// the scene, the collected draw batches, the scene-level bind groups, the
/// scene's material bind groups, and the shared material pipeline cache.
///
/// Bind groups follow the standard shader ABI (see [`crate::abi`]): the renderer
/// fills them but passes choose whether and at which slot to bind them, so a
/// workflow that needs no lights or IBL simply ignores those fields.
///
/// `gpu` and `targets` are *not* fields here: the [`RenderHost`](crate::render_core::RenderHost)
/// lends them to `execute` as separate arguments. Keeping them out of the frame
/// is what lets the renderer hold `&mut host` while the frame borrows the
/// renderer's other fields.
pub struct SceneFrame<'a> {
    /// The scene being rendered, locked for the duration of the frame.
    pub scene: &'a SceneData,
    /// Collected, sorted, partitioned draw batches for this frame.
    pub draw: &'a DrawData,
    /// Uploaded mesh vertex/index buffers, keyed by mesh id.
    pub(crate) gpu_meshes: &'a GenCache<MeshId, MeshGpuResources>,
    /// The scene-level bind groups for this frame (camera, lights, IBL).
    pub bindings: SceneBindingRefs<'a>,
    /// Derived from `bindings.ibl` — `has_ibl` is true iff `bindings.ibl` is `Some`.
    pub scene_props: SceneProperties,
    /// This frame's camera projection. Screen-space passes that interpret the
    /// depth buffer need it: depth is reciprocal under perspective and linear
    /// under orthographic.
    pub projection: Projection,
    /// This scene's uploaded per-material bind groups.
    pub(crate) materials: &'a MaterialCache,
    /// The shared material pipeline cache, for pipelines built on demand.
    pub(crate) pipelines: &'a mut MaterialPipelineCache,
    /// The renderer's clear color for this frame.
    pub background_color: wgpu::Color,
}

/// How [`SceneFrame::draw_batches`] should draw a batch list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DrawOptions {
    /// Run a depth-only pre-pass for `Blend`-mode materials first, so their
    /// opaque portions establish correct depth occlusion before the main draw
    /// loop renders them with blending.
    pub depth_prepass: bool,
    /// Only batches of this primitive kind are drawn; the rest are left for
    /// another pass.
    pub filter: PrimitiveFilter,
}

impl Default for DrawOptions {
    fn default() -> Self {
        Self { depth_prepass: true, filter: PrimitiveFilter::All }
    }
}

impl SceneFrame<'_> {
    /// Draw a [`DrawBatch`] into `render_pass`.
    ///
    /// Looks up the mesh's GPU vertex/index buffers and issues the instanced draw
    /// call. Silently skips batches whose GPU resources haven't been uploaded yet.
    ///
    /// This binds no pipeline and no material — the caller's pipeline stays
    /// bound, which is what a pass drawing scene geometry through its own
    /// shader wants. For standard material shading use
    /// [`draw_batches`](Self::draw_batches).
    pub fn draw_batch(&self, gpu: &Gpu, render_pass: &mut wgpu::RenderPass<'_>, batch: &DrawBatch) {
        let Some(gpu_mesh) = self.gpu_meshes.get(batch.mesh_id) else { return };
        gpu_mesh.draw_instances(
            &gpu.device,
            render_pass,
            batch.primitive_type,
            &batch.instances,
            batch.index_count,
        );
    }

    /// Bind the scene-level bind groups shared by all geometry passes:
    /// - Group 0: camera (view/projection + eye position)
    /// - Group 1: lights
    /// - Group 3: IBL environment, when active
    ///
    /// See [`crate::abi`] for the slot assignments a conforming shader expects.
    pub fn bind_scene_groups(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_bind_group(abi::GROUP_CAMERA, self.bindings.camera, &[]);
        render_pass.set_bind_group(abi::GROUP_LIGHTS, self.bindings.lights, &[]);
        if let Some(ibl) = self.bindings.ibl {
            render_pass.set_bind_group(abi::GROUP_IBL, ibl, &[]);
        }
    }

    /// Draw a list of batches with standard material shading.
    ///
    /// For each batch this selects the surface pipeline matching its material
    /// and binds that material's group-2 bind group, switching pipelines only
    /// when the key changes. Batches whose GPU resources are not yet uploaded
    /// are skipped.
    ///
    /// Call [`bind_scene_groups`](Self::bind_scene_groups) first: the pipelines
    /// used here expect camera, lights and IBL already bound.
    pub fn draw_batches(
        &mut self,
        gpu: &Gpu,
        render_pass: &mut wgpu::RenderPass<'_>,
        batches: &[DrawBatch],
        options: DrawOptions,
    ) {
        let DrawOptions { depth_prepass, filter } = options;
        let gpu_meshes = self.gpu_meshes;
        let scene_props = self.scene_props.clone();
        let materials = self.materials;
        let pipelines = &mut *self.pipelines;

        if depth_prepass {
            // Depth pre-pass for transparent objects: render depth-only with alpha test
            // so opaque portions of blend materials establish correct depth occlusion.
            let mut prepass_pipeline_key: Option<PipelineCacheKey> = None;
            for batch in batches {
                if !filter.accepts(batch.primitive_type) {
                    continue;
                }

                let material_props = &batch.material_props;

                if material_props.alpha_mode != AlphaMode::Blend {
                    continue;
                }

                let Some(gpu_mesh) = gpu_meshes.get(batch.mesh_id) else {
                    continue;
                };

                // depth_prepass=true compiles in the alpha-test discard and masks
                // color writes; IBL is irrelevant for depth-only output (scene IBL
                // passed as false). Texture presence still matches the material so
                // its bind group stays compatible with this pipeline's layout.
                let pipeline_key = PipelineCacheKey {
                    surface: SurfaceConfig::new(material_props.clone(), false, true),
                    primitive_type: batch.primitive_type,
                };
                if prepass_pipeline_key.as_ref() != Some(&pipeline_key) {
                    let pipeline = pipelines.get_or_create(&gpu.device, pipeline_key.clone());
                    render_pass.set_pipeline(pipeline);
                    prepass_pipeline_key = Some(pipeline_key);
                }

                let Some(material_gpu) = materials.bind_group(batch.material) else {
                    continue;
                };
                render_pass.set_bind_group(abi::GROUP_MATERIAL, &material_gpu.bind_group, &[]);

                gpu_mesh.draw_instances(
                    &gpu.device,
                    render_pass,
                    batch.primitive_type,
                    &batch.instances,
                    batch.index_count,
                );
            }
        }

        // Main draw loop
        let mut current_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in batches {
            if !filter.accepts(batch.primitive_type) {
                continue;
            }

            let Some(gpu_mesh) = gpu_meshes.get(batch.mesh_id) else {
                continue;
            };

            let pipeline_key = PipelineCacheKey {
                surface: SurfaceConfig::new(
                    batch.material_props.clone(),
                    scene_props.has_ibl,
                    false,
                ),
                primitive_type: batch.primitive_type,
            };
            if current_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = pipelines.get_or_create(&gpu.device, pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                current_pipeline_key = Some(pipeline_key);
            }

            let Some(material_gpu) = materials.bind_group(batch.material) else {
                continue;
            };
            render_pass.set_bind_group(abi::GROUP_MATERIAL, &material_gpu.bind_group, &[]);

            gpu_mesh.draw_instances(
                &gpu.device,
                render_pass,
                batch.primitive_type,
                &batch.instances,
                batch.index_count,
            );
        }
    }

    /// This scene's group-2 bind group for `material`, or `None` when it has
    /// not been uploaded yet.
    #[must_use]
    pub fn material_bind_group(&self, material: BatchMaterial) -> Option<&wgpu::BindGroup> {
        self.materials.bind_group(material).map(|m| &m.bind_group)
    }

    /// The surface pipeline for `key`, compiling and caching it on first use.
    ///
    /// The cache is shared by every pass and every view at this target
    /// configuration, so a pass that draws the standard surface shader should
    /// go through here rather than build its own.
    pub fn surface_pipeline(
        &mut self,
        device: &wgpu::Device,
        key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        self.pipelines.get_or_create(device, key)
    }
}
