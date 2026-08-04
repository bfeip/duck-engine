use crate::render_core::{FrameFamily, FrameTargets, GenCache, Gpu};
use crate::scene::{MeshId, SceneData, SceneProperties};

use super::batching::{DrawBatch, DrawData};
use super::mesh::MeshGpuResources;
use super::material_system::MaterialSystem;
use super::scene_bindings::SceneBindingRefs;

/// Frame family for the standard scene renderer.
///
/// A type-level tag that ties the core dispatch machinery
/// ([`RenderHost`](crate::render_core::RenderHost),
/// [`RenderWorkflow`](crate::render_core::RenderWorkflow)) to [`SceneFrame`] as
/// its per-frame data type. Uninhabited because it is never constructed — it
/// exists only to name `SceneFrame<'_>` at the type level. See [`FrameFamily`]
/// for why this indirection is needed.
pub enum SceneFrames {}

impl FrameFamily for SceneFrames {
    type Frame<'a> = SceneFrame<'a>;
}

/// A workflow over the standard scene frame.
///
/// Convenience alias so consumers write `Box<dyn SceneWorkflow>` rather than
/// spelling out the core trait + frame family. Implement
/// [`RenderWorkflow<SceneFrames>`](crate::render_core::RenderWorkflow) for a
/// custom workflow; the blanket coercion to this trait object is automatic.
pub type SceneWorkflow = dyn crate::render_core::RenderWorkflow<SceneFrames>;

/// Per-frame data for the standard scene renderer.
///
/// Built once per frame by the renderer from disjoint field borrows of itself,
/// then handed to the active workflow. Holds everything a scene pass reads —
/// the scene, the collected draw batches, the scene-level bind groups, and the
/// material pipeline cache.
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
    pub scene: &'a SceneData,
    /// Collected, sorted, partitioned draw batches for this frame.
    pub draw: &'a DrawData,
    /// Uploaded mesh vertex/index buffers, keyed by mesh id.
    pub(crate) gpu_meshes: &'a GenCache<MeshId, MeshGpuResources>,
    /// The scene-level bind groups for this frame (camera, lights, IBL).
    pub bindings: SceneBindingRefs<'a>,
    /// Derived from `bindings.ibl` — `has_ibl` is true iff `bindings.ibl` is `Some`.
    pub scene_props: SceneProperties,
    /// Material subsystem: pipelines, shaders, and per-material bind groups.
    pub(crate) materials: &'a mut MaterialSystem,
    pub background_color: wgpu::Color,
}

impl SceneFrame<'_> {
    /// Draw a [`DrawBatch`] into `render_pass`.
    ///
    /// Looks up the mesh's GPU vertex/index buffers and issues the instanced draw
    /// call. Silently skips batches whose GPU resources haven't been uploaded yet.
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
}

/// Extension point for user-defined render passes.
///
/// Each pass receives the [`Gpu`] handles, the shared [`FrameTargets`]
/// (depth/MSAA attachments), and a mutable [`SceneFrame`] for all per-frame
/// scene state including the material pipeline cache.
///
/// Passes may be stateless (zero-size structs) or stateful (holding owned GPU
/// resources such as textures or pipelines). Stateful passes implement
/// [`resize`](Self::resize) to recreate size-dependent resources when the
/// viewport changes.
pub trait SceneRenderPass {
    fn is_active(&self, _frame: &SceneFrame<'_>) -> bool {
        true
    }

    /// Called after a viewport resize. Passes that own size-dependent resources
    /// (textures, bind groups, or pipelines with baked sample counts) should
    /// recreate them here, reading the new size/sample count from `targets`.
    /// The default is a no-op.
    fn resize(&mut self, _gpu: &Gpu, _targets: &FrameTargets) {}

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    );
}
