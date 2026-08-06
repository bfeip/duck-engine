use duck_engine_scene::resource::{AlphaMode, PrimitiveType};

use crate::DrawBatch;
use crate::abi;
use crate::render_core::{FrameTargets, Gpu, GpuTexture};
use crate::renderer::pipeline::PipelineCacheKey;
use crate::renderer::surface_config::SurfaceConfig;

use super::super::pass_context::{SceneFrame, SceneRenderPass};

/// Selects which primitive kinds a geometry pass draws, so faces and
/// lines/points can be split across separate passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PrimitiveFilter {
    All,
    Faces,
    LinesAndPoints,
}

impl PrimitiveFilter {
    const fn accepts(self, primitive_type: PrimitiveType) -> bool {
        match self {
            Self::All => true,
            Self::Faces => matches!(primitive_type, PrimitiveType::TriangleList),
            Self::LinesAndPoints => matches!(
                primitive_type,
                PrimitiveType::LineList | PrimitiveType::PointList
            ),
        }
    }
}

/// Bind the scene-level bind groups shared by all geometry passes:
/// - Group 0: Camera (view/proj + eye position)
/// - Group 1: Lights
/// - Group 3: IBL environment (when active)
pub(crate) fn bind_scene_groups<'p>(render_pass: &mut wgpu::RenderPass<'p>, frame: &SceneFrame<'p>) {
    render_pass.set_bind_group(abi::GROUP_CAMERA, frame.bindings.camera, &[]);
    render_pass.set_bind_group(abi::GROUP_LIGHTS, frame.bindings.lights, &[]);
    if let Some(ibl) = frame.bindings.ibl {
        render_pass.set_bind_group(abi::GROUP_IBL, ibl, &[]);
    }
}

/// Draw a list of batches through the scene geometry pipeline.
///
/// If `with_depth_prepass` is true, a depth-only pre-pass for `Blend`-mode materials
/// is executed first so their opaque portions establish correct depth occlusion before
/// the main draw loop renders them with blending.
///
/// `filter` restricts to one primitive kind; batches of other kinds are
/// left for another pass to draw.
pub(crate) fn draw_batches(
    gpu: &Gpu,
    render_pass: &mut wgpu::RenderPass<'_>,
    batches: &[DrawBatch],
    frame: &mut SceneFrame<'_>,
    with_depth_prepass: bool,
    filter: PrimitiveFilter,
) {
    let gpu_meshes = frame.gpu_meshes;
    let scene_props = frame.scene_props.clone();
    let materials = &mut *frame.materials;

    if with_depth_prepass {
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

            // Pipeline (mutable phase) before the material bind group (shared
            // phase): both borrow `materials`, and wgpu does not retain either
            // borrow past the set call, so they must not overlap.
            //
            // depth_prepass=true compiles in the alpha-test discard and masks
            // color writes; IBL is irrelevant for depth-only output (scene IBL
            // passed as false). Texture presence still matches the material so
            // its bind group stays compatible with this pipeline's layout.
            let pipeline_key = PipelineCacheKey {
                surface: SurfaceConfig::new(material_props.clone(), false, true),
                primitive_type: batch.primitive_type,
            };
            if prepass_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = materials.pipeline(&gpu.device, pipeline_key.clone());
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

        // Pipeline (mutable phase) before material bind group (shared phase); see
        // the prepass note above on the borrow ordering.
        let pipeline_key = PipelineCacheKey {
            surface: SurfaceConfig::new(batch.material_props.clone(), scene_props.has_ibl, false),
            primitive_type: batch.primitive_type,
        };
        if current_pipeline_key.as_ref() != Some(&pipeline_key) {
            let pipeline = materials.pipeline(&gpu.device, pipeline_key.clone());
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

/// Binds camera/lights/IBL, runs a depth pre-pass for `Blend`-mode materials,
/// then draws the batches this pass is responsible for.
/// 
/// Face passes always clear.

// Scene geometry is drawn in two of these: [`MainPass::faces`] clears the color
// and depth attachments and draws triangles, then [`MainPass::lines_and_points`]
// loads them and draws the rest. The split exists so the highlight outline mask
// can be built from a depth buffer holding faces only — lines write their true
// (unbiased) depth, which would otherwise punch holes in the mask.
//
// One consequence: transparent line batches no longer interleave back-to-front
// with transparent face batches, since `sort_batches_for_transparency` orders all
// `Blend` batches together regardless of primitive type. Transparent line
// materials are rare enough that this is an acceptable trade.
pub(crate) struct MainPass {
    filter: PrimitiveFilter,
    /// `true` clears color and depth; `false` loads both.
    clears: bool,
}

impl MainPass {
    /// Draws triangles, clearing color and depth first.
    pub(crate) const fn faces() -> Self {
        Self { filter: PrimitiveFilter::Faces, clears: true }
    }

    /// Draws lines and points on top of an already-populated color/depth buffer.
    pub(crate) const fn lines_and_points() -> Self {
        Self { filter: PrimitiveFilter::LinesAndPoints, clears: false }
    }
}

impl SceneRenderPass for MainPass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        // The clearing pass must always run, even with nothing to draw.
        self.clears || frame
            .draw
            .all_batches()
            .iter()
            .any(|b| self.filter.accepts(b.primitive_type))
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let (color_view, resolve_target) = targets.color_views(view);
        let (color_load, depth_load) = if self.clears {
            (wgpu::LoadOp::Clear(frame.background_color), wgpu::LoadOp::Clear(1.0))
        } else {
            (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if self.clears {
                "3D Scene Faces Pass"
            } else {
                "3D Scene Lines & Points Pass"
            }),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations { load: color_load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth_view(),
                depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }),
                stencil_ops: None, // No stencil buffer
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        bind_scene_groups(&mut render_pass, frame);
        let batches = frame.draw.all_batches();
        // The pre-pass runs in both halves: it filters by primitive type too, so
        // blend lines and points keep the depth pre-pass they get today.
        draw_batches(gpu, &mut render_pass, batches, frame, true, self.filter);
    }
}

/// Loads the existing color attachment, clears a separate depth buffer, and draws
/// always-on-top geometry so it depth-tests among itself but not against the scene.
/// Owns its own depth buffer so it can be independently resized.
pub(crate) struct OverlayPass {
    depth: GpuTexture,
}

impl OverlayPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        Self {
            depth: GpuTexture::depth(device, width, height, sample_count, "overlay_depth_texture"),
        }
    }
}

impl SceneRenderPass for OverlayPass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        frame.draw.has_overlay()
    }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        let (w, h) = targets.size();
        self.depth = GpuTexture::depth(&gpu.device, w, h, targets.sample_count(), "overlay_depth_texture");
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let (color_view, resolve_target) = targets.color_views(view);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        bind_scene_groups(&mut render_pass, frame);
        let batches = frame.draw.overlay_batches();
        draw_batches(gpu, &mut render_pass, batches, frame, false, PrimitiveFilter::All);
    }
}
