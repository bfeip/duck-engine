use duck_engine_scene::resource::PrimitiveType;

use crate::render_core::{
    AuxKind, AuxTarget, FrameTargets, Gpu, Pass, TargetFeatures,
};

use super::super::pass_context::{DrawOptions, SceneFrame, SceneFrames};

/// Selects which primitive kinds a geometry pass draws, so faces and
/// lines/points can be split across separate passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrimitiveFilter {
    All,
    Faces,
    LinesAndPoints,
}

impl PrimitiveFilter {
    pub(crate) const fn accepts(self, primitive_type: PrimitiveType) -> bool {
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

/// Draws scene geometry: binds camera/lights/IBL, runs a depth pre-pass for
/// `Blend`-mode materials, then draws the batches this pass is responsible for.
///
/// The stock shaded workflow uses two of these: [`MainPass::faces`] clears the
/// color and depth attachments and draws triangles, then
/// [`MainPass::lines_and_points`] loads them and draws the rest. The split
/// exists so the highlight outline mask can be built from a depth buffer
/// holding faces only — lines write their true (unbiased) depth, which would
/// otherwise punch holes in the mask. Use [`MainPass::all`] where no mask needs
/// that separation.
//
// One consequence of the split: transparent line batches no longer interleave
// back-to-front with transparent face batches, since
// `sort_batches_for_transparency` orders all `Blend` batches together regardless
// of primitive type. Transparent line materials are rare enough that this is an
// acceptable trade.
pub struct MainPass {
    filter: PrimitiveFilter,
    /// `true` clears color and depth; `false` loads both.
    clears: bool,
}

impl MainPass {
    /// Draws triangles, clearing color and depth first.
    #[must_use]
    pub const fn faces() -> Self {
        Self { filter: PrimitiveFilter::Faces, clears: true }
    }

    /// Draws lines and points on top of an already-populated color/depth buffer.
    #[must_use]
    pub const fn lines_and_points() -> Self {
        Self { filter: PrimitiveFilter::LinesAndPoints, clears: false }
    }

    /// Draws every primitive kind in one pass, clearing first.
    ///
    /// The stock shaded workflow splits this in two so the outline mask sees a
    /// faces-only depth buffer; a workflow with no mask can use this instead.
    #[must_use]
    pub const fn all() -> Self {
        Self { filter: PrimitiveFilter::All, clears: true }
    }
}

impl Pass<SceneFrames> for MainPass {
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::depth()
    }

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

        frame.bind_scene_groups(&mut render_pass);
        let batches = frame.draw.all_batches();
        // The pre-pass runs in both halves: it filters by primitive type too, so
        // blend lines and points keep the depth pre-pass they get today.
        frame.draw_batches(
            gpu,
            &mut render_pass,
            batches,
            DrawOptions { depth_prepass: true, filter: self.filter },
        );
    }
}

/// Loads the existing color attachment, clears a separate depth buffer, and draws
/// always-on-top geometry so it depth-tests among itself but not against the scene.
///
/// The separate depth buffer is the auxiliary attachment
/// [`aux::OVERLAY_DEPTH`](super::aux::OVERLAY_DEPTH), so this pass holds no
/// resources of its own and needs no `resize`.
pub struct OverlayPass;

impl OverlayPass {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for OverlayPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass<SceneFrames> for OverlayPass {
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::none().with_aux(
            AuxTarget::new(super::aux::OVERLAY_DEPTH, AuxKind::Depth).with_multisampled(),
        )
    }

    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        frame.draw.has_overlay()
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
        let Some(depth) = targets.aux(super::aux::OVERLAY_DEPTH) else {
            return;
        };

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
                view: depth.view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        frame.bind_scene_groups(&mut render_pass);
        let batches = frame.draw.overlay_batches();
        frame.draw_batches(
            gpu,
            &mut render_pass,
            batches,
            DrawOptions { depth_prepass: false, filter: PrimitiveFilter::All },
        );
    }
}
