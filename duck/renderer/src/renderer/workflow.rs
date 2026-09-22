//! The stock rendering workflows.
//!
//! Each is a function returning a [`SceneWorkflow`] — an ordered pass list, not
//! a distinct type — so a caller can start from one and adjust it rather than
//! rebuild it:
//!
//! ```ignore
//! let mut workflow = workflow::shaded(&mut renderer.pass_builder(&mut ctx));
//! workflow.insert_after(pass::ids::OVERLAY, PassId("grid"), GridPass::new(b))?;
//! renderer.set_workflow(workflow);
//! ```

use crate::render_core::Workflow;
use crate::scene::common::RgbaColor;
use crate::scene::resource::PrimitiveType;

use super::PassBuilder;
use super::pass::{
    FlatColorPass, FlatColorPassDesc, MainPass, OutlineCompositePass, OutlineMaskPass,
    OverlayPass, SilhouetteEdgesPass, SubGeomHighlightPass, ids,
};
use super::pass_context::SceneWorkflow;

/// The default shaded rendering workflow.
///
/// Runs the standard pass sequence: scene faces, the highlight outline mask,
/// scene lines and points, the outline composite, overlay (always-on-top)
/// geometry, and sub-geometry highlights.
///
/// The order carries two constraints worth preserving when editing it: the
/// outline mask sits between the two geometry passes so it depth-tests against
/// faces only, and the composite runs after lines and points so nothing cuts
/// the outline band.
#[must_use]
pub fn shaded(builder: &mut PassBuilder<'_>) -> SceneWorkflow {
    let outline_mask = OutlineMaskPass::new(builder);
    let outline_composite = OutlineCompositePass::new(builder);
    let sub_geom = SubGeomHighlightPass::new(builder);

    Workflow::new("Shaded")
        .with(ids::FACES, MainPass::faces())
        // Between the two geometry passes: the mask depth-tests against faces only.
        .with(ids::OUTLINE_MASK, outline_mask)
        .with(ids::LINES_AND_POINTS, MainPass::lines_and_points())
        // After lines and points, so nothing cuts the outline band.
        .with(ids::OUTLINE_COMPOSITE, outline_composite)
        .with(ids::OVERLAY, OverlayPass::new())
        // Sub-geometry highlights draw on top of the node outlines.
        .with(ids::SUB_GEOM_HIGHLIGHT, sub_geom)
}

/// Color configuration for [`hidden_line`].
#[derive(Clone, Debug)]
pub struct HiddenLineConfig {
    /// Background / face color (also used as the clear color).
    pub face_color: RgbaColor,
    /// Color of lines that are visible (in front of solid geometry).
    pub visible_line_color: RgbaColor,
    /// Color of lines that are occluded (behind solid geometry).
    pub hidden_line_color: RgbaColor,
}

impl Default for HiddenLineConfig {
    fn default() -> Self {
        Self {
            face_color: RgbaColor::WHITE,
            visible_line_color: RgbaColor::BLACK,
            hidden_line_color: RgbaColor { r: 0.6, g: 0.6, b: 0.6, a: 1.0 },
        }
    }
}

/// Hidden-line rendering workflow.
///
/// Renders scene geometry as solid faces with silhouette edges detected from
/// the depth buffer, plus explicit `LineList` primitives in two flat colors:
/// one for occluded lines and one for visible lines.
///
/// Pass sequence:
/// 1. [`ids::SOLID`] — clear to face color, render all triangles, write depth.
/// 2. [`ids::SILHOUETTE`] — fullscreen depth-discontinuity edge detection.
///    Skips itself on backends that cannot sample depth textures.
/// 3. [`ids::OCCLUDED_LINES`] — hidden line color where depth compare is `Greater`.
/// 4. [`ids::VISIBLE_LINES`] — visible line color where depth compare is `LessEqual`.
///
/// `config` seeds the colors; they can be changed later without rebuilding the
/// workflow, since each line pass is a [`FlatColorPass`] addressed by its id:
///
/// ```ignore
/// workflow.pass_mut::<FlatColorPass>(ids::VISIBLE_LINES)
///     .unwrap()
///     .set_color(queue, color);
/// ```
#[must_use]
pub fn hidden_line(builder: &mut PassBuilder<'_>, config: HiddenLineConfig) -> SceneWorkflow {
    let solid = FlatColorPass::new(
        builder,
        FlatColorPassDesc {
            label: "Hidden Line Solid",
            cull_mode: Some(wgpu::Face::Back),
            depth_compare: wgpu::CompareFunction::Less,
            depth_write: true,
            // Push faces slightly away so coplanar edges pass depth test.
            depth_bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            clear_color: Some(crate::rgba_to_wgpu_color(config.face_color)),
            primitive_filter: PrimitiveType::TriangleList,
            color: config.face_color,
        },
    );
    let silhouette = SilhouetteEdgesPass::new(builder);

    let mut line_pass = |label, depth_compare, color| {
        FlatColorPass::new(
            builder,
            FlatColorPassDesc {
                label,
                cull_mode: None,
                depth_compare,
                depth_write: false,
                depth_bias: wgpu::DepthBiasState::default(),
                clear_color: None,
                primitive_filter: PrimitiveType::LineList,
                color,
            },
        )
    };
    let occluded = line_pass(
        "Hidden Line Occluded",
        wgpu::CompareFunction::Greater,
        config.hidden_line_color,
    );
    let visible = line_pass(
        "Hidden Line Visible",
        wgpu::CompareFunction::LessEqual,
        config.visible_line_color,
    );

    Workflow::new("Hidden Line")
        .with(ids::SOLID, solid)
        .with(ids::SILHOUETTE, silhouette)
        .with(ids::OCCLUDED_LINES, occluded)
        .with(ids::VISIBLE_LINES, visible)
}
