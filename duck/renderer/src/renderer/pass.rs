//! The built-in render passes, and the ids the stock workflows register them
//! under.
//!
//! Every pass here implements [`Pass<SceneFrames>`](crate::render_core::Pass)
//! and is constructed from a [`PassBuilder`](super::PassBuilder), exactly as a
//! user-written pass is — so any of them can be inserted into, removed from, or
//! swapped out of a [`SceneWorkflow`](super::SceneWorkflow).

mod flat_color;
mod main_pass;
mod outline_pass;
mod silhouette;
mod sub_geom_highlight;

pub use flat_color::{FlatColorPass, FlatColorPassDesc};
pub use main_pass::{MainPass, OverlayPass, PrimitiveFilter};
pub use outline_pass::{OutlineCompositePass, OutlineMaskPass};
pub use silhouette::SilhouetteEdgesPass;
pub use sub_geom_highlight::SubGeomHighlightPass;

/// The names the stock workflows register their passes under.
///
/// Use these to address a built-in pass: insert next to one, swap one out, or
/// reach in and retune it via
/// [`Workflow::pass_mut`](crate::render_core::Workflow::pass_mut).
pub mod ids {
    use crate::render_core::PassId;

    // Shaded workflow.
    pub const FACES: PassId = PassId("faces");
    pub const OUTLINE_MASK: PassId = PassId("outline_mask");
    pub const LINES_AND_POINTS: PassId = PassId("lines_and_points");
    pub const OUTLINE_COMPOSITE: PassId = PassId("outline_composite");
    pub const OVERLAY: PassId = PassId("overlay");
    pub const SUB_GEOM_HIGHLIGHT: PassId = PassId("sub_geom_highlight");

    // Hidden-line workflow.
    pub const SOLID: PassId = PassId("solid");
    pub const SILHOUETTE: PassId = PassId("silhouette");
    pub const OCCLUDED_LINES: PassId = PassId("occluded_lines");
    pub const VISIBLE_LINES: PassId = PassId("visible_lines");
}

/// Names of the auxiliary attachments the built-in passes share.
pub mod aux {
    /// Two-channel highlight coverage: red is the primary tier, green the
    /// secondary. Written by [`OutlineMaskPass`](super::OutlineMaskPass), read
    /// by [`OutlineCompositePass`](super::OutlineCompositePass).
    pub const OUTLINE_MASK: &str = "outline_mask";

    /// The always-on-top depth buffer, kept separate from the scene's so
    /// overlay geometry depth-tests among itself but not against the scene.
    pub const OVERLAY_DEPTH: &str = "overlay_depth";
}
