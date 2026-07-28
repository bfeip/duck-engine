//! Trait for providing highlight information to the renderer.
//!
//! This trait decouples the renderer from any specific highlight implementation,
//! allowing it to query which nodes are highlighted for outline rendering.

use crate::scene::{NodeId, SubGeometryElement, SubGeometryKind};

/// Configuration for highlight rendering: whole-node outlines and sub-geometry
/// (face overlay / edge & point recolor) highlights share these settings.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Color of the highlight (RGBA, 0.0-1.0). Used for the node outline, the
    /// solid edge/point recolor, and the face overlay (whose alpha is replaced
    /// by `face_alpha`).
    pub color: [f32; 4],
    /// Width of the node outline in pixels (screen-space).
    pub width_pixels: f32,
    /// Alpha (0.0-1.0) of the transparent face-highlight overlay.
    pub face_alpha: f32,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            color: [1.0, 0.6, 0.0, 1.0], // Orange
            width_pixels: 2.0,
            face_alpha: 0.4,
        }
    }
}

/// Trait for querying highlight state during rendering.
///
/// The renderer uses this trait to determine which nodes are outlined as whole
/// nodes and which sub-geometry (faces/edges/points) is highlighted, without
/// depending on any specific highlight management implementation.
///
/// Whole-node outlines and sub-geometry highlights are kept distinct: a node is
/// outlined only when selected *as a whole node*. Selecting an individual face
/// or edge highlights just that element and does not outline the node.
pub trait HighlightQuery {
    /// Returns true if there are no highlights.
    fn is_empty(&self) -> bool;

    /// Returns all nodes selected as a whole node (and therefore outlined).
    fn outlined_nodes(&self) -> Vec<NodeId>;

    /// Returns the primary outlined node — the primary selection iff it is a
    /// whole-node selection. `None` if the primary is sub-geometry or there is
    /// no selection. Used to partition primary vs secondary outlines.
    fn primary_outlined_node(&self) -> Option<NodeId>;

    /// Returns the indices of sub-geometry of `kind` highlighted on `node_id`, if any.
    fn highlighted_for_node(&self, node_id: NodeId, kind: SubGeometryKind) -> Vec<u32>;

    /// Returns all nodes that have at least one element of `kind` highlighted.
    fn nodes_with_highlighted(&self, kind: SubGeometryKind) -> Vec<NodeId>;

    /// Returns the node and element of the primary selection iff it is sub-geometry.
    /// Used to color the primary sub-element with the primary tier.
    fn primary_sub_geometry(&self) -> Option<(NodeId, SubGeometryElement)>;

    /// Returns the highlight configuration for the primary selection.
    fn highlight_config(&self) -> HighlightConfig;

    /// Returns the highlight config for secondary (non-primary) highlights,
    /// or `None` if there are no secondary highlights.
    fn secondary_highlight_config(&self) -> Option<HighlightConfig>;
}
