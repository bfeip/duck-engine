//! Single-mode transform operators (translate, rotate, or scale).
//!
//! The interactive core — input bindings, axis-constraint cycling, gizmo
//! display and picking, and the mouse-delta → transform math — lives in
//! [`TransformDriver`] and [`TransformInteraction`]. What is being
//! transformed is a [`TransformTarget`]: [`NodeTransformTarget`] transforms
//! the selected scene nodes, and downstream crates plug in their own targets
//! without re-implementing the interaction.
//!
//! Each driver instance is locked to one [`TransformMode`] chosen at
//! construction; its gizmo handle set reflects that mode. Separate transform
//! operations are separate operators.

mod drag;
mod driver;
pub mod gizmo;
mod interaction;
mod node_target;

pub use driver::{TransformCaps, TransformDriver, TransformFrame, TransformTarget};
pub use interaction::{
    axis_from_constraint, AxisConstraint, TransformAction, TransformInteraction, TransformMode,
};
pub use node_target::NodeTransformTarget;

/// Operator for a single transform operation on the selected scene nodes.
pub type TransformOperator = TransformDriver<NodeTransformTarget>;

impl TransformDriver<NodeTransformTarget> {
    /// Creates a node transform operator locked to the given mode.
    pub fn new(mode: TransformMode) -> Self {
        Self::with_target(mode, NodeTransformTarget::new())
    }
}
