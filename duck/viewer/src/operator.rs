mod navigation;
mod selection;
mod transform;

pub use navigation::{NavigationMode, NavigationOperator};
pub use selection::{SelectionKinds, SelectionMode, SelectionOperator};
pub use transform::{
    axis_from_constraint, AxisConstraint, TransformAction, TransformAnnotations,
    TransformInteraction, TransformMode, TransformOperator,
};

use crate::event::{Event, EventContext};

/// Operators encapsulate interaction logic and own their own state.
///
/// Each operator receives every dispatched event through [`dispatch`](Operator::dispatch)
/// and returns `true` to stop propagation or `false` to let subsequent operators handle it.
/// Operators are stored in the [`EventDispatcher`](crate::event::EventDispatcher) in priority
/// order; those inserted earlier receive events first.
///
/// Operators are registered via [`Arc<Mutex<T>>`](std::sync::Arc) so callers can retain a
/// typed handle to read or mutate operator state after registration without going through
/// the dispatcher.
pub trait Operator: 'static {
    /// Called for every dispatched event.
    ///
    /// Return `true` to consume the event and stop further propagation,
    /// or `false` to allow subsequent operators to handle it.
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool;

    /// Human-readable name for this operator (used for debugging and UI display).
    fn name(&self) -> &str;
}
