use crate::selection::SelectionItem;

/// A semantic, high-level event emitted by an operator to signal that *something
/// happened* (as opposed to [`DeviceEvent`](super::DeviceEvent), which describes raw input).
///
/// Downstream crates that cannot edit this enum carry their own event types
/// through [`AppEvent::Custom`] and recover them with [`std::any::Any::downcast_ref`].
pub enum AppEvent {
    /// The selection changed. Carries the sets before and after so consumers can tell
    /// what happened (added to, replaced, or cleared).
    //
    // TODO: emitted by the selection *operator*, not the selection manager, so selection
    // changes made directly through the manager elsewhere go unreported. A future
    // MPSC-based emission channel could let the manager itself emit.
    Selection {
        /// Selected items before the change, in selection order.
        previous: Vec<SelectionItem>,
        /// Selected items after the change, in selection order.
        current: Vec<SelectionItem>,
    },
    /// A camera navigation interaction (orbit/pan/zoom) began.
    CameraInteractionStart,
    /// A camera navigation interaction ended.
    CameraInteractionEnd,
    /// A downstream-defined event.
    Custom(Box<dyn std::any::Any + Send>),
}
