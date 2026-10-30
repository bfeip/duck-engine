use std::sync::Mutex;

use crate::operator::Operator;
use crate::ScreenOverlay;

mod app;
mod context;
mod device;
mod dispatcher;

pub use app::AppEvent;
pub use context::EventContext;
pub use device::{DeviceEvent, DeviceEventKind};
pub use dispatcher::EventDispatcher;

/// Bridges `Arc<Mutex<T>>` into the dispatcher's type-erased storage.
///
/// Implemented blanket-wise for `Mutex<T: Operator>`, allowing `Arc<Mutex<T>>` to coerce
/// to `Arc<dyn ArcOperator>`. Each method but `as_ptr` locks the operator for the call.
pub(crate) trait ArcOperator {
    /// [`Operator::dispatch`]: handles `event`, returning `true` to stop its
    /// propagation.
    fn dispatch(&self, event: &Event, ctx: &mut EventContext) -> bool;
    /// [`Operator::name`], owned so it outlives the lock.
    fn name(&self) -> String;
    /// [`Operator::overlay`]: adds the operator's screen-space feedback to
    /// `overlay`.
    fn overlay(&self, overlay: &mut ScreenOverlay);
    /// The address of the `Mutex<T>` allocation, identifying the operator for
    /// pointer-equality lookups.
    fn as_ptr(&self) -> *const ();
}

impl<T: Operator> ArcOperator for Mutex<T> {
    fn dispatch(&self, event: &Event, ctx: &mut EventContext) -> bool {
        self.lock().unwrap().dispatch(event, ctx)
    }
    fn name(&self) -> String {
        self.lock().unwrap().name().to_string()
    }
    fn overlay(&self, overlay: &mut ScreenOverlay) {
        self.lock().unwrap().overlay(overlay)
    }
    fn as_ptr(&self) -> *const () {
        self as *const Mutex<T> as *const ()
    }
}

/// A single event flowing through the [`EventDispatcher`].
pub enum Event {
    /// A low-level input/device event.
    Device(DeviceEvent),
    /// A high-level semantic event emitted by an operator.
    App(AppEvent),
}

impl From<DeviceEvent> for Event {
    fn from(e: DeviceEvent) -> Self {
        Event::Device(e)
    }
}

impl From<AppEvent> for Event {
    fn from(e: AppEvent) -> Self {
        Event::App(e)
    }
}
