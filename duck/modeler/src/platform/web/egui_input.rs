//! Builds [`egui::RawInput`] from the platform-neutral viewer event stream.
//!
//! This is the `egui_winit::State` replacement for the web. It needs no
//! DOM-specific input of its own: modifier state is tracked from the modifier
//! key events the viewer already reports, and typed text from
//! [`Key::Character`].

use std::time::Instant;

use duck_engine_viewer::event::{DeviceEvent, Event};
use duck_engine_viewer::input::{ElementState, Key, MouseButton, MouseScrollDelta, NamedKey};

pub(crate) struct EguiInput {
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
    /// Last known pointer position, in points. egui wants a position on every
    /// button event, but the viewer reports position and buttons separately.
    pointer: egui::Pos2,
    start: Instant,
}

impl EguiInput {
    pub(crate) fn new() -> Self {
        Self {
            events: Vec::new(),
            modifiers: egui::Modifiers::default(),
            pointer: egui::Pos2::ZERO,
            start: Instant::now(),
        }
    }

    /// Accumulate one viewer event. Call for every event before routing it on.
    pub(crate) fn on_event(&mut self, event: &Event) {
        let Event::Device(event) = event else { return };
        match event {
            DeviceEvent::CursorMoved { position } => {
                self.pointer = egui::pos2(position.0 as f32, position.1 as f32);
                self.events.push(egui::Event::PointerMoved(self.pointer));
            }
            DeviceEvent::MouseInput { state, button } => {
                let Some(button) = pointer_button(*button) else { return };
                self.events.push(egui::Event::PointerButton {
                    pos: self.pointer,
                    button,
                    pressed: *state == ElementState::Pressed,
                    modifiers: self.modifiers,
                });
            }
            DeviceEvent::MouseWheel { delta } => {
                let (unit, delta) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (egui::MouseWheelUnit::Line, egui::vec2(*x, *y))
                    }
                    MouseScrollDelta::PixelDelta(x, y) => {
                        (egui::MouseWheelUnit::Point, egui::vec2(*x, *y))
                    }
                };
                self.events.push(egui::Event::MouseWheel {
                    unit,
                    delta,
                    modifiers: self.modifiers,
                });
            }
            DeviceEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                self.track_modifier(&event.logical_key, pressed);

                if let Some(key) = egui_key(&event.logical_key) {
                    self.events.push(egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed,
                        repeat: event.repeat,
                        modifiers: self.modifiers,
                    });
                }

                // Printable characters become text, unless a command modifier
                // is held — then the keypress is a shortcut, not typing.
                if pressed && !self.modifiers.ctrl && !self.modifiers.mac_cmd {
                    if let Key::Character(c) = event.logical_key {
                        if !c.is_control() {
                            self.events.push(egui::Event::Text(c.to_string()));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn track_modifier(&mut self, key: &Key, pressed: bool) {
        match key {
            Key::Named(NamedKey::Shift) => self.modifiers.shift = pressed,
            Key::Named(NamedKey::Alt) => self.modifiers.alt = pressed,
            Key::Named(NamedKey::Control) => {
                self.modifiers.ctrl = pressed;
                self.modifiers.command = pressed;
            }
            _ => {}
        }
    }

    /// Take the accumulated frame of input. `size` is the surface size in
    /// physical pixels.
    pub(crate) fn take(&mut self, size: (u32, u32)) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                // The canvas backing store is sized to its CSS box, so points
                // and physical pixels are the same here. Supporting HiDPI means
                // scaling the canvas by devicePixelRatio and dividing here.
                egui::vec2(size.0 as f32, size.1 as f32),
            )),
            time: Some(self.start.elapsed().as_secs_f64()),
            modifiers: self.modifiers,
            events: std::mem::take(&mut self.events),
            focused: true,
            ..Default::default()
        }
    }
}

fn pointer_button(button: MouseButton) -> Option<egui::PointerButton> {
    match button {
        MouseButton::Left => Some(egui::PointerButton::Primary),
        MouseButton::Right => Some(egui::PointerButton::Secondary),
        MouseButton::Middle => Some(egui::PointerButton::Middle),
        MouseButton::Back => Some(egui::PointerButton::Extra1),
        MouseButton::Forward => Some(egui::PointerButton::Extra2),
        MouseButton::Other(_) => None,
    }
}

fn egui_key(key: &Key) -> Option<egui::Key> {
    let named = match key {
        Key::Character(c) => return egui::Key::from_name(c.to_uppercase().to_string().as_str()),
        Key::Named(named) => named,
        Key::Unidentified => return None,
    };
    Some(match named {
        NamedKey::Escape => egui::Key::Escape,
        NamedKey::Enter => egui::Key::Enter,
        NamedKey::Tab => egui::Key::Tab,
        NamedKey::Backspace => egui::Key::Backspace,
        NamedKey::Delete => egui::Key::Delete,
        NamedKey::Space => egui::Key::Space,
        NamedKey::ArrowLeft => egui::Key::ArrowLeft,
        NamedKey::ArrowRight => egui::Key::ArrowRight,
        NamedKey::ArrowUp => egui::Key::ArrowUp,
        NamedKey::ArrowDown => egui::Key::ArrowDown,
        NamedKey::Home => egui::Key::Home,
        NamedKey::End => egui::Key::End,
        NamedKey::PageUp => egui::Key::PageUp,
        NamedKey::PageDown => egui::Key::PageDown,
        NamedKey::F1 => egui::Key::F1,
        NamedKey::F2 => egui::Key::F2,
        NamedKey::F3 => egui::Key::F3,
        NamedKey::F4 => egui::Key::F4,
        NamedKey::F5 => egui::Key::F5,
        NamedKey::F6 => egui::Key::F6,
        NamedKey::F7 => egui::Key::F7,
        NamedKey::F8 => egui::Key::F8,
        NamedKey::F9 => egui::Key::F9,
        NamedKey::F10 => egui::Key::F10,
        NamedKey::F11 => egui::Key::F11,
        NamedKey::F12 => egui::Key::F12,
        // Modifier keys are carried in `modifiers`, not as key events.
        NamedKey::Control | NamedKey::Alt | NamedKey::Shift | NamedKey::Super => return None,
    })
}
