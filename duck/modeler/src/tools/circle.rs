use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::{MetricSpace, Point3, Vector3};
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use log::warn;

use crate::document::Document;
use crate::ops::primitives::{circle, region};
use crate::preview::PreviewSession;
use crate::tools::{ModelingTool, ToolInfo};
use crate::ui::icons;
use crate::construction::ConstructionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CircleAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the radius. `normal` is the disk's plane
    /// normal.
    Defining { center: Point3, normal: Vector3 },
}

pub struct CircleTool {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<CircleAction>,
    cursor_target: Option<Point3>,
}


impl CircleTool {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                CircleAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                CircleAction::Cancel,
            );
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            preview,
            bindings,
            cursor_target: None,
        }
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let camera = ctx.camera.clone();
        let cplane_normal = self.construction_options.borrow().construction_plane.normal;
        let Some(snap) = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[], &camera, ctx, &[])
        else {
            return false;
        };
        let center = snap.position;
        // Lay the disk on the snapped geometry when the snap carries a direction.
        // Otherwise use the construction plane.
        let normal = snap.direction.unwrap_or(cplane_normal);
        let shape = match circle(center, normal, 0.01) {
            Ok(outline) => region(outline),
            Err(e) => {
                warn!("Failed to build the circle: {e:#}");
                return false;
            }
        };
        // Coarser preview tolerance since the preview is rebuilt on every move.
        let preview_options = self.construction_options.borrow().preview_options();
        if self.preview.add_preview_from_shape(&shape, &preview_options, "circle").is_none() {
            return false;
        }
        self.phase = Phase::Defining { center, normal };
        true
    }

    fn on_place_outer(
        &mut self,
        center: Point3,
        normal: Vector3,
        position: (f32, f32),
        ctx: &mut EventContext
    ) -> bool {
        let camera = ctx.camera.clone();
        // Exclude the preview so the radius can snap through a corner, not to the
        // preview's own geometry.
        let radius = self
            .construction_options
            .borrow()
            .resolve_snap(position, self.preview.preview_nodes(), &camera, ctx, &[])
            .map(|s| center.distance(s.position).max(0.01))
            .unwrap_or(0.01);

        let shape = circle(center, normal, radius).map(region);

        // Discard the preview node, then commit the world-space shape as a part.
        let _ = self.preview.commit();

        let committed = match shape {
            Ok(shape) => {
                let coptions = self.construction_options.borrow();
                let mut doc = self.document.lock().unwrap();
                doc.add_numbered_part("Circle", shape, &coptions.geometry_options)
                    .is_ok()
            }
            Err(e) => {
                warn!("Failed to build the circle: {e:#}");
                false
            }
        };

        self.phase = Phase::Idle;
        committed
    }

    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);

        let camera = ctx.camera.clone();
        // While defining, exclude our own preview so the radius doesn't snap to it.
        let snap = self.construction_options.borrow().resolve_snap(
            cursor,
            self.preview.preview_nodes(),
            &camera,
            ctx,
            &[],
        );

        // Record where the modeler should draw the 3D cursor.
        self.cursor_target = snap.map(|s| s.position);

        // Rebuild the preview disk from the snapped radius while defining. A flat
        // disk's orientation depends on the plane normal, so we re-tessellate
        // rather than scaling a unit mesh.
        if let Phase::Defining { center, normal } = self.phase {
            if let Some(snap) = snap {
                let radius = center.distance(snap.position).max(0.01);
                if let Ok(shape) = circle(center, normal, radius).map(region) {
                    let preview_options = self.construction_options.borrow().preview_options();
                    self.preview.try_replace_preview(&shape, &preview_options, "circle");
                }
            }
        }
    }
}

impl ModelingTool for CircleTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "circle", icon: icons::CIRCLE, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        // The modeler hides the cursor for the (now inactive) tool, but clear our
        // target so a stale point can't flash if we're reactivated before a move.
        self.cursor_target = None;
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor_target
    }
}

impl Operator for CircleTool {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        CircleAction::Place => {
                            if let Phase::Defining { center, normal } = self.phase {
                                self.on_place_outer(center, normal, *position, ctx)
                            } else {
                                self.on_place_center(*position, ctx)
                            }
                        }
                        CircleAction::Cancel => {
                            let was_defining = matches!(self.phase, Phase::Defining { .. });
                            if was_defining {
                                self.cancel();
                            }
                            was_defining
                        }
                    };
                }
                handled
            }
            DeviceEvent::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Circle"
    }
}
