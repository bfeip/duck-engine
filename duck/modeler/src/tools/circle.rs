use duck_engine_common::{MetricSpace, Point3, Vector3};
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
};
use log::warn;

use crate::ops::primitives::{circle, region};
use crate::preview::PreviewSession;
use crate::tools::{ModelingTool, ToolInfo, Workspace};
use crate::ui::icons;

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
    workspace: Workspace,
    preview: PreviewSession,
    bindings: InputMap<CircleAction>,
    cursor_target: Option<Point3>,
}


impl CircleTool {
    pub fn new(workspace: &Workspace) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                CircleAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                CircleAction::Cancel,
            );
        let preview = workspace.preview_session();
        Self {
            phase: Phase::Idle,
            workspace: workspace.clone(),
            preview,
            bindings,
            cursor_target: None,
        }
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let cplane_normal = self.workspace.construction.borrow().construction_plane.normal;
        let Some(snap) = self.workspace.snap(position, &[], ctx)
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
        let preview_options = self.workspace.preview_options();
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
        // Exclude the preview so the radius can snap through a corner, not to the
        // preview's own geometry.
        let radius = self.workspace.snap(position, self.preview.preview_nodes(), ctx)
            .map(|s| center.distance(s.position).max(0.01))
            .unwrap_or(0.01);

        let shape = circle(center, normal, radius).map(region);

        // Discard the preview node, then commit the world-space shape as a part.
        let _ = self.preview.commit();
        self.phase = Phase::Idle;
        match shape.and_then(|shape| self.workspace.add_numbered_part("Circle", shape)) {
            Ok(_) => true,
            Err(e) => {
                self.workspace.notifications.failure("Circle", &e);
                false
            }
        }
    }

    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);

        // While defining, exclude our own preview so the radius doesn't snap to it.
        let snap = self.workspace.snap(cursor, self.preview.preview_nodes(), ctx);

        // Record where the modeler should draw the 3D cursor.
        self.cursor_target = snap.map(|s| s.position);

        // Rebuild the preview disk from the snapped radius while defining. A flat
        // disk's orientation depends on the plane normal, so we re-tessellate
        // rather than scaling a unit mesh.
        if let Phase::Defining { center, normal } = self.phase {
            if let Some(snap) = snap {
                let radius = center.distance(snap.position).max(0.01);
                if let Ok(shape) = circle(center, normal, radius).map(region) {
                    let preview_options = self.workspace.preview_options();
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
