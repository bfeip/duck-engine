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
use glam::{dvec3, DVec3};
use log::warn;
use opencascade::primitives::{Edge, Shape, Wire};

use crate::document::Document;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

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

pub struct CircleOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<CircleAction>,
    cursor_target: Option<Point3>,
}

fn to_dvec3(p: Point3) -> DVec3 {
    dvec3(p.x as f64, p.y as f64, p.z as f64)
}

fn vec_to_dvec3(v: Vector3) -> DVec3 {
    dvec3(v.x as f64, v.y as f64, v.z as f64)
}

/// Builds a filled planar disk bounded by a circle of `radius` centered at
/// `center`, lying in the plane with the given `normal`. Falls back to the bare
/// ring (wire) if the face can't be built. Returns `None` on construction error.
fn circle_shape(center: Point3, normal: Vector3, radius: f64) -> Option<Shape> {
    let edge = Edge::circle(to_dvec3(center), vec_to_dvec3(normal), radius)
        .map_err(|e| warn!("Failed to build circle edge: {e}"))
        .ok()?;
    let wire = Wire::from_edges(&[edge])
        .map_err(|e| warn!("Failed to build circle wire: {e}"))
        .ok()?;
    // Filled disk (Region): fall back to the bare ring if the face can't be
    // built. `to_face` consumes the wire, so capture the ring shape first.
    let ring = Shape::from(&wire);
    match wire.to_face() {
        Ok(face) => Some(Shape::from(&face)),
        Err(_) => Some(ring),
    }
}

impl CircleOperator {
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
        let Some(shape) = circle_shape(center, normal, 0.01) else {
            return false;
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
            .map(|s| center.distance(s.position).max(0.01) as f64)
            .unwrap_or(0.01);

        let shape = circle_shape(center, normal, radius);

        // Discard the preview node, then commit the world-space shape as a part.
        let _ = self.preview.commit();

        let committed = if let Some(shape) = shape {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            doc.add_numbered_part("Circle", shape, &coptions.geometry_options)
                .is_ok()
        } else {
            warn!("Failed to build circle shape.");
            false
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
                let radius = center.distance(snap.position).max(0.01) as f64;
                if let Some(shape) = circle_shape(center, normal, radius) {
                    let preview_options = self.construction_options.borrow().preview_options();
                    self.preview.try_replace_preview(&shape, &preview_options, "circle");
                }
            }
        }
    }
}

impl ModelingTool for CircleOperator {
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

impl Operator for CircleOperator {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_shape_from_positive_radius() {
        let center = Point3::new(0.0, 0.0, 0.0);
        let normal = Vector3::new(0.0, 1.0, 0.0);
        assert!(circle_shape(center, normal, 1.0).is_some());
    }

    #[test]
    fn circle_shape_off_origin() {
        let center = Point3::new(3.0, 0.0, -2.0);
        let normal = Vector3::new(0.0, 1.0, 0.0);
        assert!(circle_shape(center, normal, 2.5).is_some());
    }
}
