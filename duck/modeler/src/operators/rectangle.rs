use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::{InnerSpace, Plane, Point3, Vector3};
use glam::dvec3;
use duck_engine_scene::Visibility;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use log::warn;
use opencascade::primitives::{Face, Shape, Wire};

use crate::document::Document;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

/// A dimension at or below this is degenerate: the preview is hidden and the pick
/// can't be committed.
const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RectangleAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the footprint. Preview is a flat rectangle face.
    /// `plane` is the placement plane through the center.
    Defining { center: Point3, plane: Plane },
}

pub struct RectangleOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<RectangleAction>,
    cursor_target: Option<Point3>,
}

impl RectangleOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                RectangleAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                RectangleAction::Cancel,
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

    /// Lays the unit reference face (local XY, normal +Z) flat on `plane`, scaled to
    /// the footprint. [`Plane::rotation`] maps the local +Z axis to the plane normal.
    fn footprint_transform(center: Point3, width: f32, depth: f32, plane: &Plane) -> Transform {
        Transform {
            position: center,
            rotation: plane.rotation(),
            scale: Vector3::new(width, depth, 1.0),
        }
    }

    /// In-plane extents from the center→corner vector, as full (width, depth).
    fn footprint_dims(center: Point3, corner: Point3, plane: &Plane) -> (f32, f32) {
        let (u, v) = plane.basis();
        let d = corner - center;
        let width = 2.0 * d.dot(u).abs();
        let depth = 2.0 * d.dot(v).abs();
        (width, depth)
    }

    /// A footprint is valid once both in-plane dimensions are non-degenerate.
    fn footprint_valid(width: f32, depth: f32) -> bool {
        width > EPSILON && depth > EPSILON
    }

    /// Unit reference rectangle face (local XY, normal +Z), oriented onto the
    /// construction plane via the preview transform.
    fn reference_face() -> Option<Shape> {
        let wire = Wire::rect(1.0, 1.0)
            .map_err(|e| warn!("Failed to build rectangle wire: {e}"))
            .ok()?;
        Face::from_wire(&wire)
            .map(Into::into)
            .map_err(|e| warn!("Failed to build rectangle face: {e}"))
            .ok()
    }

    /// World-space rectangle face with an analytic planar surface.
    fn analytic_face(center: Point3, width: f32, depth: f32, plane: &Plane) -> Option<Shape> {
        let (u, v) = plane.basis();
        let half_w = u * (0.5 * width);
        let half_d = v * (0.5 * depth);
        let corners = [
            center - half_w - half_d,
            center + half_w - half_d,
            center + half_w + half_d,
            center - half_w + half_d,
        ];
        let wire = Wire::from_ordered_points(
            corners.iter().map(|p| dvec3(p.x as f64, p.y as f64, p.z as f64)),
        )
        .map_err(|e| warn!("Failed to build rectangle wire: {e}"))
        .ok()?;
        Face::from_wire(&wire)
            .map(Into::into)
            .map_err(|e| warn!("Failed to build rectangle face: {e}"))
            .ok()
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let camera = ctx.camera();
        let cplane = self.construction_options.borrow().construction_plane;
        let Some(snap) = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[], &camera, ctx, &[])
        else {
            return false;
        };
        let center = snap.position;
        // Seat the rectangle on the snapped geometry when the snap carries a
        // direction. Otherwise use the  construction plane.
        let plane = Plane::from_point(snap.direction.unwrap_or(cplane.normal), center);

        let Some(preview_shape) = Self::reference_face() else {
            return false;
        };

        // A single unit face, scaled each move; preview detail is irrelevant for a flat quad.
        let options = self.construction_options.borrow().geometry_options.clone();
        if self.preview.add_preview_from_shape(&preview_shape, &options, "rectangle preview").is_none() {
            return false;
        }
        // Hidden until the cursor defines a non-degenerate footprint.
        self.preview.set_preview_visibility(Visibility::Invisible);
        self.phase = Phase::Defining { center, plane };
        true
    }

    fn on_place_corner(
        &mut self,
        center: Point3,
        plane: Plane,
        position: (f32, f32),
        ctx: &mut EventContext
    ) -> bool {
        let camera = ctx.camera();
        // Exclude the preview so the footprint can snap through it.
        let Some(corner) = self
            .construction_options
            .borrow()
            .resolve_snap(position, self.preview.preview_nodes(), &camera, ctx, &[])
            .map(|s| s.position)
        else {
            return false;
        };
        let (width, depth) = Self::footprint_dims(center, corner, &plane);
        // A degenerate footprint can't be committed; stay in the footprint stage.
        if !Self::footprint_valid(width, depth) {
            return false;
        }

        // Build the face analytically in world space
        let Some(world_shape) = Self::analytic_face(center, width, depth, &plane) else {
            return false;
        };

        // Discard the preview, then commit the world-space shape as a registered part.
        let _ = self.preview.commit();

        let committed = {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            doc.add_part("Rectangle".to_owned(), world_shape, &coptions.geometry_options)
                .is_ok()
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

        let camera = ctx.camera();
        // While defining, exclude our own preview so snapping doesn't lock onto it.
        let snap = self.construction_options.borrow().resolve_snap(
            cursor,
            self.preview.preview_nodes(),
            &camera,
            ctx,
            &[],
        );

        self.cursor_target = snap.map(|s| s.position);

        if let Phase::Defining { center, plane } = self.phase {
            let dims = snap.map(|s| Self::footprint_dims(center, s.position, &plane));
            if let Some(preview_node) = self.preview.preview_node() {
                let mut scene = ctx.scene.lock();
                match dims {
                    Some((width, depth)) if Self::footprint_valid(width, depth) => {
                        scene.set_node_visibility(preview_node, Visibility::Visible);
                        scene.set_node_transform(
                            preview_node,
                            Self::footprint_transform(center, width, depth, &plane),
                        );
                    }
                    // No snap, or a degenerate footprint: nothing to draw.
                    _ => scene.set_node_visibility(preview_node, Visibility::Invisible),
                }
            }
        }
    }
}

impl ModelingTool for RectangleOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "rectangle", icon: icons::RECTANGLE, shortcut: None }
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

impl Operator for RectangleOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        RectangleAction::Place => match self.phase {
                            Phase::Idle => self.on_place_center(*position, ctx),
                            Phase::Defining { center, plane } => {
                                self.on_place_corner(center, plane, *position, ctx)
                            }
                        },
                        RectangleAction::Cancel => {
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
        "Rectangle"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_face_builds() {
        assert!(RectangleOperator::reference_face().is_some());
    }

    #[test]
    fn footprint_dims_doubles_half_extents() {
        // xz plane: normal +Y, in-plane basis (u=+X, v=+Z).
        let plane = Plane::xz();
        let center = Point3::new(0.0, 0.0, 0.0);
        let corner = Point3::new(1.5, 0.0, 2.0);
        let (width, depth) = RectangleOperator::footprint_dims(center, corner, &plane);
        assert!(RectangleOperator::footprint_valid(width, depth));
        assert!((width - 3.0).abs() < EPSILON);
        assert!((depth - 4.0).abs() < EPSILON);
    }

    #[test]
    fn footprint_invalid_when_corner_on_center() {
        let plane = Plane::xz();
        let center = Point3::new(0.0, 0.0, 0.0);
        let (width, depth) = RectangleOperator::footprint_dims(center, center, &plane);
        assert!(!RectangleOperator::footprint_valid(width, depth));
    }
}
