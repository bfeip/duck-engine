use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{InnerSpace, Plane, Point3, Ray, Vector3};
use duck_engine_scene::resource::Visibility;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use glam::dvec3;
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
enum BoxAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the footprint. Preview is a flat rectangle face.
    /// Base being defined. `plane` is the plane the base rectangle is being defined on.
    Base { center: Point3, plane: Plane },
    /// Footprint fixed; the cursor drives the height. Preview is the 3D box.
    Height { center: Point3, width: f32, depth: f32, plane: Plane },
}

pub struct BoxOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<BoxAction>,
    cursor_target: Option<Point3>,
}

impl BoxOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                BoxAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                BoxAction::Cancel,
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

    /// Lays the flat unit footprint face (local XY, normal +Z) on `plane`, scaled to
    /// `width`×`depth`. [`Plane::rotation`] maps the local +Z axis to the plane normal.
    fn footprint_transform(center: Point3, width: f32, depth: f32, plane: &Plane) -> Transform {
        Transform {
            position: center,
            rotation: plane.rotation(),
            scale: Vector3::new(width, depth, 1.0),
        }
    }

    /// Scales the unit reference box (footprint in local XY, height along local +Z)
    /// to `width`×`depth`×`height` on `plane`. [`Plane::rotation`] maps the local +Z
    /// axis (the height axis) to the plane normal.
    fn box_transform(
        center: Point3,
        width: f32,
        depth: f32,
        height: f32,
        plane: &Plane,
    ) -> Transform {
        // Keep every scale component non-negative. A negative scale would make the
        // baked GTransform a reflection, flipping the box's face normals inward.
        let (base, height) = if height >= 0.0 {
            (center, height)
        } else {
            (center + plane.normal * height, -height)
        };
        Transform {
            position: base,
            rotation: plane.rotation(),
            scale: Vector3::new(width, depth, height),
        }
    }

    /// Unit reference box for the preview: footprint centered in local XY,
    /// height along local +Z (`[0, 1]`), scaled/oriented via the preview node
    /// transform ([`box_transform`](Self::box_transform)).
    fn reference_box() -> Shape {
        Shape::box_from_corners(dvec3(-0.5, -0.5, 0.0), dvec3(0.5, 0.5, 1.0))
    }

    /// World-space box with analytic planar faces: the footprint rectangle on
    /// `plane` extruded along its normal.
    fn analytic_box(
        center: Point3,
        width: f32,
        depth: f32,
        height: f32,
        plane: &Plane,
    ) -> Result<Shape, opencascade::Error> {
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
        )?;
        let face = Face::from_wire(&wire)?;
        let dir = plane.normal * height;
        Ok(face.extrude(dvec3(dir.x as f64, dir.y as f64, dir.z as f64)).into())
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

    /// A box is valid once it has a non-degenerate footprint and a non-zero height
    fn box_valid(width: f32, depth: f32, height: f32) -> bool {
        Self::footprint_valid(width, depth) && height.abs() > EPSILON
    }

    /// Signed height from projecting the cursor pick ray onto the plane normal through `center`.
    fn height_from_cursor(center: Point3, plane: &Plane, position: (f32, f32), ctx: &mut EventContext) -> f32 {
        let camera = ctx.camera();
        let ray: Ray = camera.ray_from_screen_point(position.0, position.1, ctx.size.0, ctx.size.1);
        ray.closest_param_on_axis(center, plane.normal).unwrap_or(0.0)
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
        // Seat the box on the snapped geometry when the snap carries a direction.
        // Otherwise use the construction plane.
        let plane = Plane::from_point(snap.direction.unwrap_or(cplane.normal), center);

        let Ok(preview_shape): Result<Shape, _> =
            Face::from_wire(&Wire::rect(1.0, 1.0).unwrap()).map(Into::into)
        else {
            return false;
        };

        // A single unit face, scaled each move; preview detail is irrelevant for a flat quad.
        let options = self.construction_options.borrow().geometry_options.clone();
        if self.preview.add_preview_from_shape(&preview_shape, &options, "box plane preview").is_none() {
            return false;
        }
        // Hidden until the cursor defines a non-degenerate footprint.
        self.preview.set_preview_visibility(Visibility::Invisible);
        self.phase = Phase::Base { center, plane };
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

        // Swap the flat footprint preview for the 3D box preview.
        let preview_shape = Self::reference_box();
        let options = self.construction_options.borrow().geometry_options.clone();
        if self.preview.try_replace_preview(&preview_shape, &options, "box preview").is_none() {
            return false;
        }
        // Hidden until the cursor defines a non-zero height.
        self.preview.set_preview_visibility(Visibility::Invisible);
        self.phase = Phase::Height { center, width, depth, plane };
        true
    }

    fn on_place_height(
        &mut self,
        center: Point3,
        width: f32,
        depth: f32,
        plane: Plane,
        position: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let height = Self::height_from_cursor(center, &plane, position, ctx);
        // A zero-height (degenerate) box can't be committed; stay in the height stage.
        if !Self::box_valid(width, depth, height) {
            return false;
        }

        // Build the box analytically in world space
        let Ok(world_shape) = Self::analytic_box(center, width, depth, height, &plane) else {
            return false;
        };

        // Discard the preview, then commit the world-space shape as a registered part.
        let _ = self.preview.commit();

        let committed = {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            doc.add_part("Box".to_owned(), world_shape, &coptions.geometry_options)
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

        match self.phase {
            Phase::Idle => {
                self.cursor_target = snap.map(|s| s.position);
            }
            Phase::Base { center, plane } => {
                self.cursor_target = snap.map(|s| s.position);
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
            Phase::Height { center, width, depth, plane } => {
                let height = Self::height_from_cursor(center, &plane, cursor, ctx);
                self.cursor_target = Some(center + plane.normal * height);
                if let Some(preview_node) = self.preview.preview_node() {
                    let mut scene = ctx.scene.lock();
                    if Self::box_valid(width, depth, height) {
                        scene.set_node_visibility(preview_node, Visibility::Visible);
                        scene.set_node_transform(
                            preview_node,
                            Self::box_transform(center, width, depth, height, &plane),
                        );
                    } else {
                        // Degenerate height: nothing to draw.
                        scene.set_node_visibility(preview_node, Visibility::Invisible);
                    }
                }
            }
        }
    }
}

impl ModelingTool for BoxOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "box", icon: icons::BOX, shortcut: None }
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

impl Operator for BoxOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        BoxAction::Place => match self.phase {
                            Phase::Idle => self.on_place_center(*position, ctx),
                            Phase::Base { center, plane } => {
                                self.on_place_corner(center, plane, *position, ctx)
                            }
                            Phase::Height { center, width, depth, plane } => {
                                self.on_place_height(center, width, depth, plane, *position, ctx)
                            }
                        },
                        BoxAction::Cancel => {
                            let was_defining = !matches!(self.phase, Phase::Idle);
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
        "Box"
    }
}
