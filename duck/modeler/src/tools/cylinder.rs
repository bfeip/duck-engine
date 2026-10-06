use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{MetricSpace, Plane, Point3, Ray, Real, Vector3};
use duck_engine_scene::resource::Visibility;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, Modifiers, MouseButton, NamedKey},
    operator::{Handle, HandleDrag, HandleEvent, HandleId, HandleReach, HandleShape, Operator},
    selection::SelectionManager,
};
use log::{error, warn};
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::ops::primitives::{circle, cylinder, region};
use crate::preview::PreviewSession;
use crate::tools::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::tweak::{
    commit_tweak, dimension_field, grip_dimension, handle_tweak, tweak_panel, PrimitiveParams,
    TweakAction, TweakParams,
};
use crate::construction::ConstructionOptions;

/// A dimension at or below this is degenerate: the preview is hidden and the pick
/// can't be committed.
const EPSILON: Real = 1e-6;

/// The cylinder's dimension grips.
const RADIUS_HANDLE: HandleId = HandleId(0);
const HEIGHT_HANDLE: HandleId = HandleId(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CylinderAction {
    Place,
    /// End the operation: commit if the cylinder is fully defined, else abort.
    Finish,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the radius. Preview is the flat base face.
    /// `plane` is the placement plane through the center.
    Radius { center: Point3, plane: Plane },
    /// Radius fixed; the cursor drives the height. Preview is the 3D cylinder.
    Height { center: Point3, radius: Real, plane: Plane },
    /// Every point picked; the options panel drives the dimensions until the
    /// cylinder is applied or cancelled.
    Tweak(CylinderParams),
}

/// The dimensions of a placed cylinder, adjustable before it is committed.
/// `base` is the first point picked and never moves: the radius grows about the
/// axis through it and the height grows away from it along `plane.normal`.
#[derive(Clone, Copy)]
pub(super) struct CylinderParams {
    base: Point3,
    plane: Plane,
    radius: Real,
    height: Real,
}

impl CylinderParams {
    /// Parameters for a finished pick, normalized so the height is positive and
    /// grows away from `base`. A downward pick flips the plane normal rather
    /// than moving the base off the picked point, so later height edits move
    /// only the far cap.
    fn from_pick(base: Point3, radius: Real, height: Real, plane: Plane) -> Self {
        let (plane, height) = if height >= 0.0 {
            (plane, height)
        } else {
            (Plane::from_point(-plane.normal, base), -height)
        };
        Self { base, plane, radius, height }
    }
}

impl PrimitiveParams for CylinderParams {
    const NAME: &'static str = "Cylinder";

    /// Scales the unit reference cylinder (base at the origin, axis +Z, radius 1,
    /// height 1) to these dimensions.  Every scale
    /// component stays non-negative — a negative one would make the baked
    /// transform a reflection, flipping the face normals inward.
    fn preview_transform(&self) -> Transform {
        Transform {
            position: self.base,
            rotation: self.plane.rotation(),
            scale: Vector3::new(self.radius, self.radius, self.height),
        }
    }

    fn build(&self) -> anyhow::Result<Shape> {
        Ok(cylinder(self.base, self.plane.normal, self.radius, self.height))
    }
}

impl TweakParams for CylinderParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = dimension_field(ui, "Radius", &mut self.radius);
        changed |= dimension_field(ui, "Height", &mut self.height);
        changed
    }

    /// A radius grip on the base rim and a height grip on the far cap, each tied
    /// back to the base by a leader line so both meet at the bottom centre.
    fn handles(&self) -> Vec<Handle> {
        let (u, _) = self.plane.basis();
        let grip = |id, anchor, direction: Vector3| {
            Handle::new(id, HandleShape::Cube, anchor)
                .with_direction(direction)
                .with_reach(HandleReach::Leader(self.base))
        };
        vec![
            grip(RADIUS_HANDLE, self.base + u * self.radius, u),
            grip(HEIGHT_HANDLE, self.base + self.plane.normal * self.height, self.plane.normal),
        ]
    }

    /// The grabbed rim or cap follows the cursor one for one: the radius is
    /// measured from the axis and the height from `base`, neither of which moves.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        match drag.id {
            RADIUS_HANDLE => {
                let (u, _) = grabbed.plane.basis();
                self.radius = grip_dimension(grabbed.radius, drag.distance_along(u));
            }
            HEIGHT_HANDLE => {
                self.height =
                    grip_dimension(grabbed.height, drag.distance_along(grabbed.plane.normal));
            }
            _ => {}
        }
    }
}

pub struct CylinderTool {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<CylinderAction>,
    cursor_target: Option<Point3>,
    /// Dimensions as they were when the held grip was grabbed; `None` when no
    /// grip is held. A drag reports its total offset, so it is applied to this
    /// rather than to the live dimensions.
    grabbed: Option<CylinderParams>,
    // Set once the cylinder is applied, so
    // the tool cedes back to selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}



impl CylinderTool {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                CylinderAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                CylinderAction::Finish,
            );
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            preview,
            bindings,
            cursor_target: None,
            grabbed: None,
            finished: false,
        }
    }

    /// Writes `params` back into the tweak phase and refreshes the preview.
    /// The 3D twin of the panel's [`TweakAction::Changed`] arm.
    fn set_tweak(&mut self, params: CylinderParams) {
        self.preview.set_preview_transform(params.preview_transform());
        self.phase = Phase::Tweak(params);
    }

    /// Lays the flat unit base disk (local XY, normal +Z) on `plane`, scaled to
    /// `radius`. [`Plane::rotation`] maps the local +Z axis to the plane normal.
    fn disk_transform(center: Point3, radius: Real, plane: &Plane) -> Transform {
        Transform {
            position: center,
            rotation: plane.rotation(),
            scale: Vector3::new(radius, radius, 1.0),
        }
    }

    /// Unit reference cylinder for the preview: base at the origin, axis +Z,
    /// radius 1, height 1. Scaled and oriented via
    /// [`CylinderParams::preview_transform`].
    fn reference_cylinder() -> Shape {
        Shape::cylinder_radius_height(1.0, 1.0)
    }

    /// Unit reference base disk for the radius phase: a lone face of radius 1
    /// centered on the origin in local XY, so it reads as a sketch until the
    /// cursor gives the cylinder a height. Placed via the preview node transform
    /// ([`disk_transform`](Self::disk_transform)).
    fn reference_disk() -> anyhow::Result<Shape> {
        Ok(region(circle(Point3::new(0.0, 0.0, 0.0), Vector3::unit_z(), 1.0)?))
    }

    /// A radius is valid once it is non-degenerate.
    fn radius_valid(radius: Real) -> bool {
        radius > EPSILON
    }

    /// A cylinder is valid once it has a non-degenerate radius and a non-zero height.
    fn cylinder_valid(radius: Real, height: Real) -> bool {
        Self::radius_valid(radius) && height.abs() > EPSILON
    }

    /// Signed height from projecting the cursor pick ray onto the plane normal through `center`.
    fn height_from_cursor(center: Point3, plane: &Plane, position: (f32, f32), ctx: &mut EventContext) -> Real {
        let camera = ctx.camera.clone();
        let ray: Ray = camera.ray_from_screen_point(position.0, position.1, ctx.size.0, ctx.size.1);
        ray.closest_param_on_axis(center, plane.normal).unwrap_or(0.0)
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let camera = ctx.camera.clone();
        let cplane = self.construction_options.borrow().construction_plane;
        let Some(snap) = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[], &camera, ctx, &[])
        else {
            return false;
        };
        let center = snap.position;
        // Orient the cylinder to the snapped geometry when the snap carries a
        // direction. Otherwise fall back to the construction plane.
        let plane = Plane::from_point(snap.direction.unwrap_or(cplane.normal), center);

        // A single unit disk, scaled each move; preview detail is irrelevant here.
        let preview_shape = match Self::reference_disk() {
            Ok(shape) => shape,
            Err(e) => {
                warn!("Failed to build the base disk: {e:#}");
                return false;
            }
        };
        let options = self.construction_options.borrow().geometry_options.clone();
        if self.preview.add_preview_from_shape(&preview_shape, &options, "cylinder base preview").is_none() {
            return false;
        }
        // Hidden until the cursor defines a non-degenerate radius.
        self.preview.set_preview_visibility(Visibility::Invisible);
        self.phase = Phase::Radius { center, plane };
        true
    }

    fn on_place_radius(
        &mut self,
        center: Point3,
        plane: Plane,
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
            .map(|s| center.distance(s.position))
            .unwrap_or(0.0);
        // A degenerate radius can't be committed; stay in the radius stage.
        if !Self::radius_valid(radius) {
            return false;
        }

        // Swap the flat base preview for the 3D cylinder preview.
        let options = self.construction_options.borrow().geometry_options.clone();
        if self
            .preview
            .try_replace_preview(&Self::reference_cylinder(), &options, "cylinder preview")
            .is_none()
        {
            return false;
        }
        // Hidden until the cursor defines a non-zero height.
        self.preview.set_preview_visibility(Visibility::Invisible);
        self.phase = Phase::Height { center, radius, plane };
        true
    }

    fn on_place_height(
        &mut self,
        center: Point3,
        radius: Real,
        plane: Plane,
        position: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let height = Self::height_from_cursor(center, &plane, position, ctx);
        // A zero-height (degenerate) cylinder can't be tweaked; stay in the height stage.
        if !Self::cylinder_valid(radius, height) {
            return false;
        }

        // Hand the cylinder to the options panel rather than committing it: the
        // preview stays live and the dimensions stay editable until Apply.
        let params = CylinderParams::from_pick(center, radius, height, plane);
        self.preview.set_preview_transform(params.preview_transform());
        self.phase = Phase::Tweak(params);
        true
    }

    /// Commit the cylinder and finish the tool. A failed build keeps the
    /// panel open so the dimensions can be corrected.
    fn apply(&mut self) -> anyhow::Result<()> {
        let Phase::Tweak(params) = self.phase else { return Ok(()) };
        let options = self.construction_options.borrow().geometry_options.clone();
        commit_tweak(&params, &mut self.preview, &self.document, &options)?;
        self.phase = Phase::Idle;
        self.finished = true;
        Ok(())
    }

    /// Apply, logging a failure. For the gestures that keep the tool active and
    /// so must report for themselves: the panel's Apply button, Enter, right-click.
    fn apply_and_report(&mut self) {
        if let Err(e) = self.apply() {
            error!("Cylinder failed: {e:#}");
        }
    }

    /// Drop the in-progress cylinder.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        // Every point is picked; the panel drives the preview from here.
        if matches!(self.phase, Phase::Tweak(_)) {
            return;
        }
        let cursor = (position.0 as f32, position.1 as f32);

        let camera = ctx.camera.clone();
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
            Phase::Radius { center, plane } => {
                self.cursor_target = snap.map(|s| s.position);
                let radius = snap.map(|s| center.distance(s.position));
                if let Some(preview_node) = self.preview.preview_node() {
                    let mut scene = ctx.scene.lock();
                    match radius {
                        Some(radius) if Self::radius_valid(radius) => {
                            scene.set_node_visibility(preview_node, Visibility::Visible);
                            scene.set_node_transform(
                                preview_node,
                                Self::disk_transform(center, radius, &plane),
                            );
                        }
                        // No snap, or a degenerate radius: nothing to draw.
                        _ => scene.set_node_visibility(preview_node, Visibility::Invisible),
                    }
                }
            }
            Phase::Height { center, radius, plane } => {
                let height = Self::height_from_cursor(center, &plane, cursor, ctx);
                self.cursor_target = Some(center + plane.normal * height);
                if let Some(preview_node) = self.preview.preview_node() {
                    let mut scene = ctx.scene.lock();
                    if Self::cylinder_valid(radius, height) {
                        scene.set_node_visibility(preview_node, Visibility::Visible);
                        scene.set_node_transform(
                            preview_node,
                            CylinderParams::from_pick(center, radius, height, plane)
                                .preview_transform(),
                        );
                    } else {
                        // Degenerate height: nothing to draw.
                        scene.set_node_visibility(preview_node, Visibility::Invisible);
                    }
                }
            }
            // Handled by the early return above.
            Phase::Tweak(_) => {}
        }
    }
}

impl ModelingTool for CylinderTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "cylinder", icon: icons::CYLINDER, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.finished = false;
        self.grabbed = None;
        // The modeler hides the cursor for the (now inactive) tool, but clear our
        // target so a stale point can't flash if we're reactivated before a move.
        self.cursor_target = None;
    }

    /// A cylinder waiting on the panel is fully defined, so leaving the tool commits it.
    fn finalize(&mut self, _selection: &mut SelectionManager) -> anyhow::Result<()> {
        if matches!(self.phase, Phase::Tweak(_)) {
            self.apply()?;
        }
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn cursor_target(&self) -> Option<Point3> {
        // Nothing left to pick while the panel is open.
        match self.phase {
            Phase::Tweak(_) => None,
            _ => self.cursor_target,
        }
    }

    /// Grips only once the cylinder is placed — until then the cursor is still
    /// defining it, and a grip would be something to fight with.
    fn handles(&self) -> Vec<Handle> {
        match &self.phase {
            Phase::Tweak(params) => params.handles(),
            _ => Vec::new(),
        }
    }

    fn on_handle(&mut self, event: &HandleEvent) {
        let Phase::Tweak(params) = self.phase else { return };
        if let Some(edited) = handle_tweak(params, &mut self.grabbed, event) {
            self.set_tweak(edited);
        }
    }

    fn panel_title(&self) -> Option<&str> {
        matches!(self.phase, Phase::Tweak(_)).then_some(CylinderParams::NAME)
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, _panel: &mut PanelContext) {
        let Phase::Tweak(params) = &mut self.phase else { return };
        let action = tweak_panel(ui, params);
        let transform = params.preview_transform();
        match action {
            TweakAction::Changed => self.preview.set_preview_transform(transform),
            TweakAction::Apply => self.apply_and_report(),
            TweakAction::Cancel => self.cancel(),
            TweakAction::None => {}
        }
    }
}

impl Operator for CylinderTool {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        CylinderAction::Place => match self.phase {
                            Phase::Idle => self.on_place_center(*position, ctx),
                            Phase::Radius { center, plane } => {
                                self.on_place_radius(center, plane, *position, ctx)
                            }
                            Phase::Height { center, radius, plane } => {
                                self.on_place_height(center, radius, plane, *position, ctx)
                            }
                            // Swallow the click: the panel owns the cylinder now, so
                            // a stray pick must not select or place anything.
                            Phase::Tweak(_) => true,
                        },
                        // Right-click ends the operation: it commits a cylinder the
                        // panel already holds, and aborts one still being picked.
                        CylinderAction::Finish => match self.phase {
                            Phase::Idle => false,
                            Phase::Tweak(_) => {
                                self.apply_and_report();
                                true
                            }
                            _ => {
                                self.cancel();
                                true
                            }
                        },
                    };
                }
                handled
            }
            DeviceEvent::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            // Keyboard equivalents of the tweak panel's Apply and Cancel buttons.
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if !matches!(self.phase, Phase::Tweak(_))
                    || key_event.state != ElementState::Pressed
                    || key_event.repeat
                {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => {
                        self.apply_and_report();
                        true
                    }
                    Key::Named(NamedKey::Escape) => {
                        self.cancel();
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Cylinder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::InnerSpace;
    use crate::tools::tweak::MIN_DIMENSION;

    #[test]
    fn cylinder_valid_accepts_nondegenerate() {
        assert!(CylinderTool::cylinder_valid(1.0, 2.0));
        assert!(CylinderTool::cylinder_valid(1.0, -2.0));
    }

    #[test]
    fn cylinder_valid_rejects_degenerate() {
        assert!(!CylinderTool::cylinder_valid(0.0, 2.0));
        assert!(!CylinderTool::cylinder_valid(1.0, 0.0));
    }

    #[test]
    fn from_pick_flips_negative_height_about_the_base() {
        let plane = Plane::xz();
        let base = Point3::new(1.0, 2.0, 3.0);
        let params = CylinderParams::from_pick(base, 2.0, -3.0, plane);
        // The picked point stays the base; the axis flips instead (XZ normal is +Y).
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.height - 3.0).abs() < EPSILON);
        assert!((params.plane.normal + plane.normal).magnitude() < EPSILON);

        let t = params.preview_transform();
        // Scale stays non-negative after the flip.
        assert!(t.scale.x >= 0.0 && t.scale.y >= 0.0 && t.scale.z >= 0.0);
        assert!((t.position - base).magnitude() < EPSILON);
    }

    #[test]
    fn height_edits_leave_the_base_cap_in_place() {
        let plane = Plane::from_point(Vector3::new(1.0, 2.0, 3.0).normalize(), Point3::new(0.0, 0.0, 0.0));
        let base = Point3::new(-4.0, 5.0, 6.0);
        let mut params = CylinderParams::from_pick(base, 2.0, 3.0, plane);
        params.height = 10.0;
        // Only the far cap moves: base, axis and radius are untouched.
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.plane.normal - plane.normal).magnitude() < EPSILON);
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }

    /// A plane aligned with no world axis, so a mistaken basis shows up.
    fn skewed_plane(origin: Point3) -> Plane {
        Plane::from_point(Vector3::new(1.0, 2.0, 3.0).normalize(), origin)
    }

    /// A drag of `offset` on the grip `id`, as the handle machinery reports it.
    fn drag(id: HandleId, offset: Vector3) -> HandleDrag {
        let grab = Point3::new(0.0, 0.0, 0.0);
        HandleDrag { id, grab, point: grab + offset, modifiers: Modifiers::default() }
    }

    #[test]
    fn grips_sit_on_the_rim_and_the_far_cap() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let plane = skewed_plane(base);
        let params = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let handles = params.handles();
        assert_eq!(handles.len(), 2);

        let expected = [
            (RADIUS_HANDLE, base + u * 2.0, u),
            (HEIGHT_HANDLE, base + plane.normal * 5.0, plane.normal),
        ];
        for (id, anchor, direction) in expected {
            let handle = handles.iter().find(|h| h.id == id).expect("grip is present");
            assert!((handle.anchor - anchor).magnitude() < 1e-5, "{id:?} anchor");
            assert!((handle.direction - direction).magnitude() < 1e-5, "{id:?} direction");
        }
    }

    #[test]
    fn every_grips_leader_runs_back_to_the_base() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let params = CylinderParams::from_pick(base, 2.0, 5.0, skewed_plane(base));

        for handle in params.handles() {
            let HandleReach::Leader(origin) = handle.reach else {
                panic!("{:?} has no leader", handle.id);
            };
            assert!((origin - base).magnitude() < EPSILON, "{:?} leader origin", handle.id);
            assert!(
                (handle.anchor - origin).magnitude() > 1e-5,
                "{:?} leader is degenerate",
                handle.id
            );
        }
    }

    /// The radius is measured from the axis, not across the diameter, so unlike
    /// the box footprint it must not be doubled.
    #[test]
    fn both_grips_follow_the_drag_one_for_one() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let plane = skewed_plane(base);
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.5), &grabbed);
        assert!((params.radius - 3.5).abs() < 1e-5);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 1.5), &grabbed);
        assert!((params.height - 6.5).abs() < 1e-5);
    }

    #[test]
    fn a_grip_moves_only_its_own_dimension() {
        let base = Point3::new(-4.0, 5.0, 6.0);
        let plane = skewed_plane(base);
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0), &grabbed);
        assert!((params.height - grabbed.height).abs() < EPSILON);
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.plane.normal - plane.normal).magnitude() < EPSILON);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 9.0), &grabbed);
        assert!((params.radius - grabbed.radius).abs() < EPSILON);
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }

    #[test]
    fn a_grip_ignores_motion_across_its_axis() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, v) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0 + v * 5.0 + plane.normal * 5.0), &grabbed);
        assert!((params.radius - 3.0).abs() < EPSILON);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 1.0 + u * 5.0 + v * 5.0), &grabbed);
        assert!((params.height - 6.0).abs() < EPSILON);
    }

    /// Dragging a grip through the axis or past the base flattens the cylinder
    /// instead of inverting it: a negative scale would reflect the baked transform.
    #[test]
    fn a_grip_dragged_through_the_cylinder_clamps() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * -50.0), &grabbed);
        assert_eq!(params.radius, MIN_DIMENSION);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * -50.0), &grabbed);
        assert_eq!(params.height, MIN_DIMENSION);
        assert!(params.preview_transform().scale.z >= 0.0);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0), &grabbed);
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0), &grabbed);
        assert!((params.radius - 3.0).abs() < EPSILON);
    }
}
