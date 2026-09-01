use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{InnerSpace, Plane, Point3, Ray, Vector3};
use duck_engine_scene::resource::Visibility;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, Modifiers, MouseButton, NamedKey},
    operator::Operator,
    selection::SelectionManager,
};
use glam::dvec3;
use log::{error, warn};
use opencascade::primitives::{Face, Shape, Wire};

use crate::document::Document;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::tweak::{commit_tweak, dimension_field, tweak_panel, TweakAction, TweakParams};
use super::ConstructionOptions;

/// A dimension at or below this is degenerate: the preview is hidden and the pick
/// can't be committed.
const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BoxAction {
    Place,
    /// End the operation: commit if the box is fully defined, else abort.
    Finish,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the footprint. Preview is a flat rectangle face.
    /// Base being defined. `plane` is the plane the base rectangle is being defined on.
    Base { center: Point3, plane: Plane },
    /// Footprint fixed; the cursor drives the height. Preview is the 3D box.
    Height { center: Point3, width: f32, depth: f32, plane: Plane },
    /// Every point picked; the options panel drives the dimensions until the
    /// box is applied or cancelled.
    Tweak(BoxParams),
}

/// The dimensions of a placed box. `base` is
/// the first point picked and never moves: the footprint grows about it and the
/// height grows away from it along `plane.normal`.
#[derive(Clone, Copy)]
pub(super) struct BoxParams {
    base: Point3,
    plane: Plane,
    width: f32,
    depth: f32,
    height: f32,
}

impl BoxParams {
    /// Parameters for a finished pick, normalized so the height is positive and
    /// grows away from `base`. A downward pick flips the plane normal rather
    /// than moving the base off the picked point, so later height edits move
    /// only the far face.
    fn from_pick(base: Point3, width: f32, depth: f32, height: f32, plane: Plane) -> Self {
        let (plane, height) = if height >= 0.0 {
            (plane, height)
        } else {
            (Plane::from_point(-plane.normal, base), -height)
        };
        Self { base, plane, width, depth, height }
    }

    /// The footprint rectangle in the plane's basis: the vector from base to the
    /// footprint center, and the half extents along the basis vectors `(u, v)`.
    ///
    /// The only anchor-dependent part of the box. A corner-anchored box returns
    /// `u * width/2 + v * depth/2` as the offset here and needs no other change:
    /// both the preview transform and the committed shape read the rectangle
    /// from this one place.
    fn local_rect(&self) -> (Vector3, f32, f32) {
        // As we only support center boxes at the moment, this is all that is needed
        (Vector3::new(0.0, 0.0, 0.0), 0.5 * self.width, 0.5 * self.depth)
    }

    /// The footprint's four world-space corners, in wire order.
    fn footprint_corners(&self) -> [Point3; 4] {
        let (u, v) = self.plane.basis();
        let (offset, half_width, half_depth) = self.local_rect();
        let centre = self.base + offset;
        let half_w = u * half_width;
        let half_d = v * half_depth;
        [
            centre - half_w - half_d,
            centre + half_w - half_d,
            centre + half_w + half_d,
            centre - half_w + half_d,
        ]
    }
}

impl TweakParams for BoxParams {
    const NAME: &'static str = "Box";

    /// Scales the unit reference box (footprint in local XY, height along local
    /// +Z) to these dimensions. Every scale component stays
    /// non-negative — a negative one would make the baked GTransform a
    /// reflection, flipping the box's face normals inward.
    fn preview_transform(&self) -> Transform {
        let (offset, _, _) = self.local_rect();
        Transform {
            position: self.base + offset,
            rotation: self.plane.rotation(),
            scale: Vector3::new(self.width, self.depth, self.height),
        }
    }

    /// World-space box with analytic planar faces: the footprint rectangle on
    /// the plane, extruded along its normal.
    fn build(&self) -> Option<Shape> {
        let wire = Wire::from_ordered_points(
            self.footprint_corners()
                .iter()
                .map(|p| dvec3(p.x as f64, p.y as f64, p.z as f64)),
        )
        .map_err(|e| warn!("Failed to build box footprint wire: {e}"))
        .ok()?;
        let face = Face::from_wire(&wire)
            .map_err(|e| warn!("Failed to build box footprint face: {e}"))
            .ok()?;
        let dir = self.plane.normal * self.height;
        Some(face.extrude(dvec3(dir.x as f64, dir.y as f64, dir.z as f64)).into())
    }

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = dimension_field(ui, "Width", &mut self.width);
        changed |= dimension_field(ui, "Length", &mut self.depth);
        changed |= dimension_field(ui, "Height", &mut self.height);
        changed
    }
}

pub struct BoxOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<BoxAction>,
    cursor_target: Option<Point3>,
    // Set once the box is applied, so the
    // tool cedes back to selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
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
                BoxAction::Finish,
            );
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            preview,
            bindings,
            cursor_target: None,
            finished: false,
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

    /// Unit reference box for the preview: footprint centered in local XY,
    /// height along local +Z (`[0, 1]`), scaled/oriented via
    /// [`BoxParams::preview_transform`].
    fn reference_box() -> Shape {
        Shape::box_from_corners(dvec3(-0.5, -0.5, 0.0), dvec3(0.5, 0.5, 1.0))
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
    fn height_from_cursor(center: Point3, plane: &Plane, position: (f32, f32), ctx: &EventContext) -> f32 {
        let camera = ctx.camera.clone();
        let ray: Ray = camera.ray_from_screen_point(position.0, position.1, ctx.size.0, ctx.size.1);
        ray.closest_param_on_axis(center, plane.normal).unwrap_or(0.0)
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &EventContext) -> bool {
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
        let camera = ctx.camera.clone();
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
        // A zero-height (degenerate) box can't be tweaked; stay in the height stage.
        if !Self::box_valid(width, depth, height) {
            return false;
        }

        // Hand the box to the options panel rather than committing it: the
        // preview stays live and the dimensions stay editable until Apply.
        let params = BoxParams::from_pick(center, width, depth, height, plane);
        self.preview.set_preview_transform(params.preview_transform());
        self.phase = Phase::Tweak(params);
        true
    }

    /// Commit the box and finish the tool. A failed build keeps the
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
            error!("Box failed: {e:#}");
        }
    }

    /// Drop the in-progress box.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
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
                            BoxParams::from_pick(center, width, depth, height, plane)
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

impl ModelingTool for BoxOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "box", icon: icons::BOX, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.finished = false;
        // The modeler hides the cursor for the (now inactive) tool, but clear our
        // target so a stale point can't flash if we're reactivated before a move.
        self.cursor_target = None;
    }

    /// A box waiting on the panel is fully defined, so leaving the tool commits it.
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

    fn panel_title(&self) -> Option<&str> {
        matches!(self.phase, Phase::Tweak(_)).then_some(BoxParams::NAME)
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
                            // Swallow the click: the panel owns the box now, so a
                            // stray pick must not select or place anything.
                            Phase::Tweak(_) => true,
                        },
                        // Right-click ends the operation: it commits a box the
                        // panel already holds, and aborts one still being picked.
                        BoxAction::Finish => match self.phase {
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
        "Box"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane aligned with no world axis, so a mistaken basis shows up.
    fn skewed_plane(origin: Point3) -> Plane {
        Plane::from_point(Vector3::new(1.0, 2.0, 3.0).normalize(), origin)
    }

    #[test]
    fn from_pick_flips_negative_height_about_the_base() {
        let base = Point3::new(1.0, 2.0, 3.0);
        let plane = Plane::xz();
        let params = BoxParams::from_pick(base, 4.0, 5.0, -6.0, plane);
        // The picked point stays the base; the normal flips instead (XZ normal is +Y).
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.height - 6.0).abs() < EPSILON);
        assert!((params.plane.normal + plane.normal).magnitude() < EPSILON);

        let t = params.preview_transform();
        // Scale stays non-negative: a negative one would reflect the baked
        // transform and invert the face normals.
        assert!(t.scale.x >= 0.0 && t.scale.y >= 0.0 && t.scale.z >= 0.0);
        assert!((t.position - base).magnitude() < EPSILON);
    }

    #[test]
    fn footprint_is_centred_on_the_base() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let params = BoxParams::from_pick(base, 4.0, 6.0, 2.0, skewed_plane(base));
        let corners = params.footprint_corners();
        let offset = corners.iter().fold(Vector3::new(0.0, 0.0, 0.0), |acc, c| acc + (c - base))
            / corners.len() as f32;
        assert!(offset.magnitude() < 1e-5);
    }

    #[test]
    fn height_edits_leave_the_footprint_in_place() {
        let base = Point3::new(-4.0, 5.0, 6.0);
        let plane = skewed_plane(base);
        let mut params = BoxParams::from_pick(base, 3.0, 7.0, 2.0, plane);
        let before = params.footprint_corners();

        params.height = 11.0;

        // Only the top face moves: base, plane and footprint are untouched.
        let after = params.footprint_corners();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).magnitude() < EPSILON);
        }
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }
}
