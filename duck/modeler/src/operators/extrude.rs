use anyhow::Result;
use duck_engine_common::{consts, InnerSpace, Plane, Point3, Real, Vector3};
use duck_engine_scene::resource::{NodeId, SubGeometryKind};
use duck_engine_viewer::{
    operator::{
        DragKind, Handle, HandleDrag, HandleId, HandleReach, HandleShape, SelectionKinds,
        SelectionMode,
    },
    selection::{SelectionItem, SelectionManager},
};
use opencascade::primitives::Shape;

use crate::document::{Document, SourceFate};
use crate::extrude::{build_extrusion, execute_extrude, ExtrudeFrame, ExtrudeParams, ExtrudeTarget};
use crate::tool::ToolInfo;
use crate::ui::icons;
use super::targeted::{EditLock, PreviewStyle, TargetedOp, TargetedTool};
use super::tweak::{angle_field, length_field, TweakParams};
use super::ConstructionOptions;

/// The extrusion's grips.
const DISTANCE_HANDLE: HandleId = HandleId(0);
const DRAFT_HANDLE: HandleId = HandleId(1);
const DIRECTION_HANDLE: HandleId = HandleId(2);
const THICKNESS_HANDLE: HandleId = HandleId(3);

/// Steepest draft either way: short of the right angle at which the walls
/// would lie flat.
const MAX_DRAFT: Real = 85.0 * consts::PI / 180.0;

/// Furthest the direction may lean from the profile normal: short of the right
/// angle at which the sweep would lie in the profile's own plane.
const MAX_TILT: Real = 85.0 * consts::PI / 180.0;

/// A tilt below this reads as straight out of the profile.
const TILT_EPSILON: Real = 1e-4;

impl TweakParams for ExtrudeParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let min_distance = self.min_distance();
        let mut changed = length_field(ui, "Distance", &mut self.distance, min_distance..=Real::MAX);
        changed |= angle_field(ui, "Draft", &mut self.draft, MAX_DRAFT);
        changed |= length_field(ui, "Thickness", &mut self.thickness, Real::MIN..=Real::MAX);

        ui.label("Direction");
        ui.horizontal(|ui| {
            let tilt = self.tilt();
            if tilt < TILT_EPSILON {
                ui.label("Normal");
            } else {
                ui.label(format!("{:.1}° from normal", tilt.to_degrees()));
                if ui.small_button("Reset").clicked() {
                    self.direction = self.frame.normal;
                    changed = true;
                }
            }
        });
        ui.end_row();

        changed
    }

    /// An arrow at the tip, tied back to the origin by a leader that reads as
    /// the extrusion's axis. Once there is a length to shape, a ring around the
    /// tip sets the draft, a quad halfway up the axis tilts it, and a grip on
    /// an arm across the base sets the wall thickness.
    fn handles(&self) -> Vec<Handle> {
        let origin = self.frame.origin;
        // The arrow points the way the extrusion grows, which a negative
        // distance reverses.
        let growth = self.direction * self.distance.signum();
        let mut handles = vec![
            Handle::new(DISTANCE_HANDLE, HandleShape::Cone, self.tip())
                .with_direction(growth)
                .with_reach(HandleReach::Leader(origin)),
        ];
        if !self.is_degenerate() {
            let across = thickness_axis(self);
            handles.extend([
                Handle::new(DRAFT_HANDLE, HandleShape::Ring, self.tip())
                    .with_direction(self.direction)
                    .with_drag(DragKind::Plane),
                Handle::new(DIRECTION_HANDLE, HandleShape::Quad, midpoint(self))
                    .with_direction(self.direction)
                    .with_drag(DragKind::Plane),
                Handle::new(THICKNESS_HANDLE, HandleShape::Cube, origin + across * self.thickness)
                    .with_direction(across)
                    .with_reach(HandleReach::Arm),
            ]);
        }
        handles
    }

    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        match drag.id {
            // The tip follows the cursor along the axis, one for one.
            DISTANCE_HANDLE => {
                let distance = grabbed.distance + drag.distance_along(grabbed.direction);
                self.distance = distance.max(grabbed.min_distance());
            }
            // Turning the ring about the axis adds the swept angle to the draft.
            DRAFT_HANDLE => {
                let swept = drag.angle_about(grabbed.tip(), grabbed.direction);
                self.draft = (grabbed.draft + swept).clamp(-MAX_DRAFT, MAX_DRAFT);
            }
            // The thickness follows its grip across the profile, one for one.
            THICKNESS_HANDLE => {
                self.thickness = grabbed.thickness + drag.distance_along(thickness_axis(grabbed));
            }
            // Swing the axis through the dragged midpoint, so the quad stays
            // under the cursor. The length is untouched.
            DIRECTION_HANDLE if !grabbed.is_degenerate() => {
                let lean = midpoint(grabbed) + drag.delta() - grabbed.frame.origin;
                let direction = lean.normalize() * grabbed.distance.signum();
                self.direction = limit_tilt(direction, grabbed.frame.normal);
            }
            _ => {}
        }
    }
}

/// The middle of the extrusion's axis, where the direction grip rides.
fn midpoint(params: &ExtrudeParams) -> Point3 {
    params.frame.origin + params.direction * (0.5 * params.distance)
}

/// The axis in the profile's plane the thickness grip slides along.
fn thickness_axis(params: &ExtrudeParams) -> Vector3 {
    Plane::from_point(params.frame.normal, params.frame.origin).basis().0
}

/// `direction`, swung back toward `normal` if it leans further than
/// [`MAX_TILT`] from it.
fn limit_tilt(direction: Vector3, normal: Vector3) -> Vector3 {
    let along = direction.dot(normal);
    if along >= MAX_TILT.cos() {
        return direction;
    }
    let across = direction - normal * along;
    if across.magnitude2() < Real::EPSILON {
        // Straight back through the profile: there is no side to lean to.
        return normal;
    }
    normal * MAX_TILT.cos() + across.normalize() * MAX_TILT.sin()
}

/// Extrudes the selected face into a solid or edge into a face, with a draft
/// and a wall thickness.
pub type ExtrudeOperator = TargetedTool<Extrude>;

/// The extrude operation, on the primary selection. Editing locks it outright.
#[derive(Default)]
pub struct Extrude;

impl TargetedOp for Extrude {
    type Target = ExtrudeTarget;
    type Params = ExtrudeParams;

    const LOCK: EditLock = EditLock::Target;

    fn info(&self) -> ToolInfo {
        ToolInfo { id: "extrude", icon: icons::EXTRUDE, shortcut: None }
    }

    fn selection_mode(&self) -> SelectionMode {
        // Extrude operates on a face (→ solid) or an edge (→ face).
        SelectionMode::SubGeometry(SelectionKinds::FACE | SelectionKinds::EDGE)
    }

    fn title(&self, _params: Option<&ExtrudeParams>) -> &'static str {
        "Extrude"
    }

    fn prompt(&self, _selection: &SelectionManager) -> &'static str {
        "Select a face or edge to extrude."
    }

    fn node(&self, target: &ExtrudeTarget) -> NodeId {
        target.node()
    }

    /// The face or edge chosen as the primary selection, if any.
    fn select(
        &self,
        selection: &SelectionManager,
        _locked: Option<&ExtrudeTarget>,
    ) -> (Option<ExtrudeTarget>, usize) {
        let target = match selection.primary() {
            Some(SelectionItem::SubGeometry { node_id, element }) => match element.kind {
                SubGeometryKind::Face => {
                    Some(ExtrudeTarget::Face { node: node_id, face_index: element.index })
                }
                SubGeometryKind::Edge => {
                    Some(ExtrudeTarget::Edge { node: node_id, edge_index: element.index })
                }
                SubGeometryKind::Pointset => None,
            },
            _ => None,
        };
        (target, 0)
    }

    /// A locked target is never re-resolved, so there is no edit to carry over.
    fn resolve(
        &self,
        doc: &Document,
        target: &ExtrudeTarget,
        construction: &ConstructionOptions,
        _edited: Option<&ExtrudeParams>,
    ) -> Result<ExtrudeParams> {
        // Edges extrude out of the sketch (construction) plane; faces ignore this
        // and use their own normal.
        let sketch_normal = construction.construction_plane.normal;
        Ok(ExtrudeParams::new(ExtrudeFrame::new(doc, *target, sketch_normal)?))
    }

    fn is_degenerate(&self, params: &ExtrudeParams) -> bool {
        params.is_degenerate()
    }

    /// The extrusion stands in for a bare sketch. A solid source stays in view:
    /// the pad overlapping it already looks like the fused result.
    fn preview_style(&self, params: &ExtrudeParams) -> PreviewStyle {
        PreviewStyle::Alongside { hide_source: params.frame.fate == SourceFate::Replace }
    }

    fn build(&self, doc: &Document, _target: &ExtrudeTarget, params: &ExtrudeParams) -> Result<Shape> {
        build_extrusion(doc, params)
    }

    fn apply(
        &self,
        doc: &mut Document,
        _target: &ExtrudeTarget,
        params: &ExtrudeParams,
        construction: &ConstructionOptions,
    ) -> Result<()> {
        execute_extrude(doc, params, &construction.geometry_options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::resource::{NodeFlags, SubGeometryElement, Visibility};
    use duck_engine_scene::Scene;
    use duck_engine_viewer::input::Modifiers;
    use duck_engine_viewer::operator::HandleEvent;

    use crate::notifications::Notifications;
    use crate::operators::targeted::Phase;
    use crate::tool::ModelingTool;

    const EPSILON: Real = 1e-5;

    /// A document holding a 2×2×2 box centred on the origin.
    fn document_with_box() -> (Arc<Mutex<Document>>, NodeId) {
        let mut doc = Document::new(Scene::default());
        let part = doc
            .add_part("box", Shape::box_centered(2.0, 2.0, 2.0), &CadTessellationOptions::default())
            .expect("box tessellates");
        let node = doc.node_for_part(part).expect("part has a node");
        (Arc::new(Mutex::new(doc)), node)
    }

    fn operator(document: &Arc<Mutex<Document>>) -> ExtrudeOperator {
        let construction = Rc::new(RefCell::new(ConstructionOptions::new()));
        ExtrudeOperator::new(construction, Arc::clone(document), Notifications::default())
    }

    fn select_face(selection: &mut SelectionManager, node: NodeId, index: u32) {
        selection.set(SelectionItem::SubGeometry {
            node_id: node,
            element: SubGeometryElement::new(SubGeometryKind::Face, index),
        });
    }

    /// A pad of `distance` on the first face of a box.
    fn pad(distance: Real) -> ExtrudeParams {
        let (document, node) = document_with_box();
        let doc = document.lock().unwrap();
        let target = ExtrudeTarget::Face { node, face_index: 0 };
        let frame = ExtrudeFrame::new(&doc, target, Vector3::unit_y()).expect("face resolves");
        ExtrudeParams { distance, ..ExtrudeParams::new(frame) }
    }

    /// A drag of `offset` on the grip `id`, as the handle machinery reports it.
    fn drag(id: HandleId, grab: Point3, offset: Vector3) -> HandleDrag {
        HandleDrag { id, grab, point: grab + offset, modifiers: Modifiers::default() }
    }

    /// Some direction square to the pad's axis.
    fn across(params: &ExtrudeParams) -> Vector3 {
        let normal = params.frame.normal;
        let other = if normal.x.abs() < 0.9 { Vector3::unit_x() } else { Vector3::unit_y() };
        normal.cross(other).normalize()
    }

    #[test]
    fn a_zero_length_extrusion_shows_only_the_distance_arrow() {
        let params = pad(0.0);
        let handles = params.handles();
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].id, DISTANCE_HANDLE);
        assert!((handles[0].anchor - params.frame.origin).magnitude() < EPSILON);
    }

    #[test]
    fn the_arrow_sits_on_the_tip_with_a_leader_to_the_origin() {
        let params = pad(3.0);
        let handles = params.handles();
        assert_eq!(handles.len(), 4);

        let arrow = handles.iter().find(|h| h.id == DISTANCE_HANDLE).expect("arrow is present");
        assert!((arrow.anchor - params.tip()).magnitude() < EPSILON);
        assert!((arrow.direction - params.direction).magnitude() < EPSILON);
        let HandleReach::Leader(origin) = arrow.reach else { panic!("the arrow has no leader") };
        assert!((origin - params.frame.origin).magnitude() < EPSILON);

        let quad = handles.iter().find(|h| h.id == DIRECTION_HANDLE).expect("quad is present");
        assert!((quad.anchor - (params.frame.origin + params.direction * 1.5)).magnitude() < EPSILON);
        assert_eq!(quad.drag, DragKind::Plane);
    }

    /// The draft ring circles the tip in the plane square to the axis, and the
    /// thickness grip rides an arm across the base, out by the thickness.
    #[test]
    fn the_shaping_grips_surround_the_axis() {
        let params = ExtrudeParams { thickness: 0.2, ..pad(3.0) };
        let handles = params.handles();

        let ring = handles.iter().find(|h| h.id == DRAFT_HANDLE).expect("ring is present");
        assert!((ring.anchor - params.tip()).magnitude() < EPSILON);
        assert!((ring.direction - params.direction).magnitude() < EPSILON);
        assert_eq!(ring.drag, DragKind::Plane);

        let across = thickness_axis(&params);
        assert!(across.dot(params.frame.normal).abs() < EPSILON, "the grip leaves the profile plane");
        let grip = handles.iter().find(|h| h.id == THICKNESS_HANDLE).expect("grip is present");
        assert!((grip.anchor - (params.frame.origin + across * 0.2)).magnitude() < EPSILON);
        assert!((grip.direction - across).magnitude() < EPSILON);
        assert_eq!(grip.reach, HandleReach::Arm);
    }

    #[test]
    fn turning_the_ring_adds_the_swept_angle_to_the_draft() {
        let grabbed = pad(2.0);
        let (tip, axis) = (grabbed.tip(), grabbed.direction);
        let from = thickness_axis(&grabbed);
        let toward = |angle: Real| {
            let sideways = axis.cross(from);
            tip + from * angle.cos() + sideways * angle.sin()
        };
        let turn = |angle: Real| HandleDrag {
            id: DRAFT_HANDLE,
            grab: tip + from,
            point: toward(angle),
            modifiers: Modifiers::default(),
        };

        let mut params = grabbed;
        params.apply_handle(&turn(Real::to_radians(10.0)), &grabbed);
        assert!((params.draft - Real::to_radians(10.0)).abs() < 1e-4, "got {}", params.draft.to_degrees());

        params.apply_handle(&turn(Real::to_radians(-120.0)), &grabbed);
        assert_eq!(params.draft, -MAX_DRAFT, "the draft is held short of flat");
    }

    #[test]
    fn the_thickness_follows_its_grip_one_for_one() {
        let grabbed = ExtrudeParams { thickness: 0.2, ..pad(2.0) };
        let across = thickness_axis(&grabbed);
        let grab = grabbed.frame.origin + across * 0.2;

        let mut params = grabbed;
        params.apply_handle(&drag(THICKNESS_HANDLE, grab, across * 0.3), &grabbed);
        assert!((params.thickness - 0.5).abs() < EPSILON);

        // Past zero it grows the walls on the other side.
        params.apply_handle(&drag(THICKNESS_HANDLE, grab, across * -0.5), &grabbed);
        assert!((params.thickness + 0.3).abs() < EPSILON);
    }

    #[test]
    fn the_distance_follows_the_arrow_one_for_one() {
        let grabbed = pad(1.0);
        let mut params = grabbed;
        params.apply_handle(&drag(DISTANCE_HANDLE, grabbed.tip(), grabbed.direction * 1.5), &grabbed);
        assert!((params.distance - 2.5).abs() < EPSILON);
    }

    #[test]
    fn a_pad_stops_at_its_face() {
        let grabbed = pad(1.0);
        let mut params = grabbed;
        params.apply_handle(&drag(DISTANCE_HANDLE, grabbed.tip(), grabbed.direction * -5.0), &grabbed);
        assert_eq!(params.distance, 0.0);
    }

    /// Dragging the quad sideways leans the axis toward the drag and leaves the
    /// length alone; the quad lands back under the cursor.
    #[test]
    fn the_quad_leans_the_axis_toward_the_drag() {
        let grabbed = pad(2.0);
        let side = across(&grabbed);
        let mut params = grabbed;
        params.apply_handle(&drag(DIRECTION_HANDLE, midpoint(&grabbed), side * 1.0), &grabbed);

        assert!((params.distance - grabbed.distance).abs() < EPSILON);
        assert!((params.direction.magnitude() - 1.0).abs() < EPSILON);
        // Midpoint one unit up the normal, dragged one unit across: 45°.
        assert!((params.tilt() - consts::FRAC_PI_4).abs() < 1e-4);
        assert!(params.direction.dot(side) > 0.0, "leaned away from the drag");
        let expected = grabbed.frame.origin + (grabbed.frame.normal + side).normalize();
        assert!((midpoint(&params) - expected).magnitude() < 1e-4);
    }

    #[test]
    fn the_lean_is_held_short_of_the_profile_plane() {
        let grabbed = pad(2.0);
        let mut params = grabbed;
        params.apply_handle(&drag(DIRECTION_HANDLE, midpoint(&grabbed), across(&grabbed) * 1e4), &grabbed);
        assert!(params.tilt() <= MAX_TILT + 1e-4);
        assert!(params.direction.dot(params.frame.normal) > 0.0);
    }

    #[test]
    fn a_lean_straight_back_through_the_profile_snaps_to_the_normal() {
        let normal = Vector3::unit_z();
        assert!((limit_tilt(-normal, normal) - normal).magnitude() < EPSILON);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let grabbed = pad(1.0);
        let mut params = grabbed;
        let step = drag(DISTANCE_HANDLE, grabbed.tip(), grabbed.direction * 1.0);
        params.apply_handle(&step, &grabbed);
        params.apply_handle(&step, &grabbed);
        assert!((params.distance - 2.0).abs() < EPSILON);
    }

    /// Until something is edited, the extrusion follows the selection.
    #[test]
    fn an_unedited_extrusion_follows_the_selection() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();

        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        let first = op.phase.params().expect("targeted").frame.target;

        select_face(&mut selection, node, 1);
        op.follow_selection(&selection);
        let second = op.phase.params().expect("still targeted").frame.target;
        assert_ne!(first, second, "the target did not follow the selection");

        selection.clear();
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
    }

    /// Grabbing a grip begins the edit, after which the selection no longer moves
    /// the target.
    #[test]
    fn grabbing_a_grip_locks_the_target() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        assert!(!op.is_editing());

        op.on_handle(&HandleEvent::Begin(DISTANCE_HANDLE));
        assert!(op.is_editing());
        let locked = op.phase.params().expect("editing").frame.target;

        select_face(&mut selection, node, 1);
        op.follow_selection(&selection);
        assert_eq!(op.phase.params().expect("still editing").frame.target, locked);
    }

    /// An arrow dragged and released drives the preview through one rebuild.
    #[test]
    fn a_grip_drag_rebuilds_the_preview() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);

        let tip = op.phase.params().expect("targeted").tip();
        let direction = op.phase.params().expect("targeted").direction;
        op.on_handle(&HandleEvent::Begin(DISTANCE_HANDLE));
        op.on_handle(&HandleEvent::Drag(drag(DISTANCE_HANDLE, tip, direction * 2.0)));
        op.on_handle(&HandleEvent::End(DISTANCE_HANDLE));
        op.refresh_preview();

        assert!((op.phase.params().expect("editing").distance - 2.0).abs() < EPSILON);
        assert!(op.error.is_none(), "{:?}", op.error);
        assert_eq!(op.preview.preview_nodes().len(), 1);
    }

    /// Drags the arrow of an operator targeting face 0 out to `distance` and
    /// rebuilds the preview.
    fn drag_out(op: &mut ExtrudeOperator, distance: Real) {
        let (tip, direction) = {
            let params = op.phase.params().expect("targeted");
            (params.tip(), params.direction)
        };
        op.on_handle(&HandleEvent::Begin(DISTANCE_HANDLE));
        op.on_handle(&HandleEvent::Drag(drag(DISTANCE_HANDLE, tip, direction * distance)));
        op.on_handle(&HandleEvent::End(DISTANCE_HANDLE));
        op.refresh_preview();
    }

    /// Clicks pass through the preview to the parts beneath it.
    #[test]
    fn the_preview_is_not_selectable() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        drag_out(&mut op, 1.0);

        let [preview] = <[NodeId; 1]>::try_from(op.preview.preview_nodes()).expect("one preview");
        let scene = document.lock().unwrap().scene().clone();
        let flags = scene.lock().get_node(preview).expect("preview exists").flags();
        assert!(flags.contains(NodeFlags::DO_NOT_SELECT));
    }

    /// A pad fuses into its part where it stands, keeping the part's node.
    #[test]
    fn applying_a_pad_reshapes_its_part_in_place() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        drag_out(&mut op, 1.0);

        op.apply(&mut selection).expect("the pad applies");
        let doc = document.lock().unwrap();
        assert_eq!(doc.parts().count(), 1);
        let part = doc.part_for_node(node).expect("the part keeps its node");
        assert!((doc.get_part(part).unwrap().shape.volume() - 12.0).abs() < 1e-6);
        let scene = doc.scene().clone();
        assert_eq!(scene.lock().get_node(node).expect("node exists").visibility(), Visibility::Visible);
    }

    /// An extrusion has one target, so once editing there is nothing for a
    /// shift-click to refine: it stops at the tool like any other click.
    #[test]
    fn editing_swallows_shift_clicks_too() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 0);
        op.follow_selection(&selection);
        let shift = Modifiers { shift: true, ..Default::default() };
        assert!(!op.swallows_click(shift));

        op.on_handle(&HandleEvent::Begin(DISTANCE_HANDLE));
        assert!(op.swallows_click(Modifiers::default()));
        assert!(op.swallows_click(shift));
    }

    /// A target that can't be resolved says why in the panel.
    #[test]
    fn an_unresolvable_target_is_reported() {
        let (document, node) = document_with_box();
        let mut op = operator(&document);
        let mut selection = SelectionManager::new();
        select_face(&mut selection, node, 99);
        op.follow_selection(&selection);

        assert!(matches!(op.phase, Phase::AwaitingSelection));
        let error = op.error.as_deref().expect("the failure is reported");
        assert!(error.contains("not part of a known CAD part"), "got {error}");
    }
}
