use anyhow::{bail, Result};
use duck_engine_viewer::{
    common::{consts, InnerSpace, LengthDisplay, Plane, Point3, Ray, Vector3, Zero},
    event::{Event, EventContext},
    operator::{Handle, HandleDrag, HandleEvent, HandleId, HandleReach, HandleShape, SelectionKinds},
    selection::SelectionManager,
};

use crate::ops::array::{radial_array, AngularAxis, ArrayAxis, ArrayMode, RadialArray};
use crate::snap::Snap;
use crate::tools::edit::{
    angle_field, dimension_field, grip_dimension, Edit, Lever, PanelAction, Params, MIN_DIMENSION,
};
use crate::tools::{Gesture, ModelingTool, PanelContext, Pointer, ToolInfo, Workspace};
use crate::ui::icons;
use super::{count_field, mode_field, ArrayPreview, Spot, MAX_COUNT};

const NAME: &str = "Radial array";

/// The grip at the end of the sweep.
const ANGLE_HANDLE: HandleId = HandleId(0);
/// The grip where the rings reach.
const RING_HANDLE: HandleId = HandleId(1);

/// Copies the selected parts round an axis, in rings stepping outward from it.
///
/// One click places the axis through the snapped point, along the direction
/// the snap carries (an edge's tangent, a circle's or face's normal, a grid
/// axis) or else square to the construction plane, and facing the viewer, so
/// a positive angle turns counter-clockwise on screen. Until then the hover
/// previews the array about the axis under the cursor. Once placed, the panel
/// and grips edit the array until it is applied; A picks the axis again, and
/// Up and Down add and remove members around it. Escape discards the array.
///
/// The mode, angle and counts carry over to the next use.
pub struct RadialArrayTool {
    workspace: Workspace,
    edit: Edit<Layout>,
    /// While picking, the axis a click under the cursor would place.
    hovered: Option<(Point3, Vector3)>,
    preview: ArrayPreview,
    /// Set once the array is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl RadialArrayTool {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
            edit: Edit::new(Layout::default()),
            hovered: None,
            preview: ArrayPreview::new(workspace),
            finished: false,
        }
    }

    /// Whether a plain click places the axis.
    fn is_picking(&self) -> bool {
        !self.edit.params().placed && !self.preview.sources.is_empty()
    }

    /// The layout on show: while picking, the one a click under the cursor
    /// would place, if any.
    fn shown(&self) -> Option<Layout> {
        let edited = *self.edit.params();
        match (edited.placed, self.hovered) {
            (true, _) => Some(edited),
            (false, Some((origin, axis))) => Some(edited.about(origin, axis)),
            (false, None) => None,
        }
    }

    /// Takes the parts the selection holds as the sources. A change of sources
    /// moves the pivot, and emptying them drops the axis.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        if !self.preview.follow(selection) {
            return;
        }
        match self.preview.spot() {
            Some(spot) => self.edit.set(self.edit.params().at(spot)),
            None => self.restart(),
        }
    }

    /// Shows the copies where the layout on show puts them.
    fn refresh_preview(&mut self) {
        self.preview.show(self.shown().map_or_else(Vec::new, |layout| layout.array.placements()));
    }

    /// Follows `pointer` with the axis a click there would place.
    fn hover(&mut self, pointer: &Pointer) {
        let plane = self.workspace.construction.borrow().construction_plane;
        self.hovered = pointer.snap.map(|snap| axis_of(&snap, &plane, &pointer.ray));
    }

    /// Places the axis at `pointer`.
    fn take_pick(&mut self, pointer: &Pointer) {
        self.hover(pointer);
        let Some((origin, axis)) = self.hovered.take() else { return };
        self.edit.set(self.edit.params().about(origin, axis));
    }

    /// Picks the axis again, keeping the rest. Returns whether there was one
    /// to pick again.
    fn repick(&mut self) -> bool {
        let layout = *self.edit.params();
        if !layout.placed {
            return false;
        }
        self.edit.set(Layout { placed: false, ..layout });
        true
    }

    /// One more or fewer member around, by `by`. Returns whether there was an
    /// array to step.
    fn step(&mut self, by: i32) -> bool {
        if self.preview.sources.is_empty() {
            return false;
        }
        let mut layout = *self.edit.params();
        let count = &mut layout.array.around.count;
        *count = (*count as i32 + by).clamp(2, MAX_COUNT as i32) as u32;
        self.edit.set(layout);
        true
    }

    /// Applies the array once its axis is placed. Returns whether there was
    /// one to apply.
    fn finish(&mut self, selection: &mut SelectionManager) -> bool {
        if !self.edit.params().placed {
            return false;
        }
        self.apply_and_report(selection);
        true
    }

    /// Copies the sources to the array's members as one undo step, selects
    /// the whole array and finishes the tool. A failure keeps the tool open.
    fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        let layout = *self.edit.params();
        if !layout.placed {
            return Ok(());
        }
        if layout.is_degenerate() {
            bail!("The array makes no copies");
        }
        self.preview.commit(selection, |document, parts| radial_array(document, parts, &layout.array))?;
        self.restart();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply(selection) {
            self.workspace.notifications.failure(NAME, &e);
        }
    }

    /// Discards the array and finishes the tool.
    fn cancel(&mut self) {
        self.preview.set_sources(Vec::new());
        self.restart();
        self.finished = true;
    }

    /// Drops the axis and the rings' reach, keeping the mode, angle and
    /// counts.
    fn restart(&mut self) {
        let mut layout = *self.edit.params();
        layout.placed = false;
        layout.array.rings = ArrayAxis::new(layout.array.rings.count);
        self.edit = Edit::new(layout);
        self.hovered = None;
    }

    /// What to do next, if anything needs saying.
    fn prompt(&self) -> Option<&'static str> {
        if self.preview.sources.is_empty() {
            return Some("Select parts to array.");
        }
        (!self.edit.params().placed).then_some(
            "Click the axis to turn about. Edges, circles, faces and grid axes give their own \
             direction; elsewhere it stands square to the plane.",
        )
    }
}

impl ModelingTool for RadialArrayTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "radial_array", icon: icons::RADIAL_ARRAY, shortcut: None }
    }

    /// Whole parts only — there is nothing to copy below part granularity.
    fn selects(&self) -> Option<SelectionKinds> {
        Some(SelectionKinds::NODE)
    }

    /// While picking, the cursor snaps and a plain click places the axis; once
    /// placed, a plain click stops at the tool. Shift-clicks always reach the
    /// selection.
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        match gesture {
            Gesture::Frame => {
                self.follow_selection(ctx.selection);
                self.refresh_preview();
                false
            }
            Gesture::Hover(at) if self.is_picking() => {
                let pointer = self.workspace.pointer(at, self.preview.nodes(), ctx);
                self.hover(&pointer);
                false
            }
            Gesture::Click { at, modifiers } if self.is_picking() && !modifiers.shift => {
                let pointer = self.workspace.pointer(at, self.preview.nodes(), ctx);
                self.take_pick(&pointer);
                true
            }
            Gesture::Click { modifiers, .. } => self.edit.params().placed && !modifiers.shift,
            Gesture::Finish => self.finish(ctx.selection),
            Gesture::Cancel => {
                self.cancel();
                true
            }
            Gesture::Step(by) => self.step(by),
            Gesture::Key('a') => self.repick(),
            Gesture::Hover(_) | Gesture::Key(_) => false,
        }
    }

    fn handles(&self) -> Vec<Handle> {
        self.edit.handles()
    }

    fn on_handle(&mut self, event: &HandleEvent, _ctx: &mut EventContext) {
        self.edit.on_handle(event);
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.hovered.filter(|_| self.is_picking()).map(|(origin, _)| origin)
    }

    fn panel_title(&self) -> Option<&str> {
        Some(NAME)
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        if let Some(prompt) = self.prompt() {
            ui.label(prompt);
        }
        match self.edit.panel(ui, self.workspace.display_units()) {
            PanelAction::Apply => self.apply_and_report(panel.selection),
            PanelAction::Cancel => self.cancel(),
            // The preview follows on the next frame.
            PanelAction::Changed | PanelAction::None => {}
        }
    }

    /// An array with its axis placed is a finished one, so leaving the tool
    /// applies it. One that makes no copies is nothing to apply.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        match self.edit.params().is_degenerate() {
            true => Ok(()),
            false => self.apply(selection),
        }
    }

    fn deactivate(&mut self) {
        self.preview.set_sources(Vec::new());
        self.restart();
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// The axis a click at `snap` places: through the snapped point, along the
/// direction the snap carries or else square to the construction `plane`,
/// and facing back along the cursor's `ray`.
fn axis_of(snap: &Snap, plane: &Plane, ray: &Ray) -> (Point3, Vector3) {
    let direction = snap.direction.filter(|d| d.magnitude2() > 0.0).unwrap_or(plane.normal).normalize();
    let facing = if direction.dot(ray.direction) > 0.0 { -direction } else { direction };
    (snap.position, facing)
}

/// The array, and where its sources sit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    spot: Spot,
    /// Whether the axis is placed. Until it is, the array's axis is a
    /// stand-in.
    placed: bool,
    array: RadialArray,
}

impl Default for Layout {
    /// Eight round a full turn, in one ring, about no axis yet.
    fn default() -> Self {
        let array = RadialArray {
            origin: Point3::new(0.0, 0.0, 0.0),
            axis: Vector3::unit_y(),
            mode: ArrayMode::default(),
            around: AngularAxis { count: 8, angle: consts::TAU },
            rings: ArrayAxis::new(1),
        };
        Self { spot: Spot { pivot: Point3::new(0.0, 0.0, 0.0), size: 0.0 }, placed: false, array }
    }
}

impl Layout {
    /// This layout about the axis through `origin` along the unit `axis`,
    /// placed. The rings reach outward through the pivot as far as they did,
    /// or at first as far as the pivot is from the axis; while the pivot is on
    /// the axis, they reach nowhere.
    fn about(&self, origin: Point3, axis: Vector3) -> Self {
        let from_axis = {
            let v = self.spot.pivot - origin;
            v - axis * v.dot(axis)
        };
        let reach = self.array.rings.reach.magnitude();
        let reach = match from_axis.magnitude() > MIN_DIMENSION {
            true if reach > 0.0 => from_axis.normalize_to(reach),
            true => from_axis,
            false => Vector3::zero(),
        };
        let rings = ArrayAxis { reach, ..self.array.rings };
        Self { placed: true, array: RadialArray { origin, axis, rings, ..self.array }, ..*self }
    }

    /// This layout with its sources at `spot`, the rings turned to reach
    /// outward through the new pivot.
    fn at(&self, spot: Spot) -> Self {
        let moved = Self { spot, ..*self };
        match moved.placed {
            true => moved.about(moved.array.origin, moved.array.axis),
            false => moved,
        }
    }

    /// The lever the angle grip swings on: out from the axis to the pivot.
    fn lever(&self) -> Lever {
        Lever::new(self.array.origin, self.array.axis, self.spot.pivot, self.spot.size)
    }
}

impl Params for Layout {
    fn ui(&mut self, ui: &mut egui::Ui, units: LengthDisplay) -> bool {
        let mut changed = false;
        if let Some(mode) = mode_field(ui, self.array.mode) {
            self.array.set_mode(mode);
            changed = true;
        }
        changed |= angle_field(ui, "Angle", &mut self.array.around.angle, consts::TAU);
        changed |= count_field(ui, "Number around", &mut self.array.around.count);

        let rings = &mut self.array.rings;
        let mut distance = rings.reach.magnitude();
        if distance > 0.0 && dimension_field(ui, "Ring distance", &mut distance, units) {
            rings.reach = rings.reach.normalize_to(distance);
            changed = true;
        }
        changed |= count_field(ui, "Rings", &mut rings.count);

        ui.label("Axis");
        ui.horizontal(|ui| match self.placed {
            false => {
                ui.label("Click to place it");
            }
            true => {
                ui.label("Placed");
                if ui.small_button("Re-pick (A)").clicked() {
                    self.placed = false;
                    changed = true;
                }
            }
        });
        ui.end_row();
        changed
    }

    /// Once the axis is placed: a ball where the sweep ends, swung about the
    /// axis from the pivot, and with two rings or more, a ball where the rings
    /// reach, tied back to the pivot.
    fn handles(&self) -> Vec<Handle> {
        if !self.placed {
            return Vec::new();
        }
        let mut handles = vec![self.lever().handle(ANGLE_HANDLE, self.array.around.angle)];
        let rings = self.array.rings;
        if rings.count > 1 && rings.reach.magnitude() > 0.0 {
            let pivot = self.spot.pivot;
            handles.push(
                Handle::new(RING_HANDLE, HandleShape::Ball, pivot + rings.reach)
                    .with_direction(rings.reach.normalize())
                    .with_reach(HandleReach::Leader(pivot)),
            );
        }
        handles
    }

    /// The angle ball sweeps the array with it, one for one, within a turn
    /// either way; the ring ball slides outward, keeping its heading.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        match drag.id {
            ANGLE_HANDLE => {
                let from = grabbed.array.around.angle;
                let swept = grabbed.lever().swept(drag, self.array.around.angle - from);
                self.array.around.angle = (from + swept).clamp(-consts::TAU, consts::TAU);
            }
            RING_HANDLE => {
                let reach = grabbed.array.rings.reach;
                if reach.magnitude() <= 0.0 {
                    return;
                }
                let direction = reach.normalize();
                let length = grip_dimension(reach.magnitude(), drag.distance_along(direction));
                self.array.rings.reach = direction * length;
            }
            _ => {}
        }
    }

    fn is_degenerate(&self) -> bool {
        !self.placed || self.array.placements().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_scene::resource::NodeId;
    use duck_engine_viewer::common::Real;
    use duck_engine_viewer::input::Modifiers;
    use duck_engine_viewer::selection::SelectionItem;

    use super::*;
    use crate::snap::SnapKind;
    use crate::testing::{drag, pointer_at, workspace_with_box};

    const EPSILON: Real = 1e-6;

    /// The tool on a document holding a 2×2×2 box centred on the origin, with
    /// the box selected.
    fn tool() -> (RadialArrayTool, Workspace, SelectionManager, NodeId) {
        let (ws, node) = workspace_with_box();
        let mut tool = RadialArrayTool::new(&ws);
        let mut selection = SelectionManager::new();
        selection.extend([SelectionItem::Node(node)]);
        tool.activate();
        tool.follow_selection(&selection);
        (tool, ws, selection, node)
    }

    /// A pick at `at` on a snap carrying `direction`, seen looking along
    /// `looking`.
    fn pointer_along(at: Point3, direction: Vector3, looking: Vector3) -> Pointer {
        Pointer {
            snap: Some(Snap { position: at, direction: Some(direction), kind: SnapKind::GridAxis }),
            ray: Ray::new(at - looking * 10.0, looking),
        }
    }

    /// Where every part but `source` sits, by its frame.
    fn copy_positions(ws: &Workspace, source: NodeId) -> Vec<Point3> {
        let document = ws.document.lock().unwrap();
        let source = document.part_for_node(source).unwrap();
        document.parts().filter(|part| part.id != source).map(|part| part.frame().position).collect()
    }

    fn contains(points: &[Point3], point: Point3) -> bool {
        points.iter().any(|p| (p - point).magnitude() < EPSILON)
    }

    fn part_count(ws: &Workspace) -> usize {
        ws.document.lock().unwrap().parts().count()
    }

    fn set(tool: &mut RadialArrayTool, change: impl FnOnce(&mut Layout)) {
        let mut layout = *tool.edit.params();
        change(&mut layout);
        tool.edit.set(layout);
    }

    #[test]
    fn a_plane_pick_turns_the_copies_about_its_normal_through_the_point() {
        let (mut tool, ws, mut selection, node) = tool();
        let centre = Point3::new(5.0, 0.0, 0.0);
        tool.take_pick(&pointer_at(centre));
        let layout = tool.edit.params();
        assert!(layout.placed);
        assert!((layout.array.axis.y.abs() - 1.0).abs() < EPSILON, "{:?}", layout.array.axis);

        assert!(tool.finish(&mut selection));

        let positions = copy_positions(&ws, node);
        assert_eq!(positions.len(), 7, "eight round a full turn, the source among them");
        let round = |p: &Point3| p.y.abs() < EPSILON && ((p - centre).magnitude() - 5.0).abs() < EPSILON;
        assert!(positions.iter().all(round), "{positions:?}");
        assert!(contains(&positions, Point3::new(10.0, 0.0, 0.0)), "half a turn round: {positions:?}");
        assert!(tool.is_finished());
        assert_eq!(selection.iter().count(), 8, "the whole array is selected");
    }

    #[test]
    fn a_snap_with_a_direction_turns_about_it() {
        let (mut tool, ..) = tool();
        let at = Point3::new(0.0, 0.0, 3.0);
        tool.take_pick(&pointer_along(at, Vector3::unit_x(), -Vector3::unit_y()));
        let array = tool.edit.params().array;
        assert!((array.origin - at).magnitude() < EPSILON);
        assert!((array.axis - Vector3::unit_x()).magnitude() < EPSILON, "{:?}", array.axis);
    }

    #[test]
    fn the_axis_faces_the_viewer() {
        let at = Point3::new(5.0, 0.0, 0.0);
        let plane = Plane::xz();
        for looking in [Vector3::unit_z(), -Vector3::unit_z()] {
            let pointer = pointer_along(at, Vector3::unit_z(), looking);
            let (_, axis) = axis_of(&pointer.snap.unwrap(), &plane, &pointer.ray);
            assert!((axis + looking).magnitude() < EPSILON, "looking along {looking:?}: {axis:?}");
        }
    }

    #[test]
    fn the_hover_previews_the_array_and_moves_it_without_respawning() {
        let (mut tool, ..) = tool();
        tool.refresh_preview();
        assert!(tool.preview.nodes().is_empty(), "no axis, no array");

        tool.hover(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        tool.refresh_preview();
        let first = tool.preview.nodes().to_vec();
        assert_eq!(first.len(), 7);

        tool.hover(&pointer_at(Point3::new(6.0, 0.0, 0.0)));
        tool.refresh_preview();
        assert_eq!(tool.preview.nodes(), first.as_slice(), "the copies only move");
        let scene = tool.workspace.document.lock().unwrap().scene().clone();
        let position = scene.lock().get_node(first[3]).unwrap().transform().position;
        assert!((position - Point3::new(12.0, 0.0, 0.0)).magnitude() < EPSILON, "{position:?}");
    }

    #[test]
    fn arrows_step_the_count_around() {
        let (mut tool, ..) = tool();
        assert!(tool.step(1));
        assert_eq!(tool.edit.params().array.around.count, 9);
        set(&mut tool, |layout| layout.array.around.count = 3);
        assert!(tool.step(-1));
        assert!(tool.step(-1));
        assert_eq!(tool.edit.params().array.around.count, 2, "two at the least");
    }

    #[test]
    fn rings_step_outward_as_far_as_the_pivot_is_from_the_axis() {
        let (mut tool, ws, mut selection, node) = tool();
        set(&mut tool, |layout| layout.array.rings.count = 2);
        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        assert!((tool.edit.params().array.rings.reach - Vector3::new(-5.0, 0.0, 0.0)).magnitude() < EPSILON);

        assert!(tool.finish(&mut selection));

        let positions = copy_positions(&ws, node);
        assert_eq!(positions.len(), 15, "eight round in each of two rings, the source among them");
        assert!(contains(&positions, Point3::new(-5.0, 0.0, 0.0)), "the outer ring: {positions:?}");
        assert!(contains(&positions, Point3::new(15.0, 0.0, 0.0)), "round the outer ring: {positions:?}");
    }

    #[test]
    fn a_pivot_on_the_axis_has_one_ring_and_still_a_grip() {
        let (mut tool, ..) = tool();
        set(&mut tool, |layout| layout.array.rings.count = 3);
        tool.take_pick(&pointer_at(Point3::new(0.0, -1.0, 0.0)));
        let layout = *tool.edit.params();
        assert_eq!(layout.array.placements().len(), 7, "the rings reach nowhere");

        let [grip] = <[Handle; 1]>::try_from(layout.handles()).expect("only the angle grip");
        let arm = Vector3::new(grip.anchor.x, 0.0, grip.anchor.z);
        assert!((arm.magnitude() - (3.0 as Real).sqrt()).abs() < EPSILON, "swung at the box's size: {arm:?}");
    }

    #[test]
    fn a_repicks_the_axis_keeping_the_rest() {
        let (mut tool, ..) = tool();
        assert!(!tool.repick(), "nothing to pick again");
        set(&mut tool, |layout| layout.array.rings.count = 2);
        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        tool.step(1);

        assert!(tool.repick());
        assert!(tool.is_picking());
        tool.take_pick(&pointer_at(Point3::new(0.0, 0.0, 7.0)));

        let array = tool.edit.params().array;
        assert!((array.origin - Point3::new(0.0, 0.0, 7.0)).magnitude() < EPSILON);
        assert_eq!(array.around.count, 9);
        assert!((array.rings.reach - Vector3::new(0.0, 0.0, -5.0)).magnitude() < EPSILON, "{:?}", array.rings.reach);
    }

    #[test]
    fn grips_come_once_the_axis_is_placed_the_ring_grip_with_rings() {
        let (mut tool, ..) = tool();
        assert!(tool.handles().is_empty());
        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        assert_eq!(tool.handles().len(), 1);

        set(&mut tool, |layout| layout.array.rings.count = 2);
        let handles = tool.handles();
        let ring = handles.iter().find(|handle| handle.id == RING_HANDLE).expect("a ring grip");
        assert!((ring.anchor - Point3::new(-5.0, 0.0, 0.0)).magnitude() < EPSILON, "{:?}", ring.anchor);

        let grabbed = *tool.edit.params();
        let mut layout = grabbed;
        layout.apply_handle(&drag(RING_HANDLE, ring.anchor, Vector3::new(-2.0, 1.0, 0.0)), &grabbed);
        assert!((layout.array.rings.reach - Vector3::new(-7.0, 0.0, 0.0)).magnitude() < EPSILON);
    }

    /// The swing wraps at half a turn, so the grip must carry the sweep on
    /// round rather than jump back.
    #[test]
    fn the_angle_grip_carries_the_sweep_on_past_half_a_turn() {
        let (mut tool, ..) = tool();
        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        let start = consts::FRAC_PI_4;
        set(&mut tool, |layout| layout.array.around.angle = start);
        let grabbed = *tool.edit.params();
        let lever = grabbed.lever();
        let [grip] = <[Handle; 1]>::try_from(grabbed.handles()).expect("the angle grip");
        assert!((grip.anchor - lever.swung(start)).magnitude() < EPSILON);

        let mut layout = grabbed;
        for degrees in [90.0, 170.0, 190.0, 270.0] {
            let to = start + Real::to_radians(degrees);
            let swing = HandleDrag {
                id: ANGLE_HANDLE,
                grab: lever.swung(start),
                point: lever.swung(to),
                modifiers: Modifiers::default(),
            };
            layout.apply_handle(&swing, &grabbed);
        }
        let expected = start + 1.5 * consts::PI;
        assert!((layout.array.around.angle - expected).abs() < EPSILON, "{}", layout.array.around.angle.to_degrees());
    }

    #[test]
    fn finishing_before_a_pick_is_left_to_others() {
        let (mut tool, ws, mut selection, _node) = tool();
        assert!(!tool.finish(&mut selection));
        assert!(!tool.is_finished());
        assert_eq!(part_count(&ws), 1);
    }

    #[test]
    fn cancelling_adds_nothing_and_leaves() {
        let (mut tool, ws, _selection, _node) = tool();
        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        tool.refresh_preview();

        tool.cancel();

        assert!(tool.is_finished());
        assert!(tool.preview.nodes().is_empty());
        assert_eq!(part_count(&ws), 1);
    }

    #[test]
    fn leaving_the_tool_applies_only_a_placed_array() {
        let (mut tool, ws, mut selection, _node) = tool();
        tool.hover(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        tool.finalize(&mut selection).expect("nothing to apply");
        assert_eq!(part_count(&ws), 1, "a hovered axis is not placed");

        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        tool.finalize(&mut selection).expect("the array is applied");
        assert_eq!(part_count(&ws), 8);
    }

    #[test]
    fn the_mode_angle_and_counts_outlast_the_axis() {
        let (mut tool, ..) = tool();
        set(&mut tool, |layout| {
            layout.array.set_mode(ArrayMode::Spacing);
            layout.array.around.angle = consts::FRAC_PI_6;
            layout.array.rings.count = 2;
        });
        tool.take_pick(&pointer_at(Point3::new(5.0, 0.0, 0.0)));
        tool.step(1);
        tool.deactivate();

        let layout = tool.edit.params();
        assert!(!layout.placed);
        assert_eq!(layout.array.mode, ArrayMode::Spacing);
        assert!((layout.array.around.angle - consts::FRAC_PI_6).abs() < EPSILON);
        assert_eq!((layout.array.around.count, layout.array.rings.count), (9, 2));
        assert_eq!(layout.array.rings.reach, Vector3::zero(), "the reach goes with the axis");
    }
}
