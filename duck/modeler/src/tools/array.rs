use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::{
    common::{InnerSpace, LengthDisplay, Point3, Transform, Vector3},
    event::{Event, EventContext},
    operator::{Handle, HandleDrag, HandleEvent, HandleId, HandleReach, HandleShape, SelectionKinds},
    selection::{SelectionItem, SelectionManager},
};

use crate::document::{Document, PartId};
use crate::ops::array::{rectangular_array, ArrayAxis, ArrayMode, RectangularArray};
use crate::preview::PreviewSession;
use crate::ui::icons;
use super::edit::{dimension_field, grip_dimension, Edit, PanelAction, Params, MIN_DIMENSION};
use super::targets::selected_parts;
use super::transform::parts_frame;
use super::{Gesture, ModelingTool, PanelContext, Pointer, ToolInfo, Workspace};

const NAME: &str = "Rectangular array";

/// Most members a panel field or arrow key gives one axis.
const MAX_COUNT: u32 = 100;

/// The grip at the end of each axis's reach.
const AXIS_HANDLES: [HandleId; 2] = [HandleId(0), HandleId(1)];

/// Copies the selected parts onto a grid of one or two directions.
///
/// Each click picks where a direction reaches from the selection's centre: to
/// the last member in Extent mode, to the next in Spacing mode. After the first
/// direction, Enter or right-click applies a row; a second click adds a second
/// direction, and from then the panel and grips edit the array until it is
/// applied. Up and Down add and remove members along the direction being
/// picked, or the second once both are. Escape discards the array.
///
/// The mode and counts carry over to the next use.
pub struct RectangularArrayTool {
    workspace: Workspace,
    edit: Edit<Layout>,
    phase: Phase,
    /// While picking, the snapped point under the cursor.
    hovered: Option<Point3>,
    preview: ArrayPreview,
    /// Set once the array is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Picking where this axis reaches.
    Picking(usize),
    /// Both axes picked; the panel and grips edit the array.
    Placed,
}

impl RectangularArrayTool {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
            edit: Edit::new(Layout::default()),
            phase: Phase::Picking(0),
            hovered: None,
            preview: ArrayPreview::new(workspace),
            finished: false,
        }
    }

    /// Whether a plain click picks a direction.
    fn is_picking(&self) -> bool {
        matches!(self.phase, Phase::Picking(_)) && !self.preview.sources.is_empty()
    }

    /// Whether the first direction is picked, so there is an array to apply.
    fn has_array(&self) -> bool {
        self.phase != Phase::Picking(0)
    }

    /// The layout on show: while picking, the one a click under the cursor
    /// would pick.
    fn shown(&self) -> Layout {
        let edited = *self.edit.params();
        match (self.phase, self.hovered) {
            (Phase::Picking(axis), Some(point)) => edited.reaching(axis, point).unwrap_or(edited),
            _ => edited,
        }
    }

    /// Takes the parts the selection holds as the sources. A change of sources
    /// moves the pivot, and emptying them drops the picks.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        let document = self.workspace.document.lock().unwrap();
        let sources = selected_parts(selection, &document);
        if sources == self.preview.sources {
            return;
        }
        let primary = selection.primary().map(|item| item.node_id());
        let pivot = parts_frame(&sources, primary, &document).map(|frame| frame.pivot);
        drop(document);

        if sources.is_empty() {
            self.restart();
        }
        if let Some(pivot) = pivot {
            self.edit.set(Layout { pivot, ..*self.edit.params() });
        }
        self.preview.set_sources(sources);
    }

    /// Shows the copies where the layout on show puts them.
    fn refresh_preview(&mut self) {
        self.preview.show(self.shown().array.offsets());
    }

    /// Follows `pointer` with the cursor, and the preview with what a click
    /// there would pick.
    fn hover(&mut self, pointer: &Pointer) {
        self.hovered = pointer.snap.map(|snap| snap.position);
    }

    /// Takes the pick at `pointer`. One that would leave its axis without reach
    /// takes nothing.
    fn take_pick(&mut self, pointer: &Pointer) {
        let Phase::Picking(axis) = self.phase else { return };
        let Some(snap) = pointer.snap else { return };
        let Some(picked) = self.edit.params().reaching(axis, snap.position) else { return };
        self.edit.set(picked);
        self.phase = if axis == 0 { Phase::Picking(1) } else { Phase::Placed };
        self.hovered = None;
    }

    /// One more or fewer member, by `by`, along the axis being picked, or the
    /// second once both are. Returns whether there was an array to step.
    fn step(&mut self, by: i32) -> bool {
        if self.preview.sources.is_empty() {
            return false;
        }
        let (axis, least) = match self.phase {
            // A picked direction has at least two members.
            Phase::Picking(axis) => (axis, 2),
            Phase::Placed => (1, 1),
        };
        let mut layout = *self.edit.params();
        let count = &mut layout.array.axes[axis].count;
        *count = ((*count).max(least) as i32 + by).clamp(least as i32, MAX_COUNT as i32) as u32;
        self.edit.set(layout);
        true
    }

    /// Applies the array once its first direction is picked. Returns whether
    /// there was one to apply.
    fn finish(&mut self, selection: &mut SelectionManager) -> bool {
        if !self.has_array() {
            return false;
        }
        self.apply_and_report(selection);
        true
    }

    /// Copies the sources to the array's members as one undo step, selects
    /// the whole array and finishes the tool. A failure keeps the tool open.
    fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        if !self.has_array() {
            return Ok(());
        }
        let layout = *self.edit.params();
        if layout.is_degenerate() {
            bail!("The array makes no copies");
        }
        let members = {
            let mut document = self.workspace.document.lock().unwrap();
            let parts: Vec<PartId> =
                self.preview.sources.iter().filter_map(|&node| document.part_for_node(node)).collect();
            let copies = rectangular_array(&mut document, &parts, &layout.array)?;
            let copies = copies.iter().filter_map(|&part| document.node_for_part(part));
            self.preview.sources.iter().copied().chain(copies).collect::<Vec<_>>()
        };
        self.preview.set_sources(Vec::new());
        selection.clear();
        selection.extend(members.into_iter().map(SelectionItem::Node));
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

    /// Drops the picks, keeping the mode and counts.
    fn restart(&mut self) {
        let mut layout = *self.edit.params();
        layout.array.axes = layout.array.axes.map(|axis| ArrayAxis::new(axis.count));
        self.edit = Edit::new(layout);
        self.phase = Phase::Picking(0);
        self.hovered = None;
    }

    /// What to do next, if anything needs saying.
    fn prompt(&self) -> Option<&'static str> {
        if self.preview.sources.is_empty() {
            return Some("Select parts to array.");
        }
        match (self.phase, self.edit.params().array.mode) {
            (Phase::Picking(0), ArrayMode::Extent) => Some("Click where the last copy goes."),
            (Phase::Picking(0), ArrayMode::Spacing) => Some("Click where the next copy goes."),
            (Phase::Picking(_), _) => Some("Click a second direction, or press Enter to finish."),
            (Phase::Placed, _) => None,
        }
    }
}

impl ModelingTool for RectangularArrayTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "rectangular_array", icon: icons::RECTANGULAR_ARRAY, shortcut: None }
    }

    /// Whole parts only — there is nothing to copy below part granularity.
    fn selects(&self) -> Option<SelectionKinds> {
        Some(SelectionKinds::NODE)
    }

    /// While picking, the cursor snaps and a plain click picks; once placed, a
    /// plain click stops at the tool. Shift-clicks always reach the selection.
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
            Gesture::Click { modifiers, .. } => self.phase == Phase::Placed && !modifiers.shift,
            Gesture::Finish => self.finish(ctx.selection),
            Gesture::Cancel => {
                self.cancel();
                true
            }
            Gesture::Step(by) => self.step(by),
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
        if self.is_picking() { self.hovered } else { None }
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

    /// An array with its first direction picked is a finished one, so leaving
    /// the tool applies it. One that makes no copies is nothing to apply.
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

/// The array, and the point its reaches run from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// The centre of the sources.
    pivot: Point3,
    array: RectangularArray,
}

impl Default for Layout {
    /// A row of three, not yet picked.
    fn default() -> Self {
        let array =
            RectangularArray { mode: ArrayMode::default(), axes: [ArrayAxis::new(3), ArrayAxis::new(1)] };
        Self { pivot: Point3::new(0.0, 0.0, 0.0), array }
    }
}

impl Layout {
    /// This layout with `axis` reaching to `point`, and at least two members
    /// along it; `None` where `point` is on the pivot.
    fn reaching(&self, axis: usize, point: Point3) -> Option<Self> {
        let reach = point - self.pivot;
        if reach.magnitude() <= MIN_DIMENSION {
            return None;
        }
        let mut layout = *self;
        let picked = &mut layout.array.axes[axis];
        *picked = ArrayAxis { count: picked.count.max(2), reach };
        Some(layout)
    }

    /// The axes that reach anywhere, by index.
    fn picked(&self) -> impl Iterator<Item = (usize, ArrayAxis)> + '_ {
        self.array.axes.iter().copied().enumerate().filter(|(_, axis)| axis.reach.magnitude() > 0.0)
    }
}

impl Params for Layout {
    fn ui(&mut self, ui: &mut egui::Ui, units: LengthDisplay) -> bool {
        let mut changed = false;
        ui.label("Mode");
        ui.horizontal(|ui| {
            for (mode, label) in [(ArrayMode::Extent, "Extent"), (ArrayMode::Spacing, "Spacing")] {
                let selected = self.array.mode == mode;
                if ui.selectable_label(selected, label).clicked() && !selected {
                    self.array.set_mode(mode);
                    changed = true;
                }
            }
        });
        ui.end_row();

        for (index, axis) in self.array.axes.iter_mut().enumerate() {
            let n = index + 1;
            let mut distance = axis.reach.magnitude();
            if distance > 0.0 && dimension_field(ui, &format!("Distance {n}"), &mut distance, units) {
                axis.reach = axis.reach.normalize_to(distance);
                changed = true;
            }
            ui.label(format!("Number {n}"));
            changed |= ui.add(egui::DragValue::new(&mut axis.count).speed(0.1).range(1..=MAX_COUNT)).changed();
            ui.end_row();
        }
        changed
    }

    /// A ball where each picked direction reaches, tied back to the pivot.
    fn handles(&self) -> Vec<Handle> {
        self.picked()
            .map(|(index, axis)| {
                Handle::new(AXIS_HANDLES[index], HandleShape::Ball, self.pivot + axis.reach)
                    .with_direction(axis.reach.normalize())
                    .with_reach(HandleReach::Leader(self.pivot))
            })
            .collect()
    }

    /// The ball slides along its direction, which keeps its heading.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        let Some(index) = AXIS_HANDLES.iter().position(|&id| id == drag.id) else { return };
        let reach = grabbed.array.axes[index].reach;
        if reach.magnitude() <= 0.0 {
            return;
        }
        let direction = reach.normalize();
        let length = grip_dimension(reach.magnitude(), drag.distance_along(direction));
        self.array.axes[index].reach = direction * length;
    }

    fn is_degenerate(&self) -> bool {
        self.array.offsets().is_empty()
    }
}

/// Copies of the sources previewed at an array's members. Each draws its
/// source's own mesh, so a change of layout tessellates nothing.
struct ArrayPreview {
    session: PreviewSession,
    document: Arc<Mutex<Document>>,
    /// Part-backed selected nodes, in selection order.
    sources: Vec<NodeId>,
    /// The members the copies stand at, as offsets from their sources.
    offsets: Vec<Vector3>,
    /// Each copy, where its source rests, and the index of its member.
    copies: Vec<(NodeId, Transform, usize)>,
}

impl ArrayPreview {
    fn new(workspace: &Workspace) -> Self {
        Self {
            session: workspace.preview_session(),
            document: Arc::clone(&workspace.document),
            sources: Vec::new(),
            offsets: Vec::new(),
            copies: Vec::new(),
        }
    }

    /// The copy nodes, to keep out of snapping.
    fn nodes(&self) -> &[NodeId] {
        self.session.preview_nodes()
    }

    /// Copies `sources` from now on, dropping the copies shown so far.
    fn set_sources(&mut self, sources: Vec<NodeId>) {
        self.session.cancel();
        self.copies.clear();
        self.offsets.clear();
        self.sources = sources;
    }

    /// Shows a copy of every source at each of `offsets`: the copies shown
    /// already move there if there are as many, else they are made afresh.
    fn show(&mut self, offsets: Vec<Vector3>) {
        if offsets == self.offsets {
            return;
        }
        if offsets.len() == self.offsets.len() {
            let scene = self.document.lock().unwrap().scene().clone();
            let mut scene = scene.lock();
            for &(copy, rest, member) in &self.copies {
                scene.set_node_transform(copy, moved(rest, offsets[member]));
            }
        } else {
            self.respawn(&offsets);
        }
        self.offsets = offsets;
    }

    fn respawn(&mut self, offsets: &[Vector3]) {
        self.session.cancel();
        self.copies.clear();
        let rests: Vec<(NodeId, Transform)> = {
            let scene = self.document.lock().unwrap().scene().clone();
            let scene = scene.lock();
            self.sources.iter().filter_map(|&source| Some((source, scene.get_node(source)?.transform()))).collect()
        };
        for (source, rest) in rests {
            for (member, &offset) in offsets.iter().enumerate() {
                match self.session.add_node_copy(source, moved(rest, offset)) {
                    Some(copy) => self.copies.push((copy, rest, member)),
                    None => log::warn!("Node {source:?} has no geometry to copy"),
                }
            }
        }
    }
}

/// `rest` carried by `offset`. Parts are top-level, so their nodes' own
/// transforms are in world space.
fn moved(rest: Transform, offset: Vector3) -> Transform {
    Transform { position: rest.position + offset, ..rest }
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::common::Real;

    use super::*;
    use crate::testing::{pointer_at, workspace_with_box};

    const EPSILON: Real = 1e-6;

    /// The tool on a document holding a 2×2×2 box centred on the origin, with
    /// the box selected.
    fn tool() -> (RectangularArrayTool, Workspace, SelectionManager, NodeId) {
        let (ws, node) = workspace_with_box();
        let mut tool = RectangularArrayTool::new(&ws);
        let mut selection = SelectionManager::new();
        selection.extend([SelectionItem::Node(node)]);
        tool.activate();
        tool.follow_selection(&selection);
        (tool, ws, selection, node)
    }

    fn pick(tool: &mut RectangularArrayTool, at: Point3) {
        tool.hover(&pointer_at(at));
        tool.take_pick(&pointer_at(at));
    }

    /// Where every part but `source` sits, by its frame.
    fn copy_positions(ws: &Workspace, source: NodeId) -> Vec<Point3> {
        let document = ws.document.lock().unwrap();
        let source = document.part_for_node(source).unwrap();
        document.parts().filter(|part| part.id != source).map(|part| part.frame().position).collect()
    }

    fn assert_points(actual: &[Point3], expected: &[Point3]) {
        let close = actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(a, e)| (a - e).magnitude() < EPSILON);
        assert!(close, "expected {expected:?}, got {actual:?}");
    }

    fn part_count(ws: &Workspace) -> usize {
        ws.document.lock().unwrap().parts().count()
    }

    #[test]
    fn a_row_reaches_its_last_member_in_extent_mode() {
        let (mut tool, ws, mut selection, node) = tool();
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));

        assert!(tool.finish(&mut selection));

        assert_points(&copy_positions(&ws, node), &[Point3::new(5.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)]);
        assert!(tool.is_finished());
        assert_eq!(selection.iter().count(), 3, "the whole array is selected");
        assert!(selection.iter().any(|item| item.node_id() == node));
    }

    #[test]
    fn a_row_reaches_the_next_member_in_spacing_mode() {
        let (mut tool, ws, mut selection, node) = tool();
        let mut layout = *tool.edit.params();
        layout.array.mode = ArrayMode::Spacing;
        tool.edit.set(layout);
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));

        assert!(tool.finish(&mut selection));

        assert_points(&copy_positions(&ws, node), &[Point3::new(10.0, 0.0, 0.0), Point3::new(20.0, 0.0, 0.0)]);
    }

    #[test]
    fn a_second_pick_starts_its_direction_at_two_members() {
        let (mut tool, ws, mut selection, node) = tool();
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        assert_eq!(tool.edit.params().array.axes[1].count, 1);

        tool.hover(&pointer_at(Point3::new(0.0, 0.0, 4.0)));
        tool.refresh_preview();
        assert_eq!(tool.preview.nodes().len(), 5, "3 × 2 previewed, the source aside");

        tool.take_pick(&pointer_at(Point3::new(0.0, 0.0, 4.0)));
        assert_eq!(tool.phase, Phase::Placed);
        assert_eq!(tool.edit.params().array.axes[1].count, 2);
        assert!(tool.finish(&mut selection));
        assert_eq!(copy_positions(&ws, node).len(), 5);
    }

    #[test]
    fn a_second_direction_keeps_a_count_above_one() {
        let (mut tool, _ws, _selection, _node) = tool();
        let mut layout = *tool.edit.params();
        layout.array.axes[1].count = 4;
        tool.edit.set(layout);
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        pick(&mut tool, Point3::new(0.0, 0.0, 4.0));
        assert_eq!(tool.edit.params().array.axes[1].count, 4);
    }

    #[test]
    fn a_pick_on_the_pivot_takes_nothing() {
        let (mut tool, _ws, _selection, _node) = tool();
        pick(&mut tool, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(tool.phase, Phase::Picking(0));
    }

    #[test]
    fn arrows_step_the_direction_being_picked() {
        let (mut tool, _ws, _selection, _node) = tool();
        assert!(tool.step(1));
        assert_eq!(tool.edit.params().array.axes[0].count, 4);

        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        // The second direction is shown with two members while it is picked.
        assert!(tool.step(1));
        assert_eq!(tool.edit.params().array.axes[1].count, 3);
        assert!(tool.step(-1));
        assert!(tool.step(-1));
        assert_eq!(tool.edit.params().array.axes[1].count, 2, "a picked direction keeps two");
    }

    #[test]
    fn the_preview_follows_the_hover_without_respawning() {
        let (mut tool, _ws, _selection, _node) = tool();
        tool.hover(&pointer_at(Point3::new(10.0, 0.0, 0.0)));
        tool.refresh_preview();
        let first = tool.preview.nodes().to_vec();
        assert_eq!(first.len(), 2);

        tool.hover(&pointer_at(Point3::new(0.0, 6.0, 0.0)));
        tool.refresh_preview();
        assert_eq!(tool.preview.nodes(), first.as_slice(), "the copies only move");
        let scene = tool.workspace.document.lock().unwrap().scene().clone();
        let position = scene.lock().get_node(first[1]).unwrap().transform().position;
        assert!((position - Point3::new(0.0, 6.0, 0.0)).magnitude() < EPSILON, "{position:?}");
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
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        tool.refresh_preview();

        tool.cancel();

        assert!(tool.is_finished());
        assert!(tool.preview.nodes().is_empty());
        assert_eq!(part_count(&ws), 1);
    }

    #[test]
    fn leaving_the_tool_applies_only_a_picked_array() {
        let (mut tool, ws, mut selection, _node) = tool();
        tool.finalize(&mut selection).expect("nothing to apply");
        assert_eq!(part_count(&ws), 1);

        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        tool.finalize(&mut selection).expect("the row is applied");
        assert_eq!(part_count(&ws), 3);
    }

    #[test]
    fn the_mode_and_counts_outlast_the_picks() {
        let (mut tool, _ws, _selection, _node) = tool();
        tool.step(1);
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        tool.deactivate();

        let layout = tool.edit.params();
        assert_eq!(layout.array.axes[0].count, 4);
        assert!(layout.picked().next().is_none());
        assert_eq!(tool.phase, Phase::Picking(0));
    }

    #[test]
    fn grips_slide_each_direction_along_itself() {
        let (mut tool, _ws, _selection, _node) = tool();
        pick(&mut tool, Point3::new(10.0, 0.0, 0.0));
        let handles = tool.handles();
        assert_eq!(handles.len(), 1);
        assert!((handles[0].anchor - Point3::new(10.0, 0.0, 0.0)).magnitude() < EPSILON);

        let grab = handles[0].anchor;
        let grabbed = *tool.edit.params();
        let mut layout = grabbed;
        layout.apply_handle(&crate::testing::drag(AXIS_HANDLES[0], grab, Vector3::new(2.0, 3.0, 0.0)), &grabbed);
        assert!((layout.array.axes[0].reach - Vector3::new(12.0, 0.0, 0.0)).magnitude() < EPSILON);
    }
}
