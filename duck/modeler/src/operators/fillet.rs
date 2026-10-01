use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::{NodeFlags, NodeId, SubGeometryKind};
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, KeyEvent, Modifiers, MouseButton, NamedKey},
    operator::{
        Handle, HandleDrag, HandleEvent, HandleId, HandleReach, HandleShape, Operator,
        SelectionKinds, SelectionMode,
    },
    selection::{SelectionItem, SelectionManager},
};
use duck_engine_viewer::common::Real;

use crate::document::Document;
use crate::fillet::{
    build_fillet, execute_fillet, BlendKind, FilletFrame, FilletParams, FilletTarget,
};
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::tweak::{handle_tweak, length_field, tweak_panel, TweakAction, TweakParams};
use super::ConstructionOptions;

/// The blend's grip.
const SIZE_HANDLE: HandleId = HandleId(0);

enum Phase {
    /// No edge selected.
    AwaitingSelection,
    /// Edges are selected but nothing has been edited, so the target still
    /// follows the selection.
    Targeted(FilletTarget, FilletParams),
    /// Editing has begun: the part is locked until the blend is applied or
    /// cancelled, though its edges can still be shift-clicked in and out.
    Editing(FilletTarget, FilletParams),
}

impl Phase {
    fn target(&self) -> Option<&FilletTarget> {
        match self {
            Phase::AwaitingSelection => None,
            Phase::Targeted(target, _) | Phase::Editing(target, _) => Some(target),
        }
    }

    fn params(&self) -> Option<&FilletParams> {
        match self {
            Phase::AwaitingSelection => None,
            Phase::Targeted(_, params) | Phase::Editing(_, params) => Some(params),
        }
    }
}

impl TweakParams for FilletParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Type");
        ui.horizontal(|ui| {
            for kind in [BlendKind::Fillet, BlendKind::Chamfer] {
                changed |= ui.selectable_value(&mut self.kind, kind, kind.name()).changed();
            }
        });
        ui.end_row();

        let label = match self.kind {
            BlendKind::Fillet => "Radius",
            BlendKind::Chamfer => "Distance",
        };
        changed |= length_field(ui, label, &mut self.size, 0.0..=Real::MAX);
        changed
    }

    /// A grip on the primary edge, tied back to it by a leader. It rides out of
    /// the corner as a fillet grows and into the part as a chamfer does, and
    /// points the way it has moved.
    fn handles(&self) -> Vec<Handle> {
        let direction = if self.signed_size() < 0.0 { -self.frame.outward } else { self.frame.outward };
        vec![
            Handle::new(SIZE_HANDLE, HandleShape::Cone, self.grip())
                .with_direction(direction)
                .with_reach(HandleReach::Leader(self.frame.apex)),
        ]
    }

    /// The grip follows the cursor along the corner's bisector, one for one.
    /// Dragged back past the edge, a fillet turns into a chamfer, and back.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        if drag.id == SIZE_HANDLE {
            self.set_signed_size(grabbed.signed_size() + drag.distance_along(grabbed.frame.outward));
        }
    }
}

/// The selected edges on one part, primary first, and how many selected edges
/// on other parts are left out.
///
/// The part is `locked` once editing has begun. Until then it is the primary
/// edge's: the selection's primary if that is an edge, else the first edge
/// selected.
fn selected_target(
    selection: &SelectionManager,
    locked: Option<NodeId>,
) -> (Option<FilletTarget>, usize) {
    let as_edge = |item: &SelectionItem| match *item {
        SelectionItem::SubGeometry { node_id, element } if element.kind == SubGeometryKind::Edge => {
            Some((node_id, element.index))
        }
        _ => None,
    };
    let edges: Vec<_> = selection.iter().filter_map(as_edge).collect();
    let primary = selection.primary().as_ref().and_then(as_edge).or_else(|| edges.first().copied());
    let Some(node) = locked.or(primary.map(|(node, _)| node)) else {
        return (None, 0);
    };

    let mut on_part: Vec<u32> =
        edges.iter().filter(|(on, _)| *on == node).map(|&(_, index)| index).collect();
    let ignored = edges.len() - on_part.len();
    // The primary edge carries the grip, so it leads.
    if let Some((_, index)) = primary.filter(|(on, _)| *on == node) {
        on_part.retain(|&other| other != index);
        on_part.insert(0, index);
    }
    let target = (!on_part.is_empty()).then_some(FilletTarget { node, edges: on_part });
    (target, ignored)
}

/// "3 edges", noting any on other parts that are left out.
fn edge_summary(edges: usize, ignored: usize) -> String {
    let edges = match edges {
        1 => "1 edge".to_owned(),
        count => format!("{count} edges"),
    };
    match ignored {
        0 => edges,
        count => format!("{edges} ({count} on other parts ignored)"),
    }
}

/// Rounds or bevels selected edges of a part: one tool, whose grip makes a
/// fillet on one side of the edge and a chamfer on the other.
pub struct FilletOperator {
    phase: Phase,
    /// The selected edges the tool last acted on, so it reacts only when the
    /// selection changes.
    followed: Option<FilletTarget>,
    /// Selected edges on parts other than the target's, which the blend leaves
    /// out.
    ignored_edges: usize,
    /// Parameters as they were when the grip was grabbed; `None` when it is not
    /// held. A drag reports its total offset, so it is applied to this rather
    /// than to the live parameters.
    grabbed: Option<FilletParams>,
    /// The edges and parameters the preview was last built for.
    built: Option<(FilletTarget, FilletParams)>,
    /// Why the target or the parameters could not be built, shown in the panel.
    error: Option<String>,
    /// The previewed blend, shown in place of its part.
    preview: PreviewSession,
    /// Set once the blend is applied or cancelled, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    notifications: Notifications,
}

impl FilletOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::AwaitingSelection,
            followed: None,
            ignored_edges: 0,
            grabbed: None,
            built: None,
            error: None,
            preview,
            finished: false,
            document,
            construction_options,
            notifications,
        }
    }

    fn is_editing(&self) -> bool {
        matches!(self.phase, Phase::Editing(..))
    }

    /// Points the blend at the selected edges once they change. Editing locks
    /// the part but keeps following its edges, holding the kind and size.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        let edited = match &self.phase {
            Phase::Editing(target, params) => Some((target.node, *params)),
            _ => None,
        };
        let (target, ignored) = selected_target(selection, edited.map(|(node, _)| node));
        self.ignored_edges = ignored;
        if target == self.followed {
            return;
        }
        self.followed = target.clone();

        let Some(target) = target else {
            // The last edge went: there is nothing left to blend.
            self.abandon();
            return;
        };
        let frame = FilletFrame::new(&self.document.lock().unwrap(), target.node, target.edges[0]);
        let frame = match frame {
            Ok(frame) => frame,
            Err(e) => {
                self.abandon();
                self.error = Some(format!("{e:#}"));
                return;
            }
        };
        self.error = None;
        self.phase = match edited {
            Some((_, params)) => Phase::Editing(target, FilletParams { frame, ..params }),
            None => Phase::Targeted(target, FilletParams::new(frame)),
        };
    }

    /// Takes `params` as the blend's, which begins the edit and so locks the
    /// part.
    fn edit(&mut self, params: FilletParams) {
        if let Some(target) = self.phase.target() {
            self.phase = Phase::Editing(target.clone(), params);
        }
    }

    /// Tessellation options for the preview, which stands in for the part: the
    /// part's own, at the coarser preview tolerance.
    fn preview_options(&self, doc: &Document, node: NodeId) -> CadTessellationOptions {
        let construction = self.construction_options.borrow();
        let mut options = match doc.part_for_node(node).and_then(|part| doc.get_part(part)) {
            Some(part) => part.options().clone(),
            None => construction.geometry_options.clone(),
        };
        options.tessellation_tolerance = construction.preview_tolerance;
        options
    }

    /// Rebuilds the preview when the edges or parameters have moved on since it
    /// was last built. A build that fails keeps the last good preview and says
    /// why.
    fn refresh_preview(&mut self) {
        let Phase::Editing(target, params) = &self.phase else { return };
        if self.built.as_ref().is_some_and(|(built, with)| built == target && with == params) {
            return;
        }
        self.built = Some((target.clone(), *params));

        if params.is_degenerate() {
            // Nothing to show, and the hidden part comes back.
            self.preview.clear_previews();
            self.error = None;
            return;
        }

        let (shape, options) = {
            let doc = self.document.lock().unwrap();
            (build_fillet(&doc, target, params), self.preview_options(&doc, target.node))
        };
        let shape = match shape {
            Ok(shape) => shape,
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                return;
            }
        };
        if self.preview.try_replace_preview(&shape, &options, "Fillet preview").is_none() {
            self.error = Some("The blend could not be tessellated".to_owned());
            return;
        }
        self.error = None;
        self.preview.hide_source_node(target.node);
        // Clicks reach the hidden part beneath, whose edges shift-clicks toggle.
        self.preview.set_preview_flags(NodeFlags::DO_NOT_SELECT);
    }

    /// Commit the blend and finish the tool. A failure keeps the preview and
    /// panel so the parameters can be corrected.
    fn apply(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        let (Some(target), Some(params)) = (self.phase.target(), self.phase.params()) else {
            return Ok(());
        };
        execute_fillet(&mut self.document.lock().unwrap(), target, params)?;

        // The part the preview hid was reshaped where it stands: it comes back
        // rather than being handed over for deletion.
        self.preview.cancel();
        // The reshape renumbered its edges.
        selection.clear();
        self.reset();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active and
    /// so must report for themselves: the panel's Apply button, Enter, right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        let name = self.phase.params().map_or(BlendKind::Fillet, |params| params.kind).name();
        if let Err(e) = self.apply(selection) {
            log::error!("{name} failed: {e:#}");
            self.notifications.error(format!("{name} failed: {e:#}"));
        }
    }

    /// Abandon the blend and finish the tool, bringing back the part the
    /// preview hid.
    fn cancel(&mut self) {
        self.preview.cancel();
        self.reset();
        self.finished = true;
    }

    /// Drop the blend in progress without finishing the tool: the preview goes
    /// and its part comes back.
    fn abandon(&mut self) {
        self.preview.clear_previews();
        self.reset();
    }

    /// Forget the blend in progress.
    fn reset(&mut self) {
        self.phase = Phase::AwaitingSelection;
        self.grabbed = None;
        self.built = None;
        self.error = None;
    }

    /// Whether a left click stops here. Once editing, the grip and panel own the
    /// blend and a stray pick must not reselect anything, but shift-clicks still
    /// pass, to add or drop edges.
    fn swallows_click(&self, modifiers: Modifiers) -> bool {
        self.is_editing() && !modifiers.shift
    }

    /// Enter applies a blend being edited and Escape abandons the tool; F and C
    /// switch between fillet and chamfer.
    fn on_key(
        &mut self,
        event: &KeyEvent,
        modifiers: Modifiers,
        selection: &mut SelectionManager,
    ) -> bool {
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }
        match event.logical_key {
            Key::Named(NamedKey::Enter) if self.is_editing() => self.apply_and_report(selection),
            Key::Named(NamedKey::Escape) if self.phase.params().is_some() => self.cancel(),
            Key::Character(c) if modifiers == Modifiers::default() => {
                let kind = match c.to_ascii_lowercase() {
                    'f' => BlendKind::Fillet,
                    'c' => BlendKind::Chamfer,
                    _ => return false,
                };
                let Some(params) = self.phase.params().copied() else { return false };
                self.edit(FilletParams { kind, ..params });
            }
            _ => return false,
        }
        true
    }
}

impl ModelingTool for FilletOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "fillet", icon: icons::FILLET, shortcut: Some('f') }
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.reset();
        self.followed = None;
        self.ignored_edges = 0;
        self.finished = false;
    }

    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::SubGeometry(SelectionKinds::EDGE)
    }

    /// A blend being edited is a finished one, so leaving the tool commits it.
    /// A zero-size one is nothing to commit.
    fn finalize(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        match &self.phase {
            Phase::Editing(_, params) if !params.is_degenerate() => self.apply(selection),
            _ => Ok(()),
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn handles(&self) -> Vec<Handle> {
        self.phase.params().map(TweakParams::handles).unwrap_or_default()
    }

    /// Grabbing the grip begins the edit, which locks the part.
    fn on_handle(&mut self, event: &HandleEvent) {
        let Some(params) = self.phase.params().copied() else { return };
        if let HandleEvent::Begin(_) = event {
            self.edit(params);
        }
        if let Some(edited) = handle_tweak(params, &mut self.grabbed, event) {
            self.edit(edited);
        }
    }

    fn panel_title(&self) -> Option<&str> {
        let kind = self.phase.params().map_or(BlendKind::Fillet, |params| params.kind);
        Some(kind.name())
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let (Some(target), Some(mut params)) = (self.phase.target(), self.phase.params().copied())
        else {
            if self.error.is_none() {
                ui.label("Select edges to fillet or chamfer.");
            }
            return;
        };
        ui.label(edge_summary(target.edges.len(), self.ignored_edges));

        match tweak_panel(ui, &mut params) {
            // An edit begins the operation, which locks the part.
            TweakAction::Changed => self.edit(params),
            TweakAction::Apply => self.apply_and_report(panel.selection),
            TweakAction::Cancel => self.cancel(),
            TweakAction::None => {}
        }
    }
}

impl Operator for FilletOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::Update { .. } => {
                self.follow_selection(ctx.selection);
                self.refresh_preview();
                false
            }
            DeviceEvent::MouseClick { button: MouseButton::Left, .. } => {
                self.swallows_click(ctx.modifiers)
            }
            // Right-click finalizes, matching the Boolean/Line convention.
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } if self.is_editing() => {
                self.apply_and_report(ctx.selection);
                true
            }
            DeviceEvent::KeyboardInput { event, .. } => {
                self.on_key(event, ctx.modifiers, ctx.selection)
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Fillet"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::{InnerSpace, Point3, Vector3};
    use duck_engine_scene::resource::{SubGeometryElement, Visibility};
    use duck_engine_scene::Scene;
    use duck_engine_viewer::input::PhysicalKey;
    use opencascade::primitives::Shape;

    const EPSILON: Real = 1e-5;

    /// A document holding a 2×2×2 box centred on the origin, and a unit cube
    /// off to one side.
    fn document_with_boxes() -> (Arc<Mutex<Document>>, NodeId, NodeId) {
        let mut doc = Document::new(Scene::default());
        let options = CadTessellationOptions::default();
        let mut add = |name: &str, shape: Shape| {
            let part = doc.add_part(name, shape, &options).expect("box tessellates");
            doc.node_for_part(part).expect("part has a node")
        };
        let main = add("box", Shape::box_centered(2.0, 2.0, 2.0));
        let other = add(
            "other",
            Shape::box_from_corners(glam::dvec3(3.0, 3.0, 3.0), glam::dvec3(4.0, 4.0, 4.0)),
        );
        (Arc::new(Mutex::new(doc)), main, other)
    }

    fn operator(document: &Arc<Mutex<Document>>) -> FilletOperator {
        let construction = Rc::new(RefCell::new(ConstructionOptions::new()));
        FilletOperator::new(construction, Arc::clone(document), Notifications::default())
    }

    fn edge(node: NodeId, index: u32) -> SelectionItem {
        SelectionItem::SubGeometry {
            node_id: node,
            element: SubGeometryElement::new(SubGeometryKind::Edge, index),
        }
    }

    /// An operator targeting edge 0 of the main box alone.
    fn targeting_edge(document: &Arc<Mutex<Document>>, node: NodeId) -> (FilletOperator, SelectionManager) {
        let mut op = operator(document);
        let mut selection = SelectionManager::new();
        selection.set(edge(node, 0));
        op.follow_selection(&selection);
        (op, selection)
    }

    fn params(op: &FilletOperator) -> FilletParams {
        *op.phase.params().expect("a blend is targeted")
    }

    fn edges(op: &FilletOperator) -> Vec<u32> {
        op.phase.target().expect("a blend is targeted").edges.clone()
    }

    /// A drag of `offset` on the grip, as the handle machinery reports it.
    fn drag(grab: Point3, offset: Vector3) -> HandleDrag {
        HandleDrag { id: SIZE_HANDLE, grab, point: grab + offset, modifiers: Modifiers::default() }
    }

    /// Grabs the grip and drags it `distance` out of the corner, then lets go.
    fn drag_out(op: &mut FilletOperator, distance: Real) {
        let grabbed = params(op);
        op.on_handle(&HandleEvent::Begin(SIZE_HANDLE));
        op.on_handle(&HandleEvent::Drag(drag(grabbed.grip(), grabbed.frame.outward * distance)));
        op.on_handle(&HandleEvent::End(SIZE_HANDLE));
    }

    fn key(c: char) -> KeyEvent {
        KeyEvent {
            physical_key: PhysicalKey::Unidentified,
            logical_key: Key::Character(c),
            state: ElementState::Pressed,
            repeat: false,
        }
    }

    fn visibility(document: &Arc<Mutex<Document>>, node: NodeId) -> Visibility {
        let scene = document.lock().unwrap().scene().clone();
        let scene = scene.lock();
        scene.get_node(node).expect("node exists").visibility()
    }

    fn volume(document: &Arc<Mutex<Document>>, node: NodeId) -> f64 {
        let doc = document.lock().unwrap();
        doc.get_part(doc.part_for_node(node).expect("node is a part")).unwrap().shape.volume()
    }

    #[test]
    fn the_grip_sits_out_of_the_corner_by_the_signed_size() {
        let (document, node, _) = document_with_boxes();
        let (op, _) = targeting_edge(&document, node);
        let fillet = FilletParams { size: 0.5, ..params(&op) };
        let (apex, outward) = (fillet.frame.apex, fillet.frame.outward);

        let [grip] = <[Handle; 1]>::try_from(fillet.handles()).expect("one grip");
        assert!((grip.anchor - (apex + outward * 0.5)).magnitude() < EPSILON);
        assert!((grip.direction - outward).magnitude() < EPSILON);
        let HandleReach::Leader(origin) = grip.reach else { panic!("the grip has no leader") };
        assert!((origin - apex).magnitude() < EPSILON);

        let chamfer = FilletParams { kind: BlendKind::Chamfer, ..fillet };
        let [grip] = <[Handle; 1]>::try_from(chamfer.handles()).expect("one grip");
        assert!((grip.anchor - (apex - outward * 0.5)).magnitude() < EPSILON);
        assert!((grip.direction + outward).magnitude() < EPSILON, "a chamfer's grip points into the part");
    }

    #[test]
    fn dragging_out_of_the_corner_grows_a_fillet_one_for_one() {
        let (document, node, _) = document_with_boxes();
        let (op, _) = targeting_edge(&document, node);
        let grabbed = FilletParams { size: 0.2, ..params(&op) };

        let mut edited = grabbed;
        edited.apply_handle(&drag(grabbed.grip(), grabbed.frame.outward * 0.3), &grabbed);
        assert_eq!(edited.kind, BlendKind::Fillet);
        assert!((edited.size - 0.5).abs() < EPSILON);
    }

    #[test]
    fn dragging_back_past_the_edge_turns_a_fillet_into_a_chamfer() {
        let (document, node, _) = document_with_boxes();
        let (op, _) = targeting_edge(&document, node);
        let grabbed = FilletParams { size: 0.2, ..params(&op) };
        let outward = grabbed.frame.outward;

        let mut edited = grabbed;
        edited.apply_handle(&drag(grabbed.grip(), outward * -0.5), &grabbed);
        assert_eq!(edited.kind, BlendKind::Chamfer);
        assert!((edited.size - 0.3).abs() < EPSILON);

        // And back out again.
        edited.apply_handle(&drag(grabbed.grip(), outward * 0.1), &grabbed);
        assert_eq!(edited.kind, BlendKind::Fillet);
        assert!((edited.size - 0.3).abs() < EPSILON);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let (document, node, _) = document_with_boxes();
        let (op, _) = targeting_edge(&document, node);
        let grabbed = FilletParams { size: 0.2, ..params(&op) };
        let step = drag(grabbed.grip(), grabbed.frame.outward * 0.1);

        let mut edited = grabbed;
        edited.apply_handle(&step, &grabbed);
        edited.apply_handle(&step, &grabbed);
        assert!((edited.size - 0.3).abs() < EPSILON);
    }

    /// Until something is edited, the blend follows the selected edges.
    #[test]
    fn an_unedited_blend_follows_the_selection() {
        let (document, node, _) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, node);
        assert_eq!(edges(&op), [0]);

        selection.add(edge(node, 5));
        op.follow_selection(&selection);
        assert_eq!(edges(&op), [0, 5], "the primary edge leads");

        selection.set(edge(node, 3));
        op.follow_selection(&selection);
        assert_eq!(edges(&op), [3]);

        selection.clear();
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
    }

    /// One blend works on one part: edges on others are counted, not blended.
    #[test]
    fn edges_on_other_parts_are_left_out() {
        let (document, main, other) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, main);

        selection.add(edge(other, 0));
        op.follow_selection(&selection);
        assert_eq!(op.phase.target().unwrap().node, main);
        assert_eq!(edges(&op), [0]);
        assert_eq!(op.ignored_edges, 1);
        assert_eq!(edge_summary(1, op.ignored_edges), "1 edge (1 on other parts ignored)");
    }

    /// Grabbing the grip locks the part: a plain click stops at the tool, but a
    /// shift-click still reaches the selection, and the edges it toggles on the
    /// part are blended at the size already set.
    #[test]
    fn grabbing_the_grip_locks_the_part_but_shift_clicks_still_edit_its_edges() {
        let (document, main, other) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, main);
        assert!(!op.swallows_click(Modifiers::default()));

        drag_out(&mut op, 0.3);
        assert!(op.is_editing());
        assert!(op.swallows_click(Modifiers::default()));
        assert!(!op.swallows_click(Modifiers { shift: true, ..Default::default() }));

        selection.add(edge(main, 5));
        op.follow_selection(&selection);
        assert_eq!(edges(&op), [0, 5]);
        assert!((params(&op).size - 0.3).abs() < EPSILON, "the edit keeps its size");
        assert!(op.is_editing());

        selection.add(edge(other, 0));
        op.follow_selection(&selection);
        assert_eq!(op.phase.target().unwrap().node, main, "the part stays locked");
        assert_eq!(edges(&op), [0, 5]);
    }

    #[test]
    fn dropping_the_last_edge_drops_the_blend_and_restores_the_part() {
        let (document, node, _) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();
        assert_eq!(visibility(&document, node), Visibility::Invisible);

        selection.clear();
        op.follow_selection(&selection);
        assert!(matches!(op.phase, Phase::AwaitingSelection));
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&document, node), Visibility::Visible);
    }

    /// A grip drag builds the blend and shows it in place of its part.
    #[test]
    fn a_grip_drag_previews_the_blend_in_place_of_its_part() {
        let (document, node, _) = document_with_boxes();
        let (mut op, _) = targeting_edge(&document, node);

        drag_out(&mut op, 0.3);
        op.refresh_preview();
        assert!(op.error.is_none(), "{:?}", op.error);
        assert_eq!(op.preview.preview_nodes().len(), 1);
        assert_eq!(visibility(&document, node), Visibility::Invisible);
    }

    #[test]
    fn an_oversized_blend_keeps_the_last_preview_and_says_why() {
        let (document, node, _) = document_with_boxes();
        let (mut op, _) = targeting_edge(&document, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();
        let shown = op.preview.preview_nodes().to_vec();

        drag_out(&mut op, 5.0);
        op.refresh_preview();
        let error = op.error.as_deref().expect("the oversized blend is reported");
        assert!(error.contains("too large"), "got {error}");
        assert_eq!(op.preview.preview_nodes(), shown, "the last good preview stays");
    }

    #[test]
    fn apply_reshapes_the_part_in_place_and_finishes() {
        let (document, node, _) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();

        op.apply(&mut selection).expect("the fillet applies");
        assert!(op.is_finished());
        assert!(selection.is_empty(), "the part's edges were renumbered");
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&document, node), Visibility::Visible);
        let expected = 8.0 - 2.0 * 0.09 * (1.0 - std::f64::consts::FRAC_PI_4);
        assert!((volume(&document, node) - expected).abs() < 1e-5, "got {}", volume(&document, node));
    }

    #[test]
    fn cancel_restores_the_part() {
        let (document, node, _) = document_with_boxes();
        let (mut op, _) = targeting_edge(&document, node);
        drag_out(&mut op, 0.3);
        op.refresh_preview();

        op.cancel();
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert_eq!(visibility(&document, node), Visibility::Visible);
        assert!((volume(&document, node) - 8.0).abs() < 1e-9);
    }

    /// Leaving the tool commits an edit, but a zero-size one is nothing to commit.
    #[test]
    fn finalize_commits_only_a_blend_with_size() {
        let (document, node, _) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, node);
        op.on_handle(&HandleEvent::Begin(SIZE_HANDLE));
        op.finalize(&mut selection).expect("nothing to commit");
        assert!((volume(&document, node) - 8.0).abs() < 1e-9);

        drag_out(&mut op, 0.3);
        op.finalize(&mut selection).expect("the edit commits");
        assert!(volume(&document, node) < 8.0);
    }

    #[test]
    fn the_f_and_c_keys_switch_the_kind() {
        let (document, node, _) = document_with_boxes();
        let (mut op, mut selection) = targeting_edge(&document, node);

        assert!(op.on_key(&key('c'), Modifiers::default(), &mut selection));
        assert_eq!(params(&op).kind, BlendKind::Chamfer);
        assert_eq!(op.panel_title(), Some("Chamfer"));
        assert!(op.is_editing(), "switching kind is an edit");

        assert!(op.on_key(&key('f'), Modifiers::default(), &mut selection));
        assert_eq!(params(&op).kind, BlendKind::Fillet);
        assert_eq!(op.panel_title(), Some("Fillet"));

        let control = Modifiers { control: true, ..Default::default() };
        assert!(!op.on_key(&key('c'), control, &mut selection), "a chord is someone else's");
    }
}
