use anyhow::{Context, Result};
use duck_engine_scene::cad::{tessellate_into_with_materials, CadTessellationOptions, ShapeElements};
use duck_engine_scene::common::{Real, RgbaColor, Transform};
use duck_engine_scene::resource::{FaceMaterial, LineMaterial, NodeFlags, NodeId};
use duck_engine_viewer::{
    event::{Event, EventContext},
    operator::SelectionMode,
    selection::{SelectionItem, SelectionManager},
};
use opencascade::primitives::{Compound, Edge, Shape, ShapeType};

use crate::document::{Document, DOCUMENT_UNITS};
use crate::ops::join::{
    build_join, can_unjoin, commit_join, default_tolerance, is_sheet, unjoin_parts, Join,
};
use crate::preview::PreviewSession;
use crate::tools::{Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};
use crate::ui::{icons, length_value};
use super::edit::{apply_row, error_line, PanelAction};
use super::targets::{count_summary, selected_parts};

/// Sews the selected sheets together along the edges they share: into a
/// solid where they close, a shell where they don't.
///
/// The joined parts preview live, with the edges left open marked in amber.
/// Enter, right-click, Apply or switching tools applies once something joins;
/// Escape or Cancel discards it. Either leaves the tool.
pub struct JoinTool {
    workspace: Workspace,
    preview: PreviewSession,
    /// The tolerance set in the panel, or `None` to follow the selection's
    /// default.
    tolerance: Option<Real>,
    /// The parts and tolerance the preview was last built for, or `None`
    /// while no parts are selected.
    previewed: Option<(Vec<NodeId>, Real)>,
    /// The join the preview shows, committed as it is on apply.
    join: Option<Join>,
    /// Why the preview could not be built, shown in the panel.
    error: Option<String>,
    /// Set once the join is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl JoinTool {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
            preview: workspace.preview_session(),
            tolerance: None,
            previewed: None,
            join: None,
            error: None,
            finished: false,
        }
    }

    /// Acts on `gesture`, returning whether it was consumed.
    fn on_gesture(&mut self, gesture: Gesture, selection: &mut SelectionManager) -> bool {
        match gesture {
            Gesture::Frame => {
                self.follow_selection(selection);
                false
            }
            Gesture::Finish if self.join.is_some() => {
                self.apply_and_report(selection);
                true
            }
            Gesture::Cancel => {
                self.cancel();
                true
            }
            Gesture::Finish | Gesture::Hover(_) | Gesture::Click { .. } | Gesture::Key(_) => false,
        }
    }

    /// The tolerance the join sews `nodes` with.
    fn tolerance_for(&self, doc: &Document, nodes: &[NodeId]) -> Real {
        self.tolerance.unwrap_or_else(|| default_tolerance(doc, nodes))
    }

    /// Previews the join of the selected parts, once they or the tolerance
    /// change.
    fn follow_selection(&mut self, selection: &SelectionManager) {
        let designated = {
            let doc = self.workspace.document.lock().unwrap();
            let nodes = selected_parts(selection, &doc);
            let tolerance = self.tolerance_for(&doc, &nodes);
            (!nodes.is_empty()).then_some((nodes, tolerance))
        };
        if designated == self.previewed {
            return;
        }
        self.previewed = designated;

        // Drop the old preview, showing again the parts it hid.
        self.preview.clear_previews();
        self.join = None;
        self.error = None;
        let Some((nodes, tolerance)) = self.previewed.clone() else { return };
        if let Err(e) = self.show(&nodes, tolerance) {
            self.preview.clear_previews();
            self.error = Some(format!("{e:#}"));
        }
    }

    /// Previews the join of `nodes`: the joined parts in place of those they
    /// consume, and the edges left open.
    fn show(&mut self, nodes: &[NodeId], tolerance: Real) -> Result<()> {
        let join = build_join(&self.workspace.document.lock().unwrap(), nodes, tolerance)?;
        let options = self.workspace.preview_options();
        for group in &join.groups {
            self.preview
                .add_preview_from_shape(&group.shape, &options, "Join preview")
                .context("The join could not be tessellated")?;
        }
        for node in join.consumed() {
            self.preview.hide_source_node(node);
        }
        self.show_open_edges(&join.open_edges, &options);
        // No preview is a part: picks must reach the parts beneath.
        self.preview.set_preview_flags(NodeFlags::DO_NOT_SELECT);
        self.join = Some(join);
        Ok(())
    }

    /// Marks `edges` in amber.
    fn show_open_edges(&mut self, edges: &[Edge], options: &CadTessellationOptions) {
        if edges.is_empty() {
            return;
        }
        let scene = self.workspace.document.lock().unwrap().scene().clone();
        let (face, line) = {
            let mut scene = scene.lock();
            (scene.add_face_material(FaceMaterial::new()), scene.add_line_material(open_edge_material()))
        };
        let edges: Shape = Compound::from_shapes(edges.iter().map(Shape::from)).into();
        match tessellate_into_with_materials(
            &ShapeElements::new(&edges),
            &scene,
            options,
            &face,
            &line,
            Transform::IDENTITY,
            None,
            Some("Open edges"),
        ) {
            Ok(node) => self.preview.add_preview_node(node.id()),
            Err(e) => log::warn!("Open edges could not be tessellated: {e}"),
        }
    }

    /// Commits the join previewed and finishes the tool. A failure keeps the
    /// preview so the selection can be corrected.
    fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        let Some(join) = &self.join else { return Ok(()) };
        commit_join(&mut self.workspace.document.lock().unwrap(), join)?;
        // The parts the preview hid were consumed by the join.
        let _ = self.preview.commit();
        selection.clear();
        self.previewed = None;
        self.join = None;
        self.error = None;
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply(selection) {
            self.workspace.notifications.failure("Join", &e);
        }
    }

    /// Discards the join and finishes the tool, showing again the parts the
    /// preview hid.
    fn cancel(&mut self) {
        self.preview.cancel();
        self.previewed = None;
        self.join = None;
        self.error = None;
        self.finished = true;
    }

    /// The tolerance row of the panel's grid.
    fn tolerance_field(&mut self, ui: &mut egui::Ui, nodes: &[NodeId]) {
        let mut tolerance = self.tolerance_for(&self.workspace.document.lock().unwrap(), nodes);
        let speed = tolerance * 0.05;
        let units = self.workspace.display_units();
        ui.label("Tolerance");
        // Tolerances span orders of magnitude, so they read in scientific notation.
        let field = length_value(&mut tolerance, units)
            .speed(speed)
            .range(MIN_TOLERANCE..=Real::MAX)
            .custom_formatter(move |value, _| {
                format!("{:.2e}", DOCUMENT_UNITS.to_unit(value, units.unit))
            });
        if ui.add(field).on_hover_text("How far apart edges may lie and still be sewn").changed() {
            self.tolerance = Some(tolerance);
        }
        ui.end_row();
    }
}

/// The smallest tolerance the panel accepts.
const MIN_TOLERANCE: Real = 1e-9;

/// Amber, for the edges a join leaves open.
fn open_edge_material() -> LineMaterial {
    LineMaterial::new(RgbaColor { r: 1.0, g: 0.62, b: 0.1, a: 1.0 })
}

/// "Makes 1 solid", "Makes 2 shells, 4 open edges".
fn outcome(join: &Join) -> String {
    let count = |shape_type| join.groups.iter().filter(|group| group.shape.shape_type() == shape_type).count();
    let made: Vec<_> = [(ShapeType::Solid, "solid"), (ShapeType::Shell, "shell"), (ShapeType::Face, "face")]
        .into_iter()
        .filter_map(|(shape_type, noun)| match count(shape_type) {
            0 => None,
            count => Some(count_summary(noun, count, 0)),
        })
        .collect();
    match join.open_edges.len() {
        0 => format!("Makes {}", made.join(" and ")),
        open => format!("Makes {}, {}", made.join(" and "), count_summary("open edge", open, 0)),
    }
}

/// The selected parts, one row each with a button to deselect it and `note`
/// after its name.
fn part_rows(
    ui: &mut egui::Ui,
    workspace: &Workspace,
    selection: &mut SelectionManager,
    nodes: &[NodeId],
    note: impl Fn(&Document, NodeId) -> String,
) {
    for &node in nodes {
        let note = note(&workspace.document.lock().unwrap(), node);
        let name = workspace.part_name(node);
        ui.horizontal(|ui| {
            ui.label(name);
            ui.weak(note);
            if ui.small_button("×").clicked() {
                selection.remove(&SelectionItem::Node(node));
            }
        });
    }
}

impl ModelingTool for JoinTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "join", icon: icons::JOIN, shortcut: Some('j') }
    }

    /// Whole parts.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        self.on_gesture(gesture, ctx.selection)
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Join")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        let nodes = selected_parts(panel.selection, &self.workspace.document.lock().unwrap());
        ui.label("Sheets");
        if nodes.is_empty() {
            ui.label("(click sheets, shift-click to add more)");
        }
        part_rows(ui, &self.workspace, panel.selection, &nodes, |doc, node| {
            match doc.part_at(node).is_some_and(is_sheet) {
                true => String::new(),
                false => "not a sheet, ignored".to_owned(),
            }
        });
        ui.separator();

        egui::Grid::new("join_settings").num_columns(2).show(ui, |ui| self.tolerance_field(ui, &nodes));
        if let Some(error) = &self.error {
            error_line(ui, error);
        } else if let Some(join) = &self.join {
            ui.label(outcome(join));
        }
        ui.separator();

        match apply_row(ui) {
            PanelAction::Apply => self.apply_and_report(panel.selection),
            PanelAction::Cancel => self.cancel(),
            PanelAction::None | PanelAction::Changed => {}
        }
    }

    /// A join that sews anything is complete, so leaving the tool applies it.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        self.apply(selection)
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.tolerance = None;
        self.previewed = None;
        self.join = None;
        self.error = None;
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Breaks the selected solids and shells into their faces, a part for each.
///
/// Enter, right-click, Apply or switching tools applies once a selected part
/// has faces to unjoin; Escape or Cancel leaves without.
pub struct UnjoinTool {
    workspace: Workspace,
    /// Set once the parts are unjoined or the tool is cancelled, so it cedes
    /// back to selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl UnjoinTool {
    pub fn new(workspace: &Workspace) -> Self {
        Self { workspace: workspace.clone(), finished: false }
    }

    /// Acts on `gesture`, returning whether it was consumed.
    fn on_gesture(&mut self, gesture: Gesture, selection: &mut SelectionManager) -> bool {
        match gesture {
            Gesture::Finish if self.can_apply(selection) => {
                self.apply_and_report(selection);
                true
            }
            Gesture::Cancel => {
                self.finished = true;
                true
            }
            Gesture::Frame | Gesture::Finish | Gesture::Hover(_) | Gesture::Click { .. } | Gesture::Key(_) => {
                false
            }
        }
    }

    /// Whether a selected part has faces to unjoin.
    fn can_apply(&self, selection: &SelectionManager) -> bool {
        let doc = self.workspace.document.lock().unwrap();
        selected_parts(selection, &doc).into_iter().any(|node| doc.part_at(node).is_some_and(can_unjoin))
    }

    /// Unjoins the selected parts and finishes the tool. A failure keeps the
    /// tool open.
    fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        if !self.can_apply(selection) {
            return Ok(());
        }
        {
            let mut doc = self.workspace.document.lock().unwrap();
            let nodes = selected_parts(selection, &doc);
            unjoin_parts(&mut doc, &nodes)?;
        }
        selection.clear();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply(selection) {
            self.workspace.notifications.failure("Unjoin", &e);
        }
    }
}

impl ModelingTool for UnjoinTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "unjoin", icon: icons::UNJOIN, shortcut: None }
    }

    /// Whole parts.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        self.on_gesture(gesture, ctx.selection)
    }

    fn panel_title(&self) -> Option<&str> {
        Some("Unjoin")
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        let nodes = selected_parts(panel.selection, &self.workspace.document.lock().unwrap());
        ui.label("Parts");
        if nodes.is_empty() {
            ui.label("(click solids or shells, shift-click to add more)");
        }
        part_rows(ui, &self.workspace, panel.selection, &nodes, |doc, node| match doc.part_at(node) {
            Some(part) if can_unjoin(part) => format!("→ {}", count_summary("face", part.elements().faces().len(), 0)),
            Some(_) => "nothing to unjoin".to_owned(),
            None => String::new(),
        });
        ui.separator();

        match apply_row(ui) {
            PanelAction::Apply => self.apply_and_report(panel.selection),
            PanelAction::Cancel => self.finished = true,
            PanelAction::None | PanelAction::Changed => {}
        }
    }

    /// A part with faces to unjoin is complete, so leaving the tool unjoins it.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        self.apply(selection)
    }

    fn deactivate(&mut self) {
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::resource::Visibility;

    use crate::document::PartKind;
    use crate::testing::{doc_with_box, doc_with_box_faces, visibility, workspace};

    fn part_count(ws: &Workspace) -> usize {
        ws.document.lock().unwrap().parts().count()
    }

    fn select(nodes: &[NodeId]) -> SelectionManager {
        let mut selection = SelectionManager::new();
        selection.extend(nodes.iter().map(|&node| SelectionItem::Node(node)));
        selection
    }

    /// A join tool on a box's six faces, all selected.
    fn join_tool() -> (JoinTool, Workspace, SelectionManager, Vec<NodeId>) {
        let (doc, nodes) = doc_with_box_faces();
        let ws = workspace(doc);
        (JoinTool::new(&ws), ws.clone(), select(&nodes), nodes)
    }

    #[test]
    fn finishing_joins_the_sheets_and_leaves() {
        let (mut op, ws, mut selection, _) = join_tool();
        op.on_gesture(Gesture::Frame, &mut selection);
        assert_eq!(op.join.as_ref().map(outcome).as_deref(), Some("Makes 1 solid"));

        assert!(op.on_gesture(Gesture::Finish, &mut selection));
        assert!(op.is_finished());
        assert!(op.preview.is_empty());
        assert!(selection.is_empty());
        let doc = ws.document.lock().unwrap();
        assert_eq!(doc.parts().map(|part| part.kind()).collect::<Vec<_>>(), [PartKind::Solid]);
    }

    /// The joined preview stands in for the faces it consumes, and the edges
    /// left open are marked beside it.
    #[test]
    fn the_preview_hides_what_it_joins_and_marks_open_edges() {
        let (mut op, ws, mut selection, nodes) = join_tool();
        let left_out = nodes[5];
        selection.remove(&SelectionItem::Node(left_out));
        op.on_gesture(Gesture::Frame, &mut selection);

        assert_eq!(op.join.as_ref().map(outcome).as_deref(), Some("Makes 1 shell, 4 open edges"));
        assert_eq!(op.preview.preview_nodes().len(), 2, "the shell and its open edges");
        for &node in &nodes[..5] {
            assert_eq!(visibility(&ws, node), Visibility::Invisible);
        }
        assert_eq!(visibility(&ws, left_out), Visibility::Visible);
    }

    #[test]
    fn cancelling_shows_the_sheets_again_and_leaves() {
        let (mut op, ws, mut selection, nodes) = join_tool();
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(op.on_gesture(Gesture::Cancel, &mut selection));
        assert!(op.is_finished());
        for node in nodes {
            assert_eq!(visibility(&ws, node), Visibility::Visible);
        }
        assert_eq!(part_count(&ws), 6);
    }

    /// Sheets that meet nowhere say so, and leave nothing to finish.
    #[test]
    fn nothing_to_join_says_why() {
        let (mut op, ws, _, nodes) = join_tool();
        let mut selection = select(&nodes[..1]);
        op.on_gesture(Gesture::Frame, &mut selection);

        assert!(op.error.as_deref().is_some_and(|error| error.contains("Nothing to join")));
        assert!(!op.on_gesture(Gesture::Finish, &mut selection));
        assert!(!op.is_finished());
        assert_eq!(part_count(&ws), 6);
    }

    #[test]
    fn editing_the_tolerance_rebuilds_the_preview() {
        let (mut op, _, mut selection, _) = join_tool();
        op.on_gesture(Gesture::Frame, &mut selection);
        let before = op.previewed.clone();

        op.tolerance = Some(1e-3);
        op.on_gesture(Gesture::Frame, &mut selection);
        assert_ne!(op.previewed, before);
        assert_eq!(op.previewed.as_ref().map(|(_, tolerance)| *tolerance), Some(1e-3));
        assert!(op.join.is_some());
    }

    #[test]
    fn finishing_unjoins_the_parts_and_leaves() {
        let (doc, node) = doc_with_box();
        let ws = workspace(doc);
        let mut op = UnjoinTool::new(&ws);
        let mut selection = select(&[node]);

        assert!(op.on_gesture(Gesture::Finish, &mut selection));
        assert!(op.is_finished());
        assert!(selection.is_empty());
        let doc = ws.document.lock().unwrap();
        assert_eq!(doc.parts().map(|part| part.kind()).collect::<Vec<_>>(), [PartKind::Face; 6]);
    }

    /// A lone face has nothing to unjoin, so there is nothing to finish.
    #[test]
    fn a_face_leaves_unjoin_nothing_to_finish() {
        let (doc, nodes) = doc_with_box_faces();
        let ws = workspace(doc);
        let mut op = UnjoinTool::new(&ws);
        let mut selection = select(&nodes[..1]);

        assert!(!op.on_gesture(Gesture::Finish, &mut selection));
        assert!(!op.is_finished());
        assert_eq!(part_count(&ws), 6);
    }
}
