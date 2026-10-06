//! A driver for tools that act on what is selected and hold their result as
//! preview geometry until it is applied.
//!
//! The selection picks the target until the first edit, which locks it so a
//! stray click can't retarget the operation in progress.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::{NodeFlags, NodeId, SubGeometryKind};
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, KeyEvent, Modifiers, MouseButton, NamedKey},
    operator::{Handle, HandleEvent, Operator, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tools::{ModelingTool, PanelContext, ToolInfo};
use super::tweak::{handle_tweak, tweak_panel, TweakAction, TweakParams};
use crate::construction::ConstructionOptions;

/// How much of its target an operation holds once editing has begun.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditLock {
    /// All of it: the selection is ignored and every click stops at the tool.
    Target,
    /// Its part: shift-clicks still add and drop sub-shapes on the part.
    Part,
}

/// Where an operation's result is previewed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewStyle {
    /// In place of the target's part, which applying reshapes where it stands.
    InPlace,
    /// Beside the target's part, hidden if `hide_source`. Applying consumes a
    /// hidden part and leaves a shown one standing.
    Alongside { hide_source: bool },
}

/// The operation a [`TargetedTool`] drives.
pub trait TargetedOp: 'static {
    /// What the selection designates.
    type Target: Clone + PartialEq;
    /// The operation's settings, edited by the panel and the grips.
    type Params: TweakParams + PartialEq;

    /// How much of the target an edit locks.
    const LOCK: EditLock;

    fn info(&self) -> ToolInfo;

    fn selection_mode(&self) -> SelectionMode;

    /// Name of the operation, for the panel title and failure reports.
    fn title(&self, params: Option<&Self::Params>) -> &'static str;

    /// What to select, shown while there is no target.
    fn prompt(&self, selection: &SelectionManager) -> &'static str;

    /// The part `target` belongs to.
    fn node(&self, target: &Self::Target) -> NodeId;

    /// The target the selection designates, held to `locked`'s part while
    /// editing, and how many selected items it leaves out.
    fn select(
        &self,
        selection: &SelectionManager,
        locked: Option<&Self::Target>,
    ) -> (Option<Self::Target>, usize);

    /// A line describing the target, shown above the fields.
    fn summary(&self, _target: &Self::Target, _ignored: usize) -> Option<String> {
        None
    }

    /// Settings for `target`: fresh ones, or `edited`'s carried onto it.
    fn resolve(
        &self,
        doc: &Document,
        target: &Self::Target,
        construction: &ConstructionOptions,
        edited: Option<&Self::Params>,
    ) -> Result<Self::Params>;

    /// Whether `params` leave nothing to build.
    fn is_degenerate(&self, params: &Self::Params) -> bool;

    fn preview_style(&self, params: &Self::Params) -> PreviewStyle;

    /// The result to preview.
    fn build(&self, doc: &Document, target: &Self::Target, params: &Self::Params) -> Result<Shape>;

    /// Commit the result as one undo step.
    fn apply(
        &self,
        doc: &mut Document,
        target: &Self::Target,
        params: &Self::Params,
        construction: &ConstructionOptions,
    ) -> Result<()>;

    /// `params` as edited by the unmodified, lowercased key `c`, or `None` if
    /// `c` means nothing here.
    fn on_char(&self, _c: char, _params: &Self::Params) -> Option<Self::Params> {
        None
    }
}

pub enum Phase<T, P> {
    /// Nothing selected to act on.
    AwaitingSelection,
    /// A target is selected but nothing has been edited, so it still follows
    /// the selection.
    Targeted(T, P),
    /// Editing has begun: the target is locked, as far as [`EditLock`] says,
    /// until the result is applied or cancelled.
    Editing(T, P),
}

impl<T, P> Phase<T, P> {
    pub fn target(&self) -> Option<&T> {
        match self {
            Phase::AwaitingSelection => None,
            Phase::Targeted(target, _) | Phase::Editing(target, _) => Some(target),
        }
    }

    pub fn params(&self) -> Option<&P> {
        match self {
            Phase::AwaitingSelection => None,
            Phase::Targeted(_, params) | Phase::Editing(_, params) => Some(params),
        }
    }
}

/// A tool that runs `O` on the selection, previewing it live until applied.
pub struct TargetedTool<O: TargetedOp> {
    op: O,
    pub(super) phase: Phase<O::Target, O::Params>,
    /// The target the tool last acted on, so it reacts only when the selection
    /// changes.
    followed: Option<O::Target>,
    /// Selected items the target leaves out.
    pub(super) ignored: usize,
    /// Parameters as they were when the held grip was grabbed; `None` when no
    /// grip is held. A drag reports its total offset, so it is applied to this
    /// rather than to the live parameters.
    grabbed: Option<O::Params>,
    /// The target and parameters the preview was last built for.
    built: Option<(O::Target, O::Params)>,
    /// Why the target or the parameters could not be built, shown in the panel.
    pub(super) error: Option<String>,
    pub(super) preview: PreviewSession,
    /// Set once the result is applied or cancelled, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    notifications: Notifications,
}

impl<O: TargetedOp + Default> TargetedTool<O> {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            op: O::default(),
            phase: Phase::AwaitingSelection,
            followed: None,
            ignored: 0,
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
}

impl<O: TargetedOp> TargetedTool<O> {
    pub(super) fn is_editing(&self) -> bool {
        matches!(self.phase, Phase::Editing(..))
    }

    /// Points the operation at the selection once it changes. Editing locks
    /// the target as far as [`TargetedOp::LOCK`] says, keeping the edited
    /// parameters.
    pub(super) fn follow_selection(&mut self, selection: &SelectionManager) {
        let edited = match &self.phase {
            Phase::Editing(_, _) if O::LOCK == EditLock::Target => return,
            Phase::Editing(target, params) => Some((target.clone(), *params)),
            _ => None,
        };
        let (target, ignored) = self.op.select(selection, edited.as_ref().map(|(target, _)| target));
        self.ignored = ignored;
        if target == self.followed {
            return;
        }
        self.followed = target.clone();

        let Some(target) = target else {
            // The selection emptied: there is nothing left to act on.
            self.abandon();
            return;
        };
        let params = self.op.resolve(
            &self.document.lock().unwrap(),
            &target,
            &self.construction_options.borrow(),
            edited.as_ref().map(|(_, params)| params),
        );
        let params = match params {
            Ok(params) => params,
            Err(e) => {
                self.abandon();
                self.error = Some(format!("{e:#}"));
                return;
            }
        };
        self.error = None;
        self.phase = match edited {
            Some(_) => Phase::Editing(target, params),
            None => Phase::Targeted(target, params),
        };
    }

    /// Takes `params` as the operation's, which begins the edit and so locks
    /// the target.
    fn edit(&mut self, params: O::Params) {
        if let Some(target) = self.phase.target() {
            self.phase = Phase::Editing(target.clone(), params);
        }
    }

    /// Tessellation options for the preview: the part's own when it stands in
    /// for the part, the construction ones otherwise, both at the coarser
    /// preview tolerance.
    fn preview_options(&self, doc: &Document, node: NodeId, style: PreviewStyle) -> CadTessellationOptions {
        let construction = self.construction_options.borrow();
        let part = doc.part_for_node(node).and_then(|part| doc.get_part(part));
        let mut options = match (style, part) {
            (PreviewStyle::InPlace, Some(part)) => part.options().clone(),
            _ => construction.geometry_options.clone(),
        };
        options.tessellation_tolerance = construction.preview_tolerance;
        options
    }

    /// Rebuilds the preview when the target or parameters have moved on since
    /// it was last built. A build that fails keeps the last good preview and
    /// says why.
    pub(super) fn refresh_preview(&mut self) {
        let Phase::Editing(target, params) = &self.phase else { return };
        if self.built.as_ref().is_some_and(|(built, with)| built == target && with == params) {
            return;
        }
        self.built = Some((target.clone(), *params));

        if self.op.is_degenerate(params) {
            // Nothing to show, and anything hidden comes back.
            self.preview.clear_previews();
            self.error = None;
            return;
        }

        let node = self.op.node(target);
        let style = self.op.preview_style(params);
        let (shape, options) = {
            let doc = self.document.lock().unwrap();
            (self.op.build(&doc, target, params), self.preview_options(&doc, node, style))
        };
        let shape = match shape {
            Ok(shape) => shape,
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                return;
            }
        };
        let title = self.op.title(Some(params));
        let name = format!("{title} preview");
        if self.preview.try_replace_preview(&shape, &options, &name).is_none() {
            self.error = Some(format!("The {} could not be tessellated", title.to_lowercase()));
            return;
        }
        self.error = None;
        // Clicks reach the part the preview hides or covers, whose sub-shapes
        // shift-clicks toggle.
        self.preview.set_preview_flags(NodeFlags::DO_NOT_SELECT);
        match style {
            PreviewStyle::InPlace | PreviewStyle::Alongside { hide_source: true } => {
                self.preview.hide_source_node(node);
            }
            // A build in an earlier style may have hidden it.
            PreviewStyle::Alongside { hide_source: false } => self.preview.show_sources(),
        }
    }

    /// Commit the result and finish the tool. A failure keeps the preview and
    /// panel so the parameters can be corrected.
    pub(super) fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        let (Some(target), Some(params)) = (self.phase.target(), self.phase.params()) else {
            return Ok(());
        };
        let style = self.op.preview_style(params);
        self.op.apply(
            &mut self.document.lock().unwrap(),
            target,
            params,
            &self.construction_options.borrow(),
        )?;

        match style {
            // The hidden part was consumed by the result.
            PreviewStyle::Alongside { hide_source: true } => {
                let _ = self.preview.commit();
            }
            // The part stands, reshaped where it is or untouched beside the
            // result: it comes back, even if a build in an earlier style hid it.
            PreviewStyle::InPlace | PreviewStyle::Alongside { hide_source: false } => {
                self.preview.cancel();
            }
        }
        // The selected sub-shapes were renumbered or removed.
        selection.clear();
        self.reset();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active and
    /// so must report for themselves: the panel's Apply button, Enter, right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        let title = self.op.title(self.phase.params());
        if let Err(e) = self.apply(selection) {
            log::error!("{title} failed: {e:#}");
            self.notifications.error(format!("{title} failed: {e:#}"));
        }
    }

    /// Abandon the operation and finish the tool, restoring anything the
    /// preview hid.
    pub(super) fn cancel(&mut self) {
        self.preview.cancel();
        self.reset();
        self.finished = true;
    }

    /// Drop the operation in progress without finishing the tool: the preview
    /// goes and anything it hid comes back.
    fn abandon(&mut self) {
        self.preview.clear_previews();
        self.reset();
    }

    /// Forget the operation in progress.
    fn reset(&mut self) {
        self.phase = Phase::AwaitingSelection;
        self.grabbed = None;
        self.built = None;
        self.error = None;
    }

    /// Whether a left click stops here. Once editing, the grips and panel own
    /// the operation and a stray pick must not reselect anything; shift-clicks
    /// pass only where they refine a locked part.
    pub(super) fn swallows_click(&self, modifiers: Modifiers) -> bool {
        self.is_editing() && !(O::LOCK == EditLock::Part && modifiers.shift)
    }

    /// Enter applies an edit and Escape abandons the tool; unmodified
    /// characters go to the operation.
    pub(super) fn on_key(
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
                let Some(params) = self.phase.params() else { return false };
                let Some(edited) = self.op.on_char(c.to_ascii_lowercase(), params) else {
                    return false;
                };
                self.edit(edited);
            }
            _ => return false,
        }
        true
    }
}

impl<O: TargetedOp> ModelingTool for TargetedTool<O> {
    fn info(&self) -> ToolInfo {
        self.op.info()
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.reset();
        self.followed = None;
        self.ignored = 0;
        self.finished = false;
    }

    fn selection_mode(&self) -> SelectionMode {
        self.op.selection_mode()
    }

    /// An edit in progress is a finished one, so leaving the tool commits it.
    /// A degenerate one is nothing to commit.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        match &self.phase {
            Phase::Editing(_, params) if !self.op.is_degenerate(params) => self.apply(selection),
            _ => Ok(()),
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn handles(&self) -> Vec<Handle> {
        self.phase.params().map(TweakParams::handles).unwrap_or_default()
    }

    /// Grabbing a grip begins the edit, which locks the target.
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
        Some(self.op.title(self.phase.params()))
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let (Some(target), Some(mut params)) = (self.phase.target(), self.phase.params().copied())
        else {
            if self.error.is_none() {
                ui.label(self.op.prompt(panel.selection));
            }
            return;
        };
        if let Some(summary) = self.op.summary(target, self.ignored) {
            ui.label(summary);
        }

        match tweak_panel(ui, &mut params) {
            // An edit begins the operation, which locks the target.
            TweakAction::Changed => self.edit(params),
            TweakAction::Apply => self.apply_and_report(panel.selection),
            TweakAction::Cancel => self.cancel(),
            TweakAction::None => {}
        }
    }
}

impl<O: TargetedOp> Operator for TargetedTool<O> {
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
        self.op.title(None)
    }
}

/// The selected sub-shapes of `kind` on one part, primary first, and how many
/// on other parts are left out.
///
/// The part is `locked` once editing has begun. Until then it is the primary's:
/// the selection's primary if that is of `kind`, else the first of `kind`
/// selected.
pub fn selected_on_part(
    selection: &SelectionManager,
    kind: SubGeometryKind,
    locked: Option<NodeId>,
) -> (Option<(NodeId, Vec<u32>)>, usize) {
    let of_kind = |item: &SelectionItem| match *item {
        SelectionItem::SubGeometry { node_id, element } if element.kind == kind => {
            Some((node_id, element.index))
        }
        _ => None,
    };
    let picked: Vec<_> = selection.iter().filter_map(of_kind).collect();
    let primary = selection.primary().as_ref().and_then(of_kind).or_else(|| picked.first().copied());
    let Some(node) = locked.or(primary.map(|(node, _)| node)) else {
        return (None, 0);
    };

    let mut on_part: Vec<u32> =
        picked.iter().filter(|(on, _)| *on == node).map(|&(_, index)| index).collect();
    let ignored = picked.len() - on_part.len();
    if let Some((_, index)) = primary.filter(|(on, _)| *on == node) {
        on_part.retain(|&other| other != index);
        on_part.insert(0, index);
    }
    ((!on_part.is_empty()).then_some((node, on_part)), ignored)
}

/// The selected faces on one part, as [`selected_on_part`] picks them, or with
/// none, that part whole — no faces — if its node is selected; and how many
/// selected items on other parts are left out.
pub fn selected_faces_or_part(
    selection: &SelectionManager,
    locked: Option<NodeId>,
) -> (Option<(NodeId, Vec<u32>)>, usize) {
    let target = match selected_on_part(selection, SubGeometryKind::Face, locked) {
        (Some(faces), _) => Some(faces),
        (None, _) => locked
            .or_else(|| selection.primary().map(|item| item.node_id()))
            .filter(|&node| selection.contains(&SelectionItem::Node(node)))
            .map(|node| (node, Vec::new())),
    };
    let ignored = target.as_ref().map_or(0, |(node, _)| {
        selection.iter().filter(|item| item.node_id() != *node).count()
    });
    (target, ignored)
}

/// "3 edges", noting any on other parts that are left out.
pub fn count_summary(noun: &str, count: usize, ignored: usize) -> String {
    let counted = match count {
        1 => format!("1 {noun}"),
        count => format!("{count} {noun}s"),
    };
    match ignored {
        0 => counted,
        ignored => format!("{counted} ({ignored} on other parts ignored)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::resource::SubGeometryElement;

    fn face(node: NodeId, index: u32) -> SelectionItem {
        SelectionItem::SubGeometry { node_id: node, element: SubGeometryElement::new(SubGeometryKind::Face, index) }
    }

    fn select(items: &[SelectionItem]) -> SelectionManager {
        let mut selection = SelectionManager::new();
        selection.extend(items.iter().copied());
        selection
    }

    #[test]
    fn faces_on_one_part_are_its_target() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[face(part, 2), face(part, 0), face(other, 1)]);
        assert_eq!(selected_faces_or_part(&selection, None), (Some((part, vec![2, 0])), 1));
    }

    #[test]
    fn a_part_selected_whole_is_its_target_with_no_faces() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[SelectionItem::Node(part), SelectionItem::Node(other)]);
        assert_eq!(selected_faces_or_part(&selection, None), (Some((part, vec![])), 1));
    }

    /// Faces picked on a part selected whole say which of its faces to use.
    #[test]
    fn faces_win_over_their_own_part() {
        let part = NodeId::new();
        let selection = select(&[SelectionItem::Node(part), face(part, 3)]);
        assert_eq!(selected_faces_or_part(&selection, None), (Some((part, vec![3])), 0));
    }

    /// Faces on one part outrank another part selected whole first, which is
    /// left out.
    #[test]
    fn faces_outrank_another_part_selected_whole() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[SelectionItem::Node(other), face(part, 1)]);
        assert_eq!(selected_faces_or_part(&selection, None), (Some((part, vec![1])), 1));
    }

    #[test]
    fn a_locked_part_must_itself_be_selected() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[SelectionItem::Node(other)]);
        assert_eq!(selected_faces_or_part(&selection, Some(part)), (None, 0));

        let selection = select(&[SelectionItem::Node(other), SelectionItem::Node(part)]);
        assert_eq!(selected_faces_or_part(&selection, Some(part)), (Some((part, vec![])), 1));
    }
}
