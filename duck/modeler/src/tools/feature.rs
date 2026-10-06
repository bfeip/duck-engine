//! The driver for tools that apply a feature to part geometry the selection
//! picks: fillet, draft, extrude, thicken and hollow.
//!
//! A [`Feature`] says what the selection designates, the settings that start
//! from it, the result they build and what that result does to its part.
//! [`FeatureTool`] runs the rest the same way for every feature:
//!
//! - It follows the selection, resolving fresh settings whenever what the
//!   selection designates changes.
//! - The first edit — a grip grabbed, a field changed, a feature key — locks
//!   the target as far as [`Feature::LOCK`] says. From then a plain click stops
//!   at the tool; under [`EditLock::Part`] a shift-click still refines the
//!   target, which keeps the edited settings.
//! - While editing, the result is rebuilt whenever its target or settings
//!   change, and previewed as its [`SourceFate`] says. A failed build keeps the
//!   last good preview and says why in the panel.
//! - Enter, right-click, Apply, or switching tools commits the result last
//!   previewed as one undo step; Escape or Cancel discards it. Either leaves
//!   the tool.

use anyhow::{Context, Result};
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::{NodeFlags, NodeId};
use duck_engine_viewer::{
    event::{Event, EventContext},
    input::Modifiers,
    operator::{Handle, HandleEvent, SelectionMode},
    selection::SelectionManager,
};
use opencascade::primitives::Shape;

use crate::construction::ConstructionOptions;
use crate::document::{Document, SourceFate};
use crate::preview::PreviewSession;
use super::edit::{error_line, Edit, PanelAction, Params};
use super::targets::Selected;
use super::{Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};

/// How much of its target a feature holds once editing has begun.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditLock {
    /// All of it: the selection is ignored and every click stops at the tool.
    Target,
    /// Its part: shift-clicks still add and drop sub-shapes on the part.
    Part,
}

/// An operation on part geometry the selection picks, run by [`FeatureTool`].
pub trait Feature: 'static {
    /// What the selection designates.
    type Target: Clone + PartialEq;
    /// The settings the panel and grips edit.
    type Params: Params + PartialEq;

    /// Palette identity.
    const TOOL: ToolInfo;
    /// What a click selects while the tool is active.
    const SELECTS: SelectionMode;
    /// How much of the target an edit locks.
    const LOCK: EditLock;

    /// What the selection designates, held to the `locked` part once editing.
    fn select(&self, selection: &SelectionManager, locked: Option<NodeId>) -> Option<Selected<Self::Target>>;

    /// Settings for `target`: fresh ones, or `edited`'s carried onto it.
    fn resolve(
        &self,
        doc: &Document,
        target: &Self::Target,
        construction: &ConstructionOptions,
        edited: Option<&Self::Params>,
    ) -> Result<Self::Params>;

    /// The result, to preview and to commit.
    fn build(&self, doc: &Document, target: &Self::Target, params: &Self::Params) -> Result<Shape>;

    /// What the result does to its part.
    fn fate(&self, params: &Self::Params) -> SourceFate;

    /// Name of the operation, for the panel title and the undo step.
    fn title(&self, params: Option<&Self::Params>) -> &'static str;

    /// The name series a result kept beside its part is numbered in.
    fn new_part_name(&self) -> &'static str {
        self.title(None)
    }

    /// What to select, shown while nothing is targeted.
    fn prompt(&self, selection: &SelectionManager) -> &'static str;

    /// A line describing the target, shown above the fields.
    fn summary(&self, _target: &Self::Target, _ignored: usize) -> Option<String> {
        None
    }

    /// `params` as edited by the unmodified, lowercased key `c`, or `None` if
    /// `c` means nothing here.
    fn on_char(&self, _c: char, _params: &Self::Params) -> Option<Self::Params> {
        None
    }
}

/// A tool that applies `F` to the geometry the selection picks, previewing it
/// live until applied.
pub struct FeatureTool<F: Feature> {
    feature: F,
    workspace: Workspace,
    preview: PreviewSession,
    /// The feature in progress, once the selection designates a target.
    session: Option<Session<F>>,
    /// What the selection last designated, so that settings are resolved again
    /// only when it changes.
    followed: Option<F::Target>,
    /// Why the target or its result could not be resolved or built, shown in
    /// the panel.
    error: Option<String>,
    /// Set once the result is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

/// A feature's target, and the settings being edited for it.
struct Session<F: Feature> {
    node: NodeId,
    target: F::Target,
    /// Selected items the target leaves out.
    ignored: usize,
    edit: Edit<F::Params>,
    /// Set by the first edit: the target is held, as far as
    /// [`Feature::LOCK`] says, until the result is applied or discarded.
    locked: bool,
    /// The result last built, and what it was built for.
    built: Option<Built<F>>,
}

/// A result as built for a target and settings; `shape` is `None` when there
/// was nothing to build or the build failed.
struct Built<F: Feature> {
    target: F::Target,
    params: F::Params,
    shape: Option<Shape>,
}

impl<F: Feature + Default> FeatureTool<F> {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            feature: F::default(),
            workspace: workspace.clone(),
            preview: workspace.preview_session(),
            session: None,
            followed: None,
            error: None,
            finished: false,
        }
    }
}

impl<F: Feature> FeatureTool<F> {
    /// The settings being edited, once there is a target.
    pub(super) fn params(&self) -> Option<&F::Params> {
        self.session.as_ref().map(|session| session.edit.params())
    }

    /// Whether editing has begun, locking the target.
    pub(super) fn is_editing(&self) -> bool {
        self.session.as_ref().is_some_and(|session| session.locked)
    }

    /// Acts on `gesture`, returning whether it was consumed.
    pub(super) fn on_gesture(&mut self, gesture: Gesture, selection: &mut SelectionManager) -> bool {
        match gesture {
            Gesture::Frame => {
                self.follow_selection(selection);
                self.refresh_preview();
                false
            }
            Gesture::Click { modifiers, .. } => self.swallows_click(modifiers),
            Gesture::Finish if self.is_editing() => {
                self.apply_and_report(selection);
                true
            }
            Gesture::Cancel => {
                self.cancel();
                true
            }
            Gesture::Key(c) => self.edit_with_key(c),
            Gesture::Finish | Gesture::Hover(_) => false,
        }
    }

    /// Points the feature at what the selection designates, once that changes.
    /// While editing, the target is held as far as [`Feature::LOCK`] says, and
    /// the edited settings carry over.
    pub(super) fn follow_selection(&mut self, selection: &SelectionManager) {
        let editing = self
            .session
            .as_ref()
            .filter(|session| session.locked)
            .map(|session| (session.node, *session.edit.params()));
        if editing.is_some() && F::LOCK == EditLock::Target {
            return;
        }
        let selected = self.feature.select(selection, editing.map(|(node, _)| node));
        let designated = selected.as_ref().map(|selected| selected.target.clone());
        if designated == self.followed {
            if let (Some(session), Some(selected)) = (&mut self.session, selected) {
                session.ignored = selected.ignored;
            }
            return;
        }
        self.followed = designated;

        let Some(selected) = selected else {
            // The selection emptied: there is nothing left to act on.
            self.abandon();
            return;
        };
        let edited = editing.map(|(_, params)| params);
        let resolved = self.feature.resolve(
            &self.workspace.document.lock().unwrap(),
            &selected.target,
            &self.workspace.construction.borrow(),
            edited.as_ref(),
        );
        match resolved {
            Ok(params) => {
                self.error = None;
                self.session = Some(Session {
                    node: selected.node,
                    target: selected.target,
                    ignored: selected.ignored,
                    edit: Edit::new(params),
                    locked: edited.is_some(),
                    built: None,
                });
            }
            Err(e) => {
                self.abandon();
                self.error = Some(format!("{e:#}"));
            }
        }
    }

    /// Rebuilds the preview when an edit's target or settings have moved on
    /// since it was last built.
    pub(super) fn refresh_preview(&mut self) {
        let Some(session) = self.session.as_ref().filter(|session| session.locked) else { return };
        let (node, target, params) = (session.node, session.target.clone(), *session.edit.params());
        if session.built.as_ref().is_some_and(|built| built.target == target && built.params == params) {
            return;
        }
        let shape = self.show_result(node, &target, &params);
        if let Some(session) = &mut self.session {
            session.built = Some(Built { target, params, shape });
        }
    }

    /// Previews the result of `params` on `target` as its fate says, and
    /// returns it; or says why it can't be built, keeping the last good preview.
    fn show_result(&mut self, node: NodeId, target: &F::Target, params: &F::Params) -> Option<Shape> {
        if params.is_degenerate() {
            // Nothing to show, and anything hidden comes back.
            self.preview.clear_previews();
            self.error = None;
            return None;
        }
        let fate = self.feature.fate(params);
        let (built, options) = {
            let doc = self.workspace.document.lock().unwrap();
            (self.feature.build(&doc, target, params), self.preview_options(&doc, node, fate))
        };
        let shape = match built {
            Ok(shape) => shape,
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                return None;
            }
        };
        let title = self.feature.title(Some(params));
        if self.preview.try_replace_preview(&shape, &options, &format!("{title} preview")).is_none() {
            self.error = Some(format!("The {} could not be tessellated", title.to_lowercase()));
            return None;
        }
        self.error = None;
        // Clicks reach the part the preview hides or covers, whose sub-shapes
        // shift-clicks toggle.
        self.preview.set_preview_flags(NodeFlags::DO_NOT_SELECT);
        match fate {
            SourceFate::Reshape | SourceFate::Replace => self.preview.hide_source_node(node),
            // A build with an earlier fate may have hidden it.
            SourceFate::Fuse | SourceFate::Keep => self.preview.show_sources(),
        }
        Some(shape)
    }

    /// Tessellation options for the preview: the part's own when the result
    /// reshapes it, the construction ones otherwise, both at the coarser
    /// preview tolerance.
    fn preview_options(&self, doc: &Document, node: NodeId, fate: SourceFate) -> CadTessellationOptions {
        let construction = self.workspace.construction.borrow();
        let mut options = match (fate, doc.part_at(node)) {
            (SourceFate::Reshape, Some(part)) => part.options().clone(),
            _ => construction.geometry_options.clone(),
        };
        options.tessellation_tolerance = construction.preview_tolerance;
        options
    }

    /// Commits the result last previewed, or builds it now if there is none,
    /// and finishes the tool. A failure keeps the preview and panel so the
    /// settings can be corrected.
    pub(super) fn apply(&mut self, selection: &mut SelectionManager) -> Result<()> {
        let Some(session) = &self.session else { return Ok(()) };
        let params = *session.edit.params();
        let fate = self.feature.fate(&params);
        let previewed = session
            .built
            .as_ref()
            .filter(|built| built.target == session.target && built.params == params)
            .and_then(|built| built.shape.clone());
        {
            let mut doc = self.workspace.document.lock().unwrap();
            let result = match previewed {
                Some(shape) => shape,
                None => self.feature.build(&doc, &session.target, &params)?,
            };
            let part = doc.part_for_node(session.node).context("The target is not a known CAD part")?;
            let (label, new_part) = (self.feature.title(Some(&params)), self.feature.new_part_name());
            doc.commit_result(part, result, fate, label, new_part, &self.workspace.geometry_options())?;
        }

        match fate {
            // The hidden part was consumed by the result.
            SourceFate::Replace => {
                let _ = self.preview.commit();
            }
            // The part stands, reshaped where it is or beside the result: it
            // comes back, even if a build with an earlier fate hid it.
            SourceFate::Reshape | SourceFate::Fuse | SourceFate::Keep => self.preview.cancel(),
        }
        // The selected sub-shapes were renumbered or removed.
        selection.clear();
        self.session = None;
        self.error = None;
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self, selection: &mut SelectionManager) {
        let title = self.feature.title(self.params());
        if let Err(e) = self.apply(selection) {
            self.workspace.notifications.failure(title, &e);
        }
    }

    /// Discards the feature and finishes the tool, restoring anything the
    /// preview hid.
    pub(super) fn cancel(&mut self) {
        self.preview.cancel();
        self.session = None;
        self.error = None;
        self.finished = true;
    }

    /// Drops the feature in progress without finishing the tool: the preview
    /// goes, and anything it hid comes back.
    fn abandon(&mut self) {
        self.preview.clear_previews();
        self.session = None;
        self.error = None;
    }

    /// Whether a click with `modifiers` stops at the tool. Once editing, the
    /// grips and panel own the feature and a stray pick must not reselect
    /// anything; shift-clicks pass only where they refine a locked part.
    pub(super) fn swallows_click(&self, modifiers: Modifiers) -> bool {
        self.is_editing() && !(F::LOCK == EditLock::Part && modifiers.shift)
    }

    /// Hands the key `c` to the feature: an edit, if it changes the settings.
    fn edit_with_key(&mut self, c: char) -> bool {
        let Some(session) = &mut self.session else { return false };
        let Some(edited) = self.feature.on_char(c, session.edit.params()) else { return false };
        session.edit.set(edited);
        session.locked = true;
        true
    }
}

impl<F: Feature> ModelingTool for FeatureTool<F> {
    fn info(&self) -> ToolInfo {
        F::TOOL
    }

    fn selection_mode(&self) -> SelectionMode {
        F::SELECTS
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        Gesture::read(event, ctx.modifiers).is_some_and(|gesture| self.on_gesture(gesture, ctx.selection))
    }

    fn handles(&self) -> Vec<Handle> {
        self.session.as_ref().map(|session| session.edit.handles()).unwrap_or_default()
    }

    /// Grabbing a grip begins the edit, which locks the target.
    fn on_handle(&mut self, event: &HandleEvent) {
        let Some(session) = &mut self.session else { return };
        if let HandleEvent::Begin(_) = event {
            session.locked = true;
        }
        session.edit.on_handle(event);
    }

    fn panel_title(&self) -> Option<&str> {
        Some(self.feature.title(self.params()))
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, panel: &mut PanelContext) {
        if let Some(error) = &self.error {
            error_line(ui, error);
        }
        let Some(session) = &mut self.session else {
            if self.error.is_none() {
                ui.label(self.feature.prompt(panel.selection));
            }
            return;
        };
        if let Some(summary) = self.feature.summary(&session.target, session.ignored) {
            ui.label(summary);
        }
        match session.edit.panel(ui) {
            // An edit begins the feature, which locks the target.
            PanelAction::Changed => session.locked = true,
            PanelAction::Apply => self.apply_and_report(panel.selection),
            PanelAction::Cancel => self.cancel(),
            PanelAction::None => {}
        }
    }

    /// An edit in progress is a finished one, so leaving the tool commits it.
    /// A degenerate one is nothing to commit.
    fn finalize(&mut self, selection: &mut SelectionManager) -> Result<()> {
        match &self.session {
            Some(session) if session.locked && !session.edit.params().is_degenerate() => self.apply(selection),
            _ => Ok(()),
        }
    }

    fn deactivate(&mut self) {
        self.preview.cancel();
        self.session = None;
        self.followed = None;
        self.error = None;
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// The driver's state, for the features' tests.
#[cfg(test)]
impl<F: Feature> FeatureTool<F> {
    pub(super) fn target(&self) -> Option<&F::Target> {
        self.session.as_ref().map(|session| &session.target)
    }

    pub(super) fn ignored(&self) -> usize {
        self.session.as_ref().map_or(0, |session| session.ignored)
    }

    pub(super) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub(super) fn preview(&self) -> &PreviewSession {
        &self.preview
    }
}
