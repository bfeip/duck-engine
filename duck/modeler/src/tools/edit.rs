//! Settings a tool edits live, by its panel fields and its grips in the
//! viewport, until it applies them; and the panel parts every tool shares.
//!
//! The primitive tools additionally drive only their preview node's transform —
//! the unit reference shape is never re-tessellated — and build the
//! world-space shape once, on apply.

use std::ops::RangeInclusive;

use anyhow::Context;
use duck_engine_viewer::common::{Real, Transform};
use duck_engine_viewer::operator::{Handle, HandleDrag, HandleEvent};
use opencascade::primitives::Shape;

use crate::preview::PreviewSession;
use crate::tools::Workspace;

/// Smallest value a dimension field accepts. Anything at or below it is
/// degenerate and can't be built.
pub(super) const MIN_DIMENSION: Real = 1e-6;

/// Settings a tool holds live, as preview geometry, until they are applied.
///
/// Two editors drive the same values: the panel's fields via
/// [`ui`](Params::ui), and the grips via [`handles`](Params::handles) and
/// [`apply_handle`](Params::apply_handle).
///
/// `Copy` because a grip drag edits from a snapshot taken when it was grabbed.
pub trait Params: Copy {
    /// The fields, one row each of a two-column grid. Returns true when a
    /// value changed.
    fn ui(&mut self, ui: &mut egui::Ui) -> bool;

    /// The grips for these settings, or empty for none.
    fn handles(&self) -> Vec<Handle> {
        Vec::new()
    }

    /// Applies a grip drag, editing from `grabbed` — the settings as they were
    /// when the grip was taken.
    ///
    /// A [`HandleDrag`] carries its total offset from the grab rather than an
    /// increment, so editing from the live value would compound it.
    fn apply_handle(&mut self, _drag: &HandleDrag, _grabbed: &Self) {}

    /// Whether these settings leave nothing to build.
    fn is_degenerate(&self) -> bool {
        false
    }
}

/// Settings under edit: the live values, and the snapshot a held grip drags
/// from.
#[derive(Clone, Copy)]
pub(super) struct Edit<P> {
    params: P,
    grabbed: Option<P>,
}

impl<P: Params> Edit<P> {
    pub(super) fn new(params: P) -> Self {
        Self { params, grabbed: None }
    }

    pub(super) fn params(&self) -> &P {
        &self.params
    }

    /// Replaces the settings, as a key that edits them does.
    pub(super) fn set(&mut self, params: P) {
        self.params = params;
    }

    pub(super) fn handles(&self) -> Vec<Handle> {
        self.params.handles()
    }

    /// Runs one grip event against the settings. Returns whether they changed.
    pub(super) fn on_handle(&mut self, event: &HandleEvent) -> bool {
        match event {
            HandleEvent::Begin(_) => {
                self.grabbed = Some(self.params);
                false
            }
            HandleEvent::Drag(drag) => {
                let Some(grabbed) = self.grabbed else { return false };
                self.params.apply_handle(drag, &grabbed);
                true
            }
            HandleEvent::End(_) => {
                self.grabbed = None;
                false
            }
            // Put the settings back as they were when the grip was taken.
            HandleEvent::Cancel(_) => match self.grabbed.take() {
                Some(grabbed) => {
                    self.params = grabbed;
                    true
                }
                None => false,
            },
        }
    }

    /// The panel's body: the fields, then Cancel / Apply.
    pub(super) fn panel(&mut self, ui: &mut egui::Ui) -> PanelAction {
        let changed = egui::Grid::new("tool_params")
            .num_columns(2)
            .show(ui, |ui| self.params.ui(ui))
            .inner;
        ui.separator();
        match apply_row(ui) {
            PanelAction::None if changed => PanelAction::Changed,
            action => action,
        }
    }
}

/// What the user asked of a tool's panel this frame.
pub(super) enum PanelAction {
    None,
    /// A value changed; the preview needs refreshing.
    Changed,
    Apply,
    Cancel,
}

/// The Cancel / Apply row that closes a tool's panel.
pub(super) fn apply_row(ui: &mut egui::Ui) -> PanelAction {
    let mut action = PanelAction::None;
    ui.horizontal(|ui| {
        if ui.button("Cancel").clicked() {
            action = PanelAction::Cancel;
        }
        if ui.button("Apply  ⏎").clicked() {
            action = PanelAction::Apply;
        }
    });
    action
}

/// `error`, in the error color.
pub(super) fn error_line(ui: &mut egui::Ui, error: &str) {
    ui.colored_label(ui.visuals().error_fg_color, error);
}

/// One labelled dimension row of a panel's grid: a length that must stay
/// above [`MIN_DIMENSION`]. Returns true when the value changed.
pub(super) fn dimension_field(ui: &mut egui::Ui, label: &str, value: &mut Real) -> bool {
    length_field(ui, label, value, MIN_DIMENSION..=Real::MAX)
}

/// One labelled length row of a panel's grid, held within `range`. Returns
/// true when the value changed.
pub(super) fn length_field(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut Real,
    range: RangeInclusive<Real>,
) -> bool {
    ui.label(label);
    let changed = ui.add(egui::DragValue::new(value).speed(0.5).range(range)).changed();
    ui.end_row();
    changed
}

/// One labelled angle row of a panel's grid, shown in degrees and held within
/// `±limit` radians. Returns true when the value changed.
pub(super) fn angle_field(ui: &mut egui::Ui, label: &str, radians: &mut Real, limit: Real) -> bool {
    ui.label(label);
    let limit = limit.to_degrees();
    let mut degrees = radians.to_degrees();
    let changed = ui
        .add(egui::DragValue::new(&mut degrees).speed(0.5).range(-limit..=limit).suffix("°"))
        .changed();
    // Written back only on an edit, so an untouched value never drifts through
    // the degree round trip.
    if changed {
        *radians = degrees.to_radians();
    }
    ui.end_row();
    changed
}

/// The settings of a placed primitive.
///
/// Both editors end in the same place — a new
/// [`preview_transform`](PrimitiveParams::preview_transform) on the preview
/// node holding the tool's unit reference shape.
pub(super) trait PrimitiveParams: Params {
    /// Panel title and committed part name.
    const NAME: &'static str;

    /// Places the tool's unit reference shape for these settings.
    fn preview_transform(&self) -> Transform;

    /// The world-space shape to commit.
    fn build(&self) -> anyhow::Result<Shape>;
}

/// A dimension moved by `delta`, held at or above [`MIN_DIMENSION`] so a grip
/// dragged past the opposite face flattens the shape rather than inverting it.
pub(super) fn grip_dimension(from: Real, delta: Real) -> Real {
    (from + delta).max(MIN_DIMENSION)
}

/// Build the world-space shape, drop the preview, and register it as a part.
/// A failed build leaves the preview session untouched so the settings can be
/// corrected and applied again.
pub(super) fn commit_primitive<P: PrimitiveParams>(
    params: &P,
    preview: &mut PreviewSession,
    workspace: &Workspace,
) -> anyhow::Result<()> {
    let shape = params.build().with_context(|| format!("Failed to build {}", P::NAME))?;

    let _ = preview.commit();
    workspace.add_numbered_part(P::NAME, shape)?;
    Ok(())
}
