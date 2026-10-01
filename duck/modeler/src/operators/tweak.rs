//! Live parameter tweaking for tools that hold their result as preview
//! geometry until it is applied.
//!
//! A tool's panel fields and its 3D grips edit the same parameters. The
//! primitive tools additionally drive only their preview node's transform — the
//! unit reference shape is never re-tessellated — and build the world-space
//! shape once, on apply.

use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use anyhow::bail;
use duck_engine_viewer::common::{Real, Transform};
use duck_engine_viewer::operator::{Handle, HandleDrag, HandleEvent};
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::preview::PreviewSession;

/// Smallest value a dimension field accepts. Anything at or below it is
/// degenerate and can't be built.
pub(super) const MIN_DIMENSION: Real = 1e-6;

/// Parameters a tool holds live, as preview geometry, until they are applied.
///
/// Two editors drive the same values: the panel's numeric fields via
/// [`ui`](TweakParams::ui), and the 3D grips via
/// [`handles`](TweakParams::handles) / [`apply_handle`](TweakParams::apply_handle).
///
/// `Copy` because a grip drag edits from a snapshot taken when it was grabbed.
pub(super) trait TweakParams: Copy {
    /// The fields, one row each of a two-column grid. Returns true when a
    /// value changed.
    fn ui(&mut self, ui: &mut egui::Ui) -> bool;

    /// The grips for these parameters, or empty for none.
    fn handles(&self) -> Vec<Handle> {
        Vec::new()
    }

    /// Applies a grip drag, editing from `grabbed` — the parameters as they
    /// were when the grip was taken.
    ///
    /// A [`HandleDrag`] carries its total offset from the grab rather than an
    /// increment, so editing from the live value would compound it.
    fn apply_handle(&mut self, _drag: &HandleDrag, _grabbed: &Self) {}
}

/// The parameters of a placed primitive.
///
/// Both editors end in the same place — a new
/// [`preview_transform`](PrimitiveParams::preview_transform) on the preview
/// node holding the tool's unit reference shape.
pub(super) trait PrimitiveParams: TweakParams {
    /// Panel title and committed part name.
    const NAME: &'static str;

    /// Places the tool's unit reference shape for these parameters.
    fn preview_transform(&self) -> Transform;

    /// The world-space shape to commit.
    fn build(&self) -> Option<Shape>;
}

/// A dimension moved by `delta`, held at or above [`MIN_DIMENSION`] so a grip
/// dragged past the opposite face flattens the shape rather than inverting it.
pub(super) fn grip_dimension(from: Real, delta: Real) -> Real {
    (from + delta).max(MIN_DIMENSION)
}

/// Runs one grip event against `params`, keeping the grab snapshot in
/// `grabbed`. Returns the parameters to show, or `None` when they are unchanged.
pub(super) fn handle_tweak<P: TweakParams>(
    params: P,
    grabbed: &mut Option<P>,
    event: &HandleEvent,
) -> Option<P> {
    match event {
        HandleEvent::Begin(_) => {
            *grabbed = Some(params);
            None
        }
        HandleEvent::Drag(drag) => {
            let grabbed = (*grabbed)?;
            let mut edited = params;
            edited.apply_handle(drag, &grabbed);
            Some(edited)
        }
        HandleEvent::End(_) => {
            *grabbed = None;
            None
        }
        // Put the parameters back as they were when the grip was taken.
        HandleEvent::Cancel(_) => grabbed.take(),
    }
}

/// What the user asked of the panel this frame.
pub(super) enum TweakAction {
    None,
    /// A value changed; the preview needs refreshing.
    Changed,
    Apply,
    Cancel,
}

/// One labelled dimension row of the tweak panel's grid: a length that must
/// stay above [`MIN_DIMENSION`]. Returns true when the value changed.
pub(super) fn dimension_field(ui: &mut egui::Ui, label: &str, value: &mut Real) -> bool {
    length_field(ui, label, value, MIN_DIMENSION..=Real::MAX)
}

/// One labelled length row of the tweak panel's grid, held within `range`.
/// Returns true when the value changed.
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

/// One labelled angle row of the tweak panel's grid, shown in degrees and held
/// within `±limit` radians. Returns true when the value changed.
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

/// Body of a tool's options window: its fields, then Cancel / Apply.
pub(super) fn tweak_panel<P: TweakParams>(ui: &mut egui::Ui, params: &mut P) -> TweakAction {
    let changed = egui::Grid::new("tweak_params")
        .num_columns(2)
        .show(ui, |ui| params.ui(ui))
        .inner;

    ui.separator();

    let mut apply_clicked = false;
    let mut cancel_clicked = false;
    ui.horizontal(|ui| {
        if ui.button("Cancel").clicked() {
            cancel_clicked = true;
        }
        if ui.button("Apply  ⏎").clicked() {
            apply_clicked = true;
        }
    });

    if apply_clicked {
        TweakAction::Apply
    } else if cancel_clicked {
        TweakAction::Cancel
    } else if changed {
        TweakAction::Changed
    } else {
        TweakAction::None
    }
}

/// Build the world-space shape, drop the preview, and register it as a part.
/// A failed build leaves the preview session untouched so the parameters can be
/// corrected and applied again.
pub(super) fn commit_tweak<P: PrimitiveParams>(
    params: &P,
    preview: &mut PreviewSession,
    document: &Arc<Mutex<Document>>,
    options: &CadTessellationOptions,
) -> anyhow::Result<()> {
    let Some(shape) = params.build() else {
        bail!("Failed to build {}", P::NAME);
    };

    let _ = preview.commit();

    let mut doc = document.lock().unwrap();
    doc.add_numbered_part(P::NAME, shape, options)?;
    Ok(())
}
