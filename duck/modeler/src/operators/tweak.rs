//! Post-placement parameter tweaking for the primitive tools.
//!
//! Once its last point is picked, a primitive tool holds the shape as live
//! preview geometry and opens an options panel instead of committing. The
//! numeric fields drive only the preview node's transform — the unit reference
//! shape is never re-tessellated — and the world-space shape is built once, on
//! apply.

use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_viewer::common::Transform;
use log::{error, warn};
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::preview::PreviewSession;

/// Smallest value a dimension field accepts. Anything at or below it is
/// degenerate and can't be built.
pub(super) const MIN_DIMENSION: f32 = 1e-6;

/// The parameters of a placed primitive, still adjustable before commit.
pub(super) trait TweakParams {
    /// Panel title and committed part name.
    const NAME: &'static str;

    /// Places the tool's unit reference shape for these parameters.
    fn preview_transform(&self) -> Transform;

    /// The world-space shape to commit.
    fn build(&self) -> Option<Shape>;

    /// The dimension fields, one [`dimension_field`] per row of a two-column
    /// grid. Returns true when a value changed.
    fn ui(&mut self, ui: &mut egui::Ui) -> bool;
}

/// What the user asked of the panel this frame.
pub(super) enum TweakAction {
    None,
    /// A dimension changed; the preview transform needs refreshing.
    Changed,
    Apply,
    Cancel,
}

/// One labelled dimension row of the tweak panel's grid. Returns true when the
/// value changed.
pub(super) fn dimension_field(ui: &mut egui::Ui, label: &str, value: &mut f32) -> bool {
    ui.label(label);
    let changed = ui
        .add(egui::DragValue::new(value).speed(0.5).range(MIN_DIMENSION..=f32::MAX))
        .changed();
    ui.end_row();
    changed
}

/// Body of a primitive's options window: its dimension fields, then Cancel / Apply.
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
pub(super) fn commit_tweak<P: TweakParams>(
    params: &P,
    preview: &mut PreviewSession,
    document: &Arc<Mutex<Document>>,
    options: &CadTessellationOptions,
) -> bool {
    let Some(shape) = params.build() else {
        warn!("Failed to build {}", P::NAME);
        return false;
    };

    let _ = preview.commit();

    let mut doc = document.lock().unwrap();
    match doc.add_part(P::NAME.to_owned(), shape, options) {
        Ok(_) => true,
        Err(e) => {
            error!("Failed to add {}: {e}", P::NAME);
            false
        }
    }
}
