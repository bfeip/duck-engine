//! The selection-kind buttons over the viewport's top-left corner.

use duck_engine_viewer::operator::SelectionKinds;

use crate::shortcuts::SELECTION_KIND_KEYS;
use crate::tools::SelectionFilter;
use crate::ui::icons::{self, Icon};
use crate::AppAction;

/// One toggle per kind a click may select. A click makes its kind the only
/// one; a shift-click adds or removes it.
#[derive(Default)]
pub struct SelectionFilterBar;

impl SelectionFilterBar {
    /// Render the buttons for `filter`, appending the kind picked to `actions`.
    pub fn show(&mut self, ctx: &egui::Context, filter: SelectionFilter, actions: &mut Vec<AppAction>) {
        // The panels are already drawn, so this is the 3D viewport's rect.
        let viewport = ctx.available_rect();
        egui::Area::new(egui::Id::new("selection_filter"))
            .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
            .constrain_to(viewport)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for (kind, key) in SELECTION_KIND_KEYS {
                        let ((uri, bytes), name) = look(kind);
                        let image = egui::Image::from_bytes(uri, bytes)
                            .fit_to_exact_size(egui::vec2(16.0, 16.0));
                        let button = egui::Button::image(image).selected(filter.kinds.contains(kind));
                        let response = ui
                            .add_enabled(filter.allowed.contains(kind), button)
                            .on_hover_text(format!("{name} ({key}) · Shift-click adds or removes"))
                            .on_disabled_hover_text(format!("{name}: not used by this tool"));
                        if response.clicked() {
                            let extend = ui.input(|input| input.modifiers.shift);
                            actions.push(AppAction::SelectableKind { kind, extend });
                        }
                    }
                });
            });
    }
}

/// The icon and plural name of `kind`.
fn look(kind: SelectionKinds) -> (Icon, &'static str) {
    match kind {
        SelectionKinds::POINT => (icons::SELECT_POINT, "Points"),
        SelectionKinds::EDGE => (icons::SELECT_EDGE, "Edges"),
        SelectionKinds::FACE => (icons::SELECT_FACE, "Faces"),
        _ => (icons::SELECT_PART, "Parts"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVERY_KIND_BUT_POINTS: SelectionKinds =
        SelectionKinds::NODE.union(SelectionKinds::FACE).union(SelectionKinds::EDGE);

    /// Runs a frame of the bar over `filter` with `events` and `modifiers`,
    /// returning the actions asked for and where its icons were drawn, left to
    /// right.
    fn frame(
        ctx: &egui::Context,
        filter: SelectionFilter,
        events: Vec<egui::Event>,
        modifiers: egui::Modifiers,
    ) -> (Vec<AppAction>, Vec<egui::Rect>) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))),
            events,
            modifiers,
            ..Default::default()
        };
        let mut actions = Vec::new();
        let output = ctx.run(input, |ctx| SelectionFilterBar.show(ctx, filter, &mut actions));
        let icons = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) if rect.fill_texture_id() != egui::TextureId::default() => {
                    Some(rect.rect)
                }
                _ => None,
            })
            .collect();
        (actions, icons)
    }

    /// The icons of the bar over `filter`, once it has been laid out.
    fn icons(ctx: &egui::Context, filter: SelectionFilter) -> Vec<egui::Rect> {
        egui_extras::install_image_loaders(ctx);
        frame(ctx, filter, Vec::new(), egui::Modifiers::NONE);
        frame(ctx, filter, Vec::new(), egui::Modifiers::NONE).1
    }

    /// Clicks at `at` with `modifiers` held, returning the actions asked for.
    fn click(
        ctx: &egui::Context,
        filter: SelectionFilter,
        at: egui::Pos2,
        modifiers: egui::Modifiers,
    ) -> Vec<AppAction> {
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers,
        };
        [egui::Event::PointerMoved(at), button(true), button(false)]
            .into_iter()
            .flat_map(|event| frame(ctx, filter, vec![event], modifiers).0)
            .collect()
    }

    #[test]
    fn a_click_takes_one_kind_and_a_shift_click_adds_or_removes() {
        let ctx = egui::Context::default();
        let filter = SelectionFilter { kinds: EVERY_KIND_BUT_POINTS, allowed: SelectionKinds::all() };
        let icons = icons(&ctx, filter);
        assert_eq!(icons.len(), 4);

        let face = click(&ctx, filter, icons[2].center(), egui::Modifiers::NONE);
        assert_eq!(face, [AppAction::SelectableKind { kind: SelectionKinds::FACE, extend: false }]);
        let point = click(&ctx, filter, icons[0].center(), egui::Modifiers::SHIFT);
        assert_eq!(point, [AppAction::SelectableKind { kind: SelectionKinds::POINT, extend: true }]);
    }

    #[test]
    fn kinds_the_tool_does_not_act_on_cannot_be_picked() {
        let ctx = egui::Context::default();
        let filter = SelectionFilter { kinds: SelectionKinds::EDGE, allowed: SelectionKinds::EDGE };
        let icons = icons(&ctx, filter);
        assert!(click(&ctx, filter, icons[3].center(), egui::Modifiers::NONE).is_empty());
        assert_eq!(
            click(&ctx, filter, icons[1].center(), egui::Modifiers::NONE),
            [AppAction::SelectableKind { kind: SelectionKinds::EDGE, extend: false }],
        );
    }
}
