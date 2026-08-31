use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{InnerSpace, MetricSpace, Point3, Quaternion, Vector3};
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, Modifiers, MouseButton, NamedKey},
    operator::Operator,
};
use glam::dvec3;
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
use crate::ui::icons;
use super::tweak::{commit_tweak, dimension_field, tweak_panel, TweakAction, TweakParams};
use super::ConstructionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SphereAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the radius. `axis` is the polar axis.
    Defining { center: Point3, axis: Vector3 },
    /// Radius picked; the options panel drives it until the sphere is applied
    /// or cancelled.
    Tweak(SphereParams),
}

fn vec_to_dvec3(v: Vector3) -> glam::DVec3 {
    dvec3(v.x as f64, v.y as f64, v.z as f64)
}

/// The dimensions of a placed sphere, adjustable before it is committed.
/// `center` is the first point picked and never moves: the radius grows about it.
#[derive(Clone, Copy)]
pub(super) struct SphereParams {
    center: Point3,
    /// Polar axis, chosen at placement (see [`SphereOperator::on_place_center`]).
    axis: Vector3,
    radius: f32,
}

impl TweakParams for SphereParams {
    const NAME: &'static str = "Sphere";

    fn preview_transform(&self) -> Transform {
        SphereOperator::preview_transform(self.center, self.radius)
    }

    fn build(&self) -> Option<Shape> {
        Some(
            Shape::sphere(self.radius as f64)
                .at(dvec3(self.center.x as f64, self.center.y as f64, self.center.z as f64))
                .axis(vec_to_dvec3(self.axis))
                .build(),
        )
    }

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        dimension_field(ui, "Radius", &mut self.radius)
    }
}

pub struct SphereOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<SphereAction>,
    /// Where the modeler's 3D cursor should sit (the latest snap point), or
    /// `None` to hide it. Read by the modeler via [`ModelingTool::cursor_target`].
    cursor_target: Option<Point3>,
    /// Set once the sphere is applied, so the
    /// tool cedes back to selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl SphereOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                SphereAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                SphereAction::Cancel,
            );
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            preview,
            bindings,
            cursor_target: None,
            finished: false,
        }
    }

    /// Scales the unit reference sphere to `radius` about `center`.
    fn preview_transform(center: Point3, radius: f32) -> Transform {
        Transform {
            position: center,
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale: Vector3::new(radius, radius, radius),
        }
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let camera = ctx.camera.clone();
        let Some(snap) = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[], &camera, ctx, &[])
        else {
            return false;
        };
        let center = snap.position;
        // Polar axis: the snapped direction (e.g. a face normal) when present, else
        // a skewed fallback that keeps the seam/poles off every world axis so a
        // later boolean's cutting plane isn't near-coincident with them (OCCT
        // boolean near-coincidence robustness).
        let axis = snap
            .direction
            .unwrap_or_else(|| Vector3::new(1.0, 2.0, 3.0).normalize());
        // Does not need preview tessellation detail because we only make the
        // sphere once, and then scale it.
        let preview_shape = Shape::sphere(1.0).build();
        let options = self.construction_options.borrow().geometry_options.clone();
        let Some(node) = self.preview.add_preview_from_shape(&preview_shape, &options, "sphere") else {
            return false;
        };
        ctx.scene
            
            .set_node_transform(node, Self::preview_transform(center, 0.01));
        self.phase = Phase::Defining { center, axis };
        true
    }

    fn on_place_outer(
        &mut self,
        center: Point3,
        axis: Vector3,
        position: (f32, f32),
        ctx: &mut EventContext
    ) -> bool {
        let camera = ctx.camera.clone();
        // Exclude the preview so the radius can snap through a corner, not to the
        // preview's own geometry.
        let radius = self
            .construction_options
            .borrow()
            .resolve_snap(position, self.preview.preview_nodes(), &camera, ctx, &[])
            .map(|s| center.distance(s.position).max(0.01))
            .unwrap_or(0.01);

        // Hand the sphere to the options panel rather than committing it: the
        // preview stays live and the radius stays editable until Apply. The polar
        // axis comes from the placement snap (chosen in `on_place_center`).
        let params = SphereParams { center, axis, radius };
        self.preview.set_preview_transform(params.preview_transform());
        self.phase = Phase::Tweak(params);
        true
    }

    /// Commit the sphere and finish the tool. A failed build keeps the
    /// panel open so the radius can be corrected.
    fn apply(&mut self) {
        let Phase::Tweak(params) = self.phase else { return };
        let options = self.construction_options.borrow().geometry_options.clone();
        if commit_tweak(&params, &mut self.preview, &self.document, &options) {
            self.phase = Phase::Idle;
            self.finished = true;
        }
    }

    /// Drop the in-progress sphere.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        // The radius is picked; the panel drives the preview from here.
        if matches!(self.phase, Phase::Tweak(_)) {
            return;
        }
        let cursor = (position.0 as f32, position.1 as f32);

        let camera = ctx.camera.clone();
        // While defining, exclude our own preview so the radius doesn't snap to it.
        let snap = self.construction_options.borrow().resolve_snap(
            cursor,
            self.preview.preview_nodes(),
            &camera,
            ctx,
            &[],
        );

        // Record where the modeler should draw the 3D cursor
        self.cursor_target = snap.map(|s| s.position);

        // Drive the preview radius from the snapped point while defining.
        if let Phase::Defining { center, .. } = self.phase {
            if let (Some(snap), Some(preview_node)) = (snap, self.preview.preview_node()) {
                let radius = center.distance(snap.position).max(0.01);
                ctx.scene
                    
                    .set_node_transform(preview_node, Self::preview_transform(center, radius));
            }
        }
    }
}

impl ModelingTool for SphereOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "sphere", icon: icons::SPHERE, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.finished = false;
        // The modeler hides the cursor for the (now inactive) tool, but clear our
        // target so a stale point can't flash if we're reactivated before a move.
        self.cursor_target = None;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn cursor_target(&self) -> Option<Point3> {
        // Nothing left to pick while the panel is open.
        match self.phase {
            Phase::Tweak(_) => None,
            _ => self.cursor_target,
        }
    }

    fn panel_title(&self) -> Option<&str> {
        matches!(self.phase, Phase::Tweak(_)).then_some(SphereParams::NAME)
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, _panel: &mut PanelContext) {
        let Phase::Tweak(params) = &mut self.phase else { return };
        let action = tweak_panel(ui, params);
        let transform = params.preview_transform();
        match action {
            TweakAction::Changed => self.preview.set_preview_transform(transform),
            TweakAction::Apply => self.apply(),
            TweakAction::Cancel => self.cancel(),
            TweakAction::None => {}
        }
    }
}

impl Operator for SphereOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        SphereAction::Place => match self.phase {
                            Phase::Defining { center, axis } => {
                                self.on_place_outer(center, axis, *position, ctx)
                            }
                            // Swallow the click: the panel owns the sphere now, so a
                            // stray pick must not select or place anything.
                            Phase::Tweak(_) => true,
                            Phase::Idle => self.on_place_center(*position, ctx),
                        },
                        SphereAction::Cancel => {
                            let was_defining = !matches!(self.phase, Phase::Idle);
                            if was_defining {
                                self.cancel();
                            }
                            was_defining
                        }
                    };
                }
                handled
            }
            DeviceEvent::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            // Keyboard equivalents of the tweak panel's Apply and Cancel buttons.
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if !matches!(self.phase, Phase::Tweak(_))
                    || key_event.state != ElementState::Pressed
                    || key_event.repeat
                {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => {
                        self.apply();
                        true
                    }
                    Key::Named(NamedKey::Escape) => {
                        self.cancel();
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Sphere"
    }
}
