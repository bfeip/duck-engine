use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{InnerSpace, MetricSpace, Point3, Quaternion, Vector3};
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use glam::dvec3;
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
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
}

fn vec_to_dvec3(v: Vector3) -> glam::DVec3 {
    dvec3(v.x as f64, v.y as f64, v.z as f64)
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
        }
    }

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

        // The polar axis comes from the placement snap (chosen in `on_place_center`):
        // the snapped direction when there was one, else a skewed fallback that keeps
        // the seam/poles off the world axes for OCCT boolean near-coincidence robustness.
        let world_shape = Shape::sphere(radius as f64)
            .at(dvec3(center.x as f64, center.y as f64, center.z as f64))
            .axis(vec_to_dvec3(axis))
            .build();

        // Discard the preview node, then commit the world-space shape as a registered part.
        let _ = self.preview.commit();

        let committed = {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            doc.add_part(
                "Sphere".to_owned(),
                world_shape,
                &coptions.geometry_options,
            )
            .is_ok()
        };

        self.phase = Phase::Idle;
        committed
    }

    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
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
        // The modeler hides the cursor for the (now inactive) tool, but clear our
        // target so a stale point can't flash if we're reactivated before a move.
        self.cursor_target = None;
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor_target
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
                        SphereAction::Place => {
                            if let Phase::Defining { center, axis } = self.phase {
                                self.on_place_outer(center, axis, *position, ctx)
                            } else {
                                self.on_place_center(*position, ctx)
                            }
                        }
                        SphereAction::Cancel => {
                            let was_defining = matches!(self.phase, Phase::Defining { .. });
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
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Sphere"
    }
}
