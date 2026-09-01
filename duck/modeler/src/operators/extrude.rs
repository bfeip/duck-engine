use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::{Point3, Ray};
use duck_engine_scene::resource::SubGeometryKind;
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, MouseButton, NamedKey},
    operator::{Operator, SelectionKinds, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::document::Document;
use crate::extrude::{execute_extrude, preview_extrude, ExtrudeFrame, ExtrudeTarget};
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

/// A drag shorter than this is degenerate: there is nothing to commit.
const MIN_LENGTH: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExtrudePhase {
    /// No valid sub-geometry chosen yet — waiting for a face/edge selection.
    AwaitingSelection,
    /// A face/edge is chosen; the cursor drives the extrusion length.
    Extruding,
    Done,
    Cancelled,
}

pub struct ExtrudeOperator {
    phase: ExtrudePhase,

    /// The sub-geometry being extruded and its fixed axis, set on entering `Extruding`.
    target: Option<ExtrudeTarget>,
    frame: Option<ExtrudeFrame>,
    /// Signed length along `frame.axis`, driven by the cursor.
    length: f64,

    /// Transient preview geometry (rebuilt as the length changes) and the source
    /// node hidden while the preview stands in for it.
    preview: PreviewSession,

    cursor_target: Option<Point3>,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
}

impl ExtrudeOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: ExtrudePhase::AwaitingSelection,
            target: None,
            frame: None,
            length: 0.0,
            preview,
            cursor_target: None,
            document,
            construction_options,
        }
    }

    /// The face/edge currently chosen as the primary selection, if any.
    fn selected_target(selection: &SelectionManager) -> Option<ExtrudeTarget> {
        match selection.primary()? {
            SelectionItem::SubGeometry { node_id, element } => match element.kind {
                SubGeometryKind::Face => {
                    Some(ExtrudeTarget::Face { node: node_id, face_index: element.index })
                }
                SubGeometryKind::Edge => {
                    Some(ExtrudeTarget::Edge { node: node_id, edge_index: element.index })
                }
                SubGeometryKind::Pointset => None,
            },
            SelectionItem::Node(_) => None,
        }
    }

    /// Lock onto `target` and enter the extruding phase. Fails silently (staying in
    /// `AwaitingSelection`) if the sub-geometry can't be resolved to a CAD sub-shape.
    fn begin(&mut self, target: ExtrudeTarget) {
        // Edges extrude out of the sketch (construction) plane; faces ignore this and
        // use their own normal.
        let sketch_normal = self.construction_options.borrow().construction_plane.normal;
        let frame = {
            let doc = self.document.lock().unwrap();
            match ExtrudeFrame::new(&doc, target, sketch_normal) {
                Ok(frame) => frame,
                Err(e) => {
                    log::warn!("Extrude target could not be resolved: {e}");
                    return;
                }
            }
        };
        self.target = Some(target);
        self.frame = Some(frame);
        self.length = 0.0;
        self.phase = ExtrudePhase::Extruding;
        self.cursor_target = Some(frame.origin);
    }

    /// Project the cursor pick ray onto the fixed axis to get a new length, then
    /// rebuild the preview.
    fn update_length(&mut self, cursor: (f32, f32), ctx: &mut EventContext) {
        let Some(frame) = self.frame else { return };
        let camera = ctx.camera.clone();
        let ray: Ray = camera.ray_from_screen_point(cursor.0, cursor.1, ctx.size.0, ctx.size.1);
        if let Some(t) = ray.closest_param_on_axis(frame.origin, frame.axis) {
            self.length = t as f64;
            self.cursor_target = Some(frame.origin + frame.axis * t);
            self.refresh_preview();
        }
    }

    /// Tear down the old preview and rebuild it for the current length, hiding the
    /// source node while the preview stands in for it.
    fn refresh_preview(&mut self) {
        let (Some(target), Some(frame)) = (self.target, self.frame) else { return };
        let options = self.construction_options.borrow().preview_options();

        // Drop the old preview and re-show the source before rebuilding.
        self.preview.clear_previews();

        let result = {
            let doc = self.document.lock().unwrap();
            preview_extrude(&doc, target, &frame, self.length, &options)
        };
        match result {
            Ok(preview) => {
                self.preview.add_preview_node(preview.node);
                // Only hide the source when the result stands in for it, so a degenerate
                // preview can never make unrelated geometry vanish.
                if preview.hide_source {
                    self.preview.hide_source_node(target.node());
                }
            }
            // A zero/degenerate length simply yields no preview; not worth surfacing.
            Err(e) => log::debug!("Extrude preview unavailable: {e}"),
        }
    }

    /// Commit the extrusion. On success sets phase = Done; the caller clears selection.
    fn apply(&mut self) -> anyhow::Result<()> {
        let (Some(target), Some(frame)) = (self.target, self.frame) else { return Ok(()) };
        let options = self.construction_options.borrow().geometry_options.clone();

        // Drop the preview; the source stays hidden — execute_extrude may delete it.
        let _ = self.preview.commit();

        let mut doc = self.document.lock().unwrap();
        execute_extrude(&mut doc, target, &frame, self.length, &options)?;
        drop(doc);

        self.reset(ExtrudePhase::Done);
        Ok(())
    }

    /// Abort, restoring the source node's visibility.
    fn cancel(&mut self) {
        self.preview.cancel();
        self.reset(ExtrudePhase::Cancelled);
    }

    fn reset(&mut self, phase: ExtrudePhase) {
        self.target = None;
        self.frame = None;
        self.length = 0.0;
        self.cursor_target = None;
        self.phase = phase;
    }
}

impl ModelingTool for ExtrudeOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "extrude", icon: icons::EXTRUDE, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.phase = ExtrudePhase::AwaitingSelection;
    }

    fn selection_mode(&self) -> SelectionMode {
        // Extrude operates on a face (→ solid pad) or an edge (→ face).
        SelectionMode::SubGeometry(SelectionKinds::FACE | SelectionKinds::EDGE)
    }

    /// A dialled-in length is a finished extrusion, so leaving the tool commits it.
    /// A zero length has no preview and is nothing to commit.
    fn finalize(&mut self, selection: &mut SelectionManager) -> anyhow::Result<()> {
        if self.phase == ExtrudePhase::Extruding && self.length.abs() > MIN_LENGTH {
            self.apply()?;
            selection.clear();
        }
        Ok(())
    }

    fn is_finished(&self) -> bool {
        matches!(self.phase, ExtrudePhase::Done | ExtrudePhase::Cancelled)
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor_target
    }
}

impl Operator for ExtrudeOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            // While awaiting a target, pick up a face/edge selection (made before or
            // after the tool was activated) and start extruding.
            DeviceEvent::Update { .. } => {
                if self.phase == ExtrudePhase::AwaitingSelection {
                    if let Some(target) = Self::selected_target(ctx.selection) {
                        self.begin(target);
                    }
                }
                false
            }
            DeviceEvent::CursorMoved { position } => {
                if self.phase == ExtrudePhase::Extruding {
                    self.update_length((position.0 as f32, position.1 as f32), ctx);
                }
                false
            }
            // Right-click finalizes (matching the Boolean/Line right-click convention).
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } => {
                if self.phase == ExtrudePhase::Extruding {
                    if let Err(e) = self.apply() {
                        log::error!("Extrude failed: {e}");
                    } else {
                        ctx.selection.clear();
                    }
                    return true;
                }
                false
            }
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) if self.phase == ExtrudePhase::Extruding => {
                        if let Err(e) = self.apply() {
                            log::error!("Extrude failed: {e}");
                        } else {
                            ctx.selection.clear();
                        }
                        true
                    }
                    Key::Named(NamedKey::Escape) if self.phase == ExtrudePhase::Extruding => {
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
        "Extrude"
    }
}
