mod boolean;
mod cursor;
mod delete;
mod document;
mod extrude;
mod grid;
mod history;
#[cfg(not(target_arch = "wasm32"))]
mod io;
mod loft;
mod notifications;
mod operators;
mod platform;
mod preview;
mod snap;
mod tool;
mod tool_manager;
mod ui;
mod undo;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use egui_wgpu::RendererOptions;

use duck_engine_viewer::{AxisTriadConfig, OffscreenViewer, ViewId, ViewLayout, WindowSurface};
use duck_engine_viewer::event::Event;
use duck_engine_viewer::input::ElementState;
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator, TransformMode};
use duck_engine_viewer::common::{
    Vector3, InnerSpace
};
use duck_engine_viewer::scene::{PositionedCamera, Scene};

use crate::operators::{
    BooleanOperator, BoxOperator, CircleOperator, ConstructionOptions, CurveOperator,
    CylinderOperator, ExtrudeOperator, LineOperator, LoftOperator, RectangleOperator,
    SphereOperator, TransformTool,
};
use crate::delete::DeleteOperator;
use crate::notifications::Notifications;
use crate::undo::{UndoAction, UndoRedoOperator};
use crate::platform::Host;
use crate::tool_manager::ToolManager;
use crate::ui::{ModelerUi, UiAction};

use document::Document;

/// Owns all rendering state: egui context + GPU renderer, the window surface
/// egui presents to, and the [`OffscreenViewer`] that renders the 3D scene into
/// a texture displayed inside the central panel.
struct ViewerState<'a> {
    // Field order is drop order. The egui GPU resources go before the viewer
    // and surface they were built against; `host` goes last, so the window
    // outlives everything that borrowed a handle from it.
    egui_renderer: egui_wgpu::Renderer,
    egui_ctx: egui::Context,
    ui: ModelerUi,
    /// Stable egui texture id the offscreen color texture is registered under.
    /// Re-pointed (not re-created) when the offscreen texture is resized.
    scene_texture_id: egui::TextureId,
    /// The central-panel image rect in physical pixels, stashed each frame for
    /// input routing. `None` until the first frame is built.
    viewport_rect: Option<egui::Rect>,
    /// True while a pointer drag that began inside the viewport is in progress;
    /// keeps routing to the viewer even if the cursor crosses a panel.
    viewport_drag_active: bool,
    /// Latest cursor position in physical pixels (window space).
    last_cursor: Option<(f32, f32)>,
    viewer: OffscreenViewer,
    /// The single view filling the central panel.
    view_id: ViewId,
    surface: WindowSurface<'a>,
    host: Host,

    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    notifications: Notifications,
    tools: ToolManager,

    delete_op: Arc<Mutex<DeleteOperator>>,
    undo_op: Arc<Mutex<UndoRedoOperator>>,

    /// The construction grid currently installed in the scene; replaced when
    /// the construction plane or grid settings change.
    grid: Option<grid::Grid>,
}

impl ViewerState<'static> {
    /// Build the application on top of an already-created surface and platform
    /// host. `egui_ctx` is the same context the host's input integration was
    /// built against.
    fn new(egui_ctx: egui::Context, surface: WindowSurface<'static>, host: Host) -> Self {
        let (width, height) = host.surface_size();
        let mut viewer = OffscreenViewer::from_gpu(
            surface.gpu(),
            surface.format(),
            width,
            height,
            surface.sample_count(),
            surface.has_compute(),
        );

        egui_extras::install_image_loaders(&egui_ctx);

        let mut egui_renderer = egui_wgpu::Renderer::new(
            viewer.device(),
            surface.format(),
            RendererOptions::default(),
        );

        // Register the offscreen scene texture once; the id is stable and
        // re-pointed on resize via `update_egui_texture_from_wgpu_texture`.
        let scene_texture_id = egui_renderer.register_native_texture(
            viewer.device(),
            viewer.texture_view(),
            wgpu::FilterMode::Linear,
        );

        let scene = Scene::default();
        let view_id = viewer.add_view("main", scene.clone(), ViewLayout::FULL);

        let construction_options = Rc::new(RefCell::new(ConstructionOptions::new()));
        let document = Arc::new(Mutex::new(Document::new(scene)));
        let notifications = Notifications::default();

        let mut view = viewer.view_mut(view_id).expect("main view");
        let dispatcher = view.dispatcher_mut();
        let sel_op = Arc::new(Mutex::new(SelectionOperator::new()));
        dispatcher.push_back(sel_op.clone());
        dispatcher.push_back(Arc::new(Mutex::new(NavigationOperator::new())));

        let mut tools = ToolManager::new(sel_op);
        tools.install(dispatcher);

        // Behind the tool host so an active tool keeps any key it consumes.
        let delete_op = Arc::new(Mutex::new(DeleteOperator::new()));
        dispatcher.push_back(Arc::clone(&delete_op));
        let undo_op = Arc::new(Mutex::new(UndoRedoOperator::new()));
        dispatcher.push_back(Arc::clone(&undo_op));
        drop(view);

        viewer.add_axis_triad(view_id, AxisTriadConfig::default());

        tools.register(TransformTool::new(TransformMode::Translate, Rc::clone(&construction_options), Arc::clone(&document), notifications.clone()));
        tools.register(TransformTool::new(TransformMode::Rotate, Rc::clone(&construction_options), Arc::clone(&document), notifications.clone()));
        tools.register(TransformTool::new(TransformMode::Scale, Rc::clone(&construction_options), Arc::clone(&document), notifications.clone()));
        tools.register(SphereOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(BoxOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(CylinderOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(RectangleOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(LineOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(CurveOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(CircleOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(BooleanOperator::new(Rc::clone(&construction_options), Arc::clone(&document), notifications.clone()));
        tools.register(ExtrudeOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(LoftOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));

        Self {
            egui_renderer,
            egui_ctx,
            ui: ModelerUi::default(),
            scene_texture_id,
            viewport_rect: None,
            viewport_drag_active: false,
            last_cursor: None,
            viewer,
            view_id,
            surface,
            host,
            construction_options,
            document,
            notifications,
            tools,
            delete_op,
            undo_op,
            grid: None,
        }
    }

    fn set_default_scene(&mut self) {
        let mut scene = Scene::default();

        // Default camera; lighting comes from the view's headlights.
        let eye = [75.0, 50.0, 75.0].into();
        let target = [0.0, 0.0, 0.0].into();
        let forward: Vector3 = target - eye;
        let right = forward.cross([0.0, 1.0, 0.0].into()).normalize();
        let up = right.cross(forward);

        let size = self.viewer.size();

        let camera = PositionedCamera {
            eye,
            target,
            up,
            aspect: size.0 as f32 / size.1 as f32,
            fovy: 35.0,
            znear: 1.0,
            zfar: 5_000f32,
            ortho: false
        };

        let coptions = self.construction_options.borrow();
        self.grid =
            Some(grid::Grid::add_to_scene(&mut scene, &coptions.grid, &coptions.construction_plane));
        drop(coptions);

        self.viewer.set_view_scene(self.view_id, scene.clone());
        self.viewer.view_mut(self.view_id).expect("main view").set_camera(camera);
        self.document.lock().unwrap().set_scene(scene);
    }
}

impl<'a> ViewerState<'a> {
    /// A clone of the main view's scene handle.
    fn scene(&self) -> Scene {
        self.viewer.view(self.view_id).expect("main view").scene()
    }

    /// Replaces the scene's grid visuals to match the current construction
    /// plane and grid settings.
    fn rebuild_grid(&mut self) {
        let scene = self.scene();
        if let Some(grid) = self.grid.take() {
            grid.remove_from_scene(&scene);
        }
        let coptions = self.construction_options.borrow();
        self.grid =
            Some(grid::Grid::add_to_scene(&scene, &coptions.grid, &coptions.construction_plane));
    }

    /// Resize the window surface. The offscreen viewer is sized from the
    /// central panel each frame instead, not from here.
    fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface.resize(width, height);
    }

    /// Record the latest absolute cursor position, in physical pixels.
    fn set_cursor(&mut self, x: f32, y: f32) {
        self.last_cursor = Some((x, y));
    }

    fn egui_wants_pointer(&self) -> bool {
        self.egui_ctx.is_using_pointer()
    }

    /// Feed an event straight to the viewer, bypassing viewport routing.
    /// Feed an event straight to the viewer, bypassing viewport routing. Used
    /// for relative mouse motion, which both platforms deliver outside the
    /// normal routing path.
    fn viewer_handle_event(&mut self, event: &Event) {
        self.viewer.handle_event(event);
    }

    /// Route a normalized input event: events belonging to the 3D viewport go
    /// to the viewer with pointer-capture semantics and viewport-local
    /// coordinates; the rest are left to egui, which has already seen them.
    fn route_input(&mut self, event: Event) {
        if self.should_route_to_viewport(&event) {
            let event = self.to_viewport_local(event);
            self.viewer.handle_event(&event);
        }
    }

    /// Whether the latest cursor position sits inside the 3D viewport rect.
    fn cursor_in_viewport(&self) -> bool {
        match (self.viewport_rect, self.last_cursor) {
            (Some(rect), Some((x, y))) => rect.contains(egui::pos2(x, y)),
            _ => false,
        }
    }

    /// Decide whether a converted event should be routed to the 3D viewer,
    /// updating the pointer-capture flag on press/release.
    fn should_route_to_viewport(&mut self, event: &Event) -> bool {
        use duck_engine_viewer::event::DeviceEvent as DE;
        match event {
            Event::Device(DE::MouseInput { state, .. }) => match state {
                ElementState::Pressed => {
                    if self.cursor_in_viewport() {
                        self.viewport_drag_active = true;
                        true
                    } else {
                        false
                    }
                }
                ElementState::Released => {
                    if self.viewport_drag_active {
                        self.viewport_drag_active = false;
                        true
                    } else {
                        false
                    }
                }
            },
            Event::Device(DE::CursorMoved { .. }) => {
                self.viewport_drag_active
                    || (self.cursor_in_viewport() && !self.egui_ctx.is_using_pointer())
            }
            Event::Device(DE::MouseWheel { .. }) => self.cursor_in_viewport(),
            Event::Device(DE::KeyboardInput { .. }) => !self.egui_ctx.wants_keyboard_input(),
            _ => false,
        }
    }

    /// Translate absolute cursor coordinates from window space into the 3D
    /// viewport's local pixel space (its top-left is the offscreen origin).
    /// Only `CursorMoved` carries an absolute position; drags/clicks are
    /// synthesized from it downstream, so translating it is sufficient.
    fn to_viewport_local(&self, event: Event) -> Event {
        use duck_engine_viewer::event::DeviceEvent as DE;
        match (event, self.viewport_rect) {
            (Event::Device(DE::CursorMoved { position }), Some(rect)) => {
                Event::Device(DE::CursorMoved {
                    position: (position.0 - rect.min.x as f64, position.1 - rect.min.y as f64),
                })
            }
            (event, _) => event,
        }
    }

    /// Applies a deferred undo/redo request: cancels any in-progress tool
    /// (deactivation tears down previews and restores hidden sources), clears
    /// the selection — re-tessellation invalidates sub-geometry indices and
    /// undo may remove selected nodes outright — then replays the step.
    fn apply_undo(&mut self, action: UndoAction) {
        self.tools.activate(None);
        self.viewer
            .view_mut(self.view_id)
            .expect("main view")
            .selection_mut()
            .clear();
        let (result, verb) = match action {
            UndoAction::Undo => (self.document.lock().unwrap().undo(), "Undid"),
            UndoAction::Redo => (self.document.lock().unwrap().redo(), "Redid"),
        };
        match result {
            Ok(Some(label)) => self.notifications.info(format!("{verb} {label}")),
            Ok(None) => {
                let noun = if action == UndoAction::Undo { "undo" } else { "redo" };
                self.notifications.info(format!("Nothing to {noun}"));
            }
            Err(e) => {
                log::error!("{verb} failed: {e:#}");
                self.notifications.error(format!("{verb} failed: {e}"));
            }
        }
    }

    /// Returns true when the user asked to quit via the menu.
    fn handle_redraw(&mut self) -> bool {
        self.viewer.update();

        // Build the egui frame: docked panels, then the central panel holding
        // the (stable) 3D scene texture. The central image rect is captured to
        // size the offscreen viewer and route viewport input.
        let raw_input = self.host.take_egui_input();
        let egui_ctx = self.egui_ctx.clone();
        let scene_texture_id = self.scene_texture_id;
        let mut viewport_rect = None;
        let mut ui_actions = Vec::new();
        let mut view = self.viewer.view_mut(self.view_id).expect("main view");
        // The UI edits a copy of the camera; it is written back on
        // `UiAction::CameraChanged` below.
        let mut ui_camera = view.camera().clone();
        let full_output = egui_ctx.run(raw_input, |ctx| {
            ui_actions = self.ui.show(
                ctx,
                &self.document,
                &mut ui_camera,
                &self.construction_options,
                view.selection_mut(),
                &mut self.tools,
                &self.notifications,
            );
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let size = ui.available_size();
                    let image =
                        egui::Image::new(egui::load::SizedTexture::new(scene_texture_id, size));
                    viewport_rect = Some(ui.add(image).rect);
                });
        });
        drop(view);
        self.host.handle_platform_output(full_output.platform_output.clone());

        // A delete request aborts any in-progress tool (deactivate tears down
        // its preview and restores hidden sources) before the parts go away.
        if self.delete_op.lock().unwrap().take_pending() {
            self.tools.activate(None);
            let mut view = self.viewer.view_mut(self.view_id).expect("main view");
            let n = delete::delete_selected_parts(&self.document, view.selection_mut());
            if n > 0 {
                let plural = if n == 1 { "" } else { "s" };
                self.notifications.info(format!("Deleted {n} part{plural}"));
            }
        }

        let pending_undo = self.undo_op.lock().unwrap().take_pending();
        if let Some(action) = pending_undo {
            self.apply_undo(action);
        }

        // After egui so a panel-driven finish (e.g. boolean Apply) cedes back
        // to selection in the same frame.
        let scene = self.scene();
        self.tools.update(&scene);

        // UI actions run outside the frame closure: the file dialogs block.
        for action in ui_actions {
            match action {
                #[cfg(not(target_arch = "wasm32"))]
                UiAction::ImportCad => {
                    let options = self.construction_options.borrow().geometry_options.clone();
                    if let Err(e) = io::import_cad_dialog(&self.document, &options) {
                        log::error!("CAD import failed: {e:#}");
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                UiAction::ExportCad => {
                    if let Err(e) = io::export_cad_dialog(&self.document) {
                        log::error!("CAD export failed: {e:#}");
                    }
                }
                // STEP/IGES needs OCCT's TKDESTEP, which is excluded from the
                // web build, and a file picker the canvas does not have yet.
                #[cfg(target_arch = "wasm32")]
                UiAction::ImportCad | UiAction::ExportCad => {
                    self.notifications.info("CAD file transfer is not available in the browser yet")
                }
                UiAction::Undo => self.apply_undo(UndoAction::Undo),
                UiAction::Redo => self.apply_undo(UndoAction::Redo),
                UiAction::ConstructionChanged => self.rebuild_grid(),
                UiAction::TessellationChanged => {
                    let show = self.construction_options.borrow().geometry_options.show_seam_edges;
                    self.document.lock().unwrap().set_seam_edges_visible(show);
                }
                UiAction::CameraChanged => {
                    let mut view = self.viewer.view_mut(self.view_id).expect("main view");
                    view.set_camera(ui_camera.clone());
                }
                UiAction::Quit => return true,
            }
        }

        // Reconcile the offscreen texture size with the central panel, then
        // re-point the (stable) egui texture id at the new view.
        let ppp = full_output.pixels_per_point;
        self.viewport_rect = viewport_rect.map(|r| {
            egui::Rect::from_min_size(
                egui::pos2(r.min.x * ppp, r.min.y * ppp),
                egui::vec2(r.width() * ppp, r.height() * ppp),
            )
        });
        if let Some(rect) = self.viewport_rect {
            let w = (rect.width().round() as u32).max(1);
            let h = (rect.height().round() as u32).max(1);
            if (w, h) != self.viewer.size() {
                self.viewer.resize(w, h);
                self.egui_renderer.update_egui_texture_from_wgpu_texture(
                    self.viewer.device(),
                    self.viewer.texture_view(),
                    wgpu::FilterMode::Linear,
                    self.scene_texture_id,
                );
            }
        }

        // Render the 3D scene into the offscreen texture (own encoder+submit).
        if let Err(e) = self.viewer.render() {
            log::error!("Offscreen render error: {}", e);
        }

        // Present: egui paints the whole window (including the scene image)
        // into the surface.
        match self.surface.acquire() {
            Ok(output) => {
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = self.viewer.device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("egui Encoder") },
                );
                self.render_egui_overlay(&full_output, ppp, &mut encoder, &view);
                self.viewer.queue().submit(std::iter::once(encoder.finish()));
                output.present();
            }
            Err(e) => log::error!("Surface acquire error: {}", e),
        }

        self.host.request_redraw();
        false
    }

    /// Render the full egui frame (panels + the 3D scene image) into `view`.
    fn render_egui_overlay(
        &mut self,
        full_output: &egui::FullOutput,
        pixels_per_point: f32,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let device = self.viewer.device();
        let queue = self.viewer.queue();

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let clipped_primitives =
            self.egui_ctx.tessellate(full_output.shapes.clone(), full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: {
                let (w, h) = self.surface.size();
                [w, h]
            },
            pixels_per_point,
        };

        self.egui_renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &clipped_primitives,
                &screen_descriptor,
            );
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

fn main() {
    platform::run();
}
