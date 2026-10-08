mod construction;
mod cursor;
mod delete;
mod document;
mod grid;
mod history;
#[cfg(not(target_arch = "wasm32"))]
mod io;
mod notifications;
mod ops;
mod platform;
mod preview;
mod shortcuts;
mod snap;
#[cfg(test)]
mod testing;
mod tools;
mod ui;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use egui_wgpu::RendererOptions;

use duck_engine_viewer::{AxisTriadConfig, OffscreenViewer, ViewId, ViewLayout, WindowSurface};
use duck_engine_viewer::event::Event;
use duck_engine_viewer::input::ElementState;
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator, TransformMode};
use duck_engine_viewer::common::{
    InnerSpace, Real, RgbaColor, Vector3
};
use duck_engine_viewer::scene::{PositionedCamera, Projection, Scene};

use crate::construction::ConstructionOptions;
use crate::tools::{
    BooleanTool, BoxTool, CircleTool, CylinderTool, DraftTool, DuplicateTool, ExtrudeTool,
    FilletTool, HollowTool, LoftTool, PathKind, PathTool, RectangleTool, SphereTool, ThickenTool,
    ToolId, ToolManager, TransformTool, Workspace,
};
use crate::notifications::Notifications;
use crate::platform::Host;
use crate::shortcuts::Shortcuts;
use crate::ui::ModelerUi;

use document::Document;

/// Viewport clear color. A cool dark grey.
const VIEWPORT_BACKGROUND: RgbaColor = RgbaColor { r: 0.035, g: 0.040, b: 0.047, a: 1.0 };

/// What the app is asked to do, by the UI or a shortcut. Gathered over a frame
/// and carried out once its UI is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppAction {
    /// Switch to this tool, or with `None` back to selection.
    SwitchTool(Option<ToolId>),
    /// Delete the selected parts.
    Delete,
    Undo,
    Redo,
    ImportCad,
    ExportCad,
    /// The construction plane or grid settings changed; the grid visuals must
    /// be rebuilt to match.
    ConstructionChanged,
    /// The camera settings changed; the edited camera must be written back to
    /// the view.
    CameraChanged,
    /// The view snapped to look along an axis, from this unit offset of the
    /// eye from its target; the construction plane follows if set to.
    ViewSnapped(Vector3),
    /// The tessellation options changed; existing parts must be rebuilt to match.
    TessellationChanged,
    Quit,
}

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
    shortcuts: Arc<Mutex<Shortcuts>>,

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
            surface.capabilities(),
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

        let sel_op = Arc::new(Mutex::new(SelectionOperator::new()));
        let mut tools = ToolManager::new(Arc::clone(&sel_op), notifications.clone());
        let workspace = Workspace {
            document: Arc::clone(&document),
            construction: Rc::clone(&construction_options),
            notifications: notifications.clone(),
        };
        tools.register(TransformTool::new(TransformMode::Translate, &workspace));
        tools.register(TransformTool::new(TransformMode::Rotate, &workspace));
        tools.register(TransformTool::new(TransformMode::Scale, &workspace));
        tools.register(DuplicateTool::new(&workspace));
        tools.register(SphereTool::new(&workspace));
        tools.register(BoxTool::new(&workspace));
        tools.register(CylinderTool::new(&workspace));
        tools.register(RectangleTool::new(&workspace));
        tools.register(PathTool::new(PathKind::Polyline, &workspace));
        tools.register(PathTool::new(PathKind::Spline, &workspace));
        tools.register(CircleTool::new(&workspace));
        tools.register(BooleanTool::new(&workspace));
        tools.register(ExtrudeTool::new(&workspace));
        tools.register(FilletTool::new(&workspace));
        tools.register(DraftTool::new(&workspace));
        tools.register(ThickenTool::new(&workspace));
        tools.register(HollowTool::new(&workspace));
        tools.register(LoftTool::new(&workspace));
        let shortcuts = Arc::new(Mutex::new(Shortcuts::new(&tools.palette_entries())));

        let mut view = viewer.view_mut(view_id).expect("main view");
        view.set_background_color(VIEWPORT_BACKGROUND);
        let dispatcher = view.dispatcher_mut();
        dispatcher.push_back(sel_op);
        dispatcher.push_back(Arc::new(Mutex::new(NavigationOperator::new())));
        tools.install(dispatcher);
        // Behind the tool host, so an active tool keeps any key it consumes.
        dispatcher.push_back(Arc::clone(&shortcuts));
        drop(view);

        viewer.add_axis_triad(view_id, AxisTriadConfig::default());

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
            shortcuts,
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
            aspect: size.0 as Real / size.1 as Real,
            projection: Projection::Orthographic { half_height: 35., half_depth: 1000. },
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

    /// Puts back the construction plane a view snap replaced once the camera
    /// has settled outside the snapped view.
    fn check_snapped_view(&mut self) {
        let view = self.viewer.view(self.view_id).expect("main view");
        if view.camera_in_transition() {
            return;
        }
        let toward_eye = -view.camera().forward();
        if self.construction_options.borrow_mut().leave_view(toward_eye) {
            self.rebuild_grid();
        }
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

    /// Replays a step of history with `step`, undo or redo, reporting it as
    /// `verb` and the step's label, or as `nothing` when there is no step. Any
    /// in-progress tool is discarded and the selection cleared first.
    fn replay(
        &mut self,
        step: fn(&mut Document) -> anyhow::Result<Option<String>>,
        verb: &str,
        nothing: &str,
    ) {
        // Discarded, not committed, so the step replayed is the one the user
        // meant.
        self.tools.discard_active();
        // Re-tessellation invalidates sub-geometry indices, and the step may
        // remove selected nodes outright.
        self.viewer.view_mut(self.view_id).expect("main view").selection_mut().clear();
        let result = step(&mut self.document.lock().unwrap());
        match result {
            Ok(Some(label)) => self.notifications.info(format!("{verb} {label}")),
            Ok(None) => self.notifications.info(nothing),
            Err(e) => self.notifications.failure(verb, &e),
        }
    }

    /// Deletes the selected parts, discarding any in-progress tool first.
    fn delete_selection(&mut self) {
        // Deactivating the tool tears down its preview, showing again what it
        // hid, before the parts go away.
        self.tools.discard_active();
        let mut view = self.viewer.view_mut(self.view_id).expect("main view");
        let deleted = delete::delete_selected_parts(&self.document, view.selection_mut());
        if deleted > 0 {
            let plural = if deleted == 1 { "" } else { "s" };
            self.notifications.info(format!("Deleted {deleted} part{plural}"));
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
        // `AppAction::CameraChanged` below.
        let mut ui_camera = view.camera().clone();
        let full_output = egui_ctx.run(raw_input, |ctx| {
            ui_actions = self.ui.show(
                ctx,
                &self.document,
                &mut ui_camera,
                &self.construction_options,
                view.selection_mut(),
                &self.tools,
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

        // The shortcuts pressed since the last frame, then what the UI asked
        // for, outside the frame closure: the file dialogs block.
        let mut actions = self.shortcuts.lock().unwrap().take();
        actions.extend(ui_actions);
        for action in actions {
            match action {
                AppAction::SwitchTool(tool) => {
                    let mut view = self.viewer.view_mut(self.view_id).expect("main view");
                    self.tools.activate(tool, view.selection_mut());
                }
                AppAction::Delete => self.delete_selection(),
                AppAction::Undo => self.replay(Document::undo, "Undid", "Nothing to undo"),
                AppAction::Redo => self.replay(Document::redo, "Redid", "Nothing to redo"),
                #[cfg(not(target_arch = "wasm32"))]
                AppAction::ImportCad => {
                    let options = self.construction_options.borrow().geometry_options.clone();
                    if let Err(e) = io::import_cad_dialog(&self.document, &options) {
                        self.notifications.failure("CAD import", &e);
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                AppAction::ExportCad => {
                    if let Err(e) = io::export_cad_dialog(&self.document) {
                        self.notifications.failure("CAD export", &e);
                    }
                }
                // STEP/IGES needs OCCT's TKDESTEP, which is excluded from the
                // web build, and a file picker the canvas does not have yet.
                #[cfg(target_arch = "wasm32")]
                AppAction::ImportCad | AppAction::ExportCad => {
                    self.notifications.info("CAD file transfer is not available in the browser yet")
                }
                AppAction::ConstructionChanged => self.rebuild_grid(),
                AppAction::TessellationChanged => {
                    let show = self.construction_options.borrow().geometry_options.show_seam_edges;
                    self.document.lock().unwrap().set_seam_edges_visible(show);
                }
                AppAction::CameraChanged => {
                    let mut view = self.viewer.view_mut(self.view_id).expect("main view");
                    view.set_camera(ui_camera.clone());
                }
                AppAction::ViewSnapped(toward_eye) => {
                    if self.construction_options.borrow_mut().face_view(toward_eye) {
                        self.rebuild_grid();
                    }
                }
                AppAction::Quit => return true,
            }
        }
        // After the actions, so a snap this frame is in place before its view
        // is checked.
        self.check_snapped_view();

        // After the UI and its actions, so a tool they finished — by a panel's
        // Apply, say — cedes back to selection in the same frame.
        let scene = self.scene();
        let mut view = self.viewer.view_mut(self.view_id).expect("main view");
        self.tools.update(&scene, view.selection_mut());
        drop(view);

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
