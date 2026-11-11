mod construction;
mod cursor;
mod delete;
mod document;
mod grid;
mod history;
mod io;
mod labels;
mod materials;
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
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use egui_wgpu::RendererOptions;

use duck_engine_viewer::{AxisTriadConfig, OffscreenViewer, ViewId, ViewLayout, WindowSurface};
use duck_engine_viewer::event::Event;
use duck_engine_viewer::input::ElementState;
use duck_engine_viewer::operator::{NavigationOperator, SelectionKinds, SelectionOperator};
use duck_engine_viewer::common::{
    InnerSpace, Real, RgbaColor, Vector3
};
use duck_engine_viewer::scene::{PositionedCamera, Projection, Scene};

use crate::construction::ConstructionOptions;
use crate::tools::{
    BooleanTool, BoxTool, CircleTool, CylinderTool, DraftTool, DuplicateTool, ExtrudeTool,
    FilletTool, HollowTool, JoinTool, LoftTool, PathKind, PathTool, RadialArrayTool, RectangleTool,
    RectangularArrayTool, RevolveTool, SphereTool, ThickenTool, ToolId, ToolManager,
    TransformMode, TransformTool, UnjoinTool, Workspace,
};
use crate::notifications::Notifications;
use crate::platform::Host;
use crate::shortcuts::Shortcuts;
use crate::ui::{Lighting, ModelerUi};

use document::Document;

/// Viewport clear color. A cool dark grey.
const VIEWPORT_BACKGROUND: RgbaColor = RgbaColor { r: 0.035, g: 0.040, b: 0.047, a: 1.0 };

/// What the app is asked to do, by the UI or a shortcut. Gathered over a frame
/// and carried out once its UI is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppAction {
    /// Switch to this tool, or with `None` back to selection.
    SwitchTool(Option<ToolId>),
    /// Make `kind` the only kind a click selects, or with `extend` add or
    /// remove it.
    SelectableKind { kind: SelectionKinds, extend: bool },
    /// Delete the selected parts.
    Delete,
    Undo,
    Redo,
    /// Ask for a document file to replace the document with.
    Open,
    /// Save the document to its file, asking where only if it has none.
    Save,
    /// Save the document wherever the user picks.
    SaveAs,
    ImportCad,
    ExportCad,
    /// Ask for an HDR image to light the scene with.
    LoadEnvironment,
    /// The construction plane or grid settings changed; the grid visuals must
    /// be rebuilt to match.
    ConstructionChanged,
    /// The camera settings changed; the edited camera must be written back to
    /// the view.
    CameraChanged,
    /// The lighting changed; the edited lighting must be written back to the
    /// view.
    LightingChanged,
    /// The view snapped to look along an axis, from this unit offset of the
    /// eye from its target; the construction plane follows if set to.
    ViewSnapped(Vector3),
    /// The tessellation options changed; existing parts must be rebuilt to match.
    TessellationChanged,
    #[cfg(not(target_os = "emscripten"))]
    Quit,
}

/// What a file the user is asked to pick is for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilePurpose {
    /// A document file to replace the document with.
    Document,
    /// STEP/IGES parts to import into the document.
    CadImport,
    /// An HDR image to light the scene with.
    Environment,
}

/// The outcome of reading a file the user picked.
///
/// The host reads it in a callback that cannot reach the rest of the app, so
/// what is left to do once the callback returns is carried out in this.
enum Received {
    /// The file was used up in the callback, by an import or an environment
    /// load. `operation` names it in a failure notice, e.g. "CAD import".
    Handled { operation: &'static str, result: anyhow::Result<()> },
    /// A document file was read from `path`, still to be installed.
    Document { path: PathBuf, result: anyhow::Result<(Document, io::Session)> },
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
    /// The document file last saved or opened, which Save writes to.
    document_file: Option<PathBuf>,
}

impl ViewerState<'static> {
    /// Build the application on top of an already-created surface and platform
    /// host. `egui_ctx` is the same context the host's input integration was
    /// built against.
    fn new(egui_ctx: egui::Context, surface: WindowSurface<'static>, host: Host) -> Self {
        let (width, height) = host.surface_size();
        let mut viewer = OffscreenViewer::from_gpu(
            surface.gpu(),
            width,
            height,
            surface.sample_count(),
        );

        egui_extras::install_image_loaders(&egui_ctx);
        ui::install_fonts(&egui_ctx);

        let mut egui_renderer = egui_wgpu::Renderer::new(
            viewer.context().device(),
            surface.format(),
            RendererOptions::default(),
        );

        // Register the offscreen scene texture once; the id is stable and
        // re-pointed on resize via `update_egui_texture_from_wgpu_texture`.
        let scene_texture_id = egui_renderer.register_native_texture(
            viewer.context().device(),
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
        tools.register(RectangularArrayTool::new(&workspace));
        tools.register(RadialArrayTool::new(&workspace));
        tools.register(SphereTool::new(&workspace));
        tools.register(BoxTool::new(&workspace));
        tools.register(CylinderTool::new(&workspace));
        tools.register(RectangleTool::new(&workspace));
        tools.register(PathTool::new(PathKind::Polyline, &workspace));
        tools.register(PathTool::new(PathKind::Spline, &workspace));
        tools.register(CircleTool::new(&workspace));
        tools.register(BooleanTool::new(&workspace));
        tools.register(JoinTool::new(&workspace));
        tools.register(UnjoinTool::new(&workspace));
        tools.register(ExtrudeTool::new(&workspace));
        tools.register(RevolveTool::new(&workspace));
        tools.register(FilletTool::new(&workspace));
        tools.register(DraftTool::new(&workspace));
        tools.register(ThickenTool::new(&workspace));
        tools.register(HollowTool::new(&workspace));
        tools.register(LoftTool::new(&workspace));
        let shortcuts = Arc::new(Mutex::new(Shortcuts::new(&tools.palette_entries())));

        let view = viewer.view_mut(view_id).expect("main view");
        view.set_background_color(VIEWPORT_BACKGROUND);
        let dispatcher = view.dispatcher_mut();
        dispatcher.push_back(sel_op);
        dispatcher.push_back(Arc::new(Mutex::new(NavigationOperator::new())));
        tools.install(dispatcher);
        // Behind the tool host, so an active tool keeps any key it consumes.
        dispatcher.push_back(Arc::clone(&shortcuts));

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
            document_file: None,
        }
    }

    fn set_default_scene(&mut self) {
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

        let mut document = Document::new(Scene::default());
        document.add_preset_materials();
        let session = io::Session { camera, ..self.session() };
        self.install_document(document, session, None);
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

    /// Whether the latest cursor position sits inside the 3D viewport rect and
    /// no egui window or popup covers it.
    fn cursor_in_viewport(&self) -> bool {
        let (Some(rect), Some((x, y))) = (self.viewport_rect, self.last_cursor) else {
            return false;
        };
        let cursor = egui::pos2(x, y);
        // Panels, the viewport's own included, are background layers; anything
        // above them covers the viewport.
        let covered = self
            .egui_ctx
            .layer_id_at(cursor / self.egui_ctx.pixels_per_point())
            .is_some_and(|layer| layer.order != egui::Order::Background);
        rect.contains(cursor) && !covered
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
        self.viewer.selection_mut(self.view_id).expect("main view").clear();
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
        let selection = self.viewer.selection_mut(self.view_id).expect("main view");
        let deleted = delete::delete_selected_parts(&self.document, selection);
        if deleted > 0 {
            let plural = if deleted == 1 { "" } else { "s" };
            self.notifications.info(format!("Deleted {deleted} part{plural}"));
        }
    }

    /// The session to save with the document: the view's camera and lighting,
    /// and the construction plane and grid.
    fn session(&self) -> io::Session {
        let view = self.viewer.view(self.view_id).expect("main view");
        let construction = self.construction_options.borrow();
        io::Session {
            camera: view.camera().clone(),
            lighting: Lighting::of(&self.viewer, self.view_id),
            construction_plane: construction.own_plane(),
            grid: construction.grid.clone(),
        }
    }

    /// Replaces the document with `document`, drawn in its own scene, restores
    /// `session` around it, and makes `file` the one Save writes to.
    fn install_document(&mut self, mut document: Document, session: io::Session, file: Option<PathBuf>) {
        // The active tool's previews live in the outgoing scene, and tearing
        // them down locks the document.
        self.tools.discard_active();
        // Bodies keep showing what they showed.
        let hidden = self.document.lock().unwrap().hidden_primitives();
        document.set_hidden_primitives(hidden);
        let scene = document.scene().clone();
        // In place: the tools hold the same document.
        *self.document.lock().unwrap() = document;
        // The new scene comes with an empty selection and no lights.
        self.viewer.set_view_scene(self.view_id, scene);
        {
            let mut construction = self.construction_options.borrow_mut();
            construction.set_own_plane(session.construction_plane);
            construction.grid = session.grid;
        }
        // The grid went with the old scene.
        self.grid = None;
        self.rebuild_grid();
        let view = self.viewer.view_mut(self.view_id).expect("main view");
        view.set_camera(session.camera);
        view.cancel_camera_transition();
        // After the scene swap: the lights belong to the view's scene.
        session.lighting.apply(&mut self.viewer, self.view_id);
        let show = self.construction_options.borrow().geometry_options.show_seam_edges;
        self.document.lock().unwrap().set_seam_edges_visible(show);
        self.document_file = file;
    }

    /// Saves the document to the file it was last saved to or opened from, or
    /// with `ask` or no such file, wherever the user picks. The active tool
    /// commits first, as on switching tools, so the file holds what the view
    /// shows.
    fn save_document(&mut self, ask: bool) {
        let selection = self.viewer.selection_mut(self.view_id).expect("main view");
        self.tools.activate(None, selection);
        let session = self.session();
        let write = |path: &Path| io::save_document(path, &self.document.lock().unwrap(), session);
        let saved = match self.document_file.as_deref() {
            Some(path) if !ask => self.host.save_file_to(path, write).map(|()| Some(path.to_owned())),
            file => {
                let name = file.and_then(Path::file_name).map(|name| name.to_string_lossy());
                let name = name.as_deref().unwrap_or("untitled.duck");
                self.host.save_file(name, &[io::DOCUMENT_FILTER], write)
            }
        };
        match saved {
            Ok(Some(path)) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                self.notifications.info(format!("Saved {name}"));
                self.document_file = Some(path);
            }
            Ok(None) => {}
            Err(e) => self.notifications.failure("Save", &e),
        }
    }

    /// Writes all parts to a STEP/IGES file wherever the host puts it.
    fn export_cad(&self) -> anyhow::Result<()> {
        if self.document.lock().unwrap().is_empty() {
            log::warn!("Nothing to export: document has no parts");
            return Ok(());
        }
        let path = self.host.save_file("export.step", &io::EXPORT_FILTERS, |path| {
            io::export_cad_file(&self.document, path)
        });
        path.map(|_| ())
    }

    /// Returns true when the user asked to quit via the menu.
    fn handle_redraw(&mut self) -> bool {
        self.viewer.update();

        // Build the egui frame: docked panels, then the central panel holding
        // the (stable) 3D scene texture. The central image rect is captured to
        // size the offscreen viewer and route viewport input.
        let mut raw_input = self.host.take_egui_input();
        ui::keep_tab_for_viewport(&self.egui_ctx, &mut raw_input);
        let egui_ctx = self.egui_ctx.clone();
        let scene_texture_id = self.scene_texture_id;
        let mut viewport_rect = None;
        let mut ui_actions = Vec::new();
        // The UI edits copies of the camera and lighting; they are written
        // back on `AppAction::CameraChanged` and `LightingChanged` below.
        let mut ui_camera = self.viewer.view(self.view_id).expect("main view").camera().clone();
        let mut ui_lighting = Lighting::of(&self.viewer, self.view_id);
        // The active tool's text, placed by the camera the scene renders with.
        let labels = self.tools.labels();
        let mut label_camera = ui_camera.clone();
        let units = self.construction_options.borrow().units;
        let selection = self.viewer.selection_mut(self.view_id).expect("main view");
        let full_output = egui_ctx.run(raw_input, |ctx| {
            ui_actions = self.ui.show(
                ctx,
                &self.document,
                &mut ui_camera,
                &mut ui_lighting,
                &self.construction_options,
                selection,
                &self.tools,
                &self.notifications,
            );
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let size = ui.available_size();
                    let image =
                        egui::Image::new(egui::load::SizedTexture::new(scene_texture_id, size));
                    let rect = ui.add(image).rect;
                    viewport_rect = Some(rect);
                    // On the viewport's own layer, so windows over it cover
                    // the labels too. The image is resized to `rect` before
                    // rendering, so its camera takes that aspect.
                    label_camera.aspect = (rect.width() / rect.height().max(1.0)) as Real;
                    labels::paint(ui.painter(), rect, &label_camera, units, &labels);
                });
        });
        self.host.handle_platform_output(full_output.platform_output.clone());

        // The shortcuts pressed since the last frame, then what the UI asked
        // for, outside the frame closure: the file dialogs block.
        let mut actions = self.shortcuts.lock().unwrap().take();
        actions.extend(ui_actions);
        for action in actions {
            match action {
                AppAction::SwitchTool(tool) => {
                    let selection = self.viewer.selection_mut(self.view_id).expect("main view");
                    self.tools.activate(tool, selection);
                }
                AppAction::SelectableKind { kind, extend } => self.tools.select_kind(kind, extend),
                AppAction::Delete => self.delete_selection(),
                AppAction::Undo => self.replay(Document::undo, "Undid", "Nothing to undo"),
                AppAction::Redo => self.replay(Document::redo, "Redid", "Nothing to redo"),
                AppAction::Open => self.host.open_file(&io::DOCUMENT_FILTER, FilePurpose::Document),
                AppAction::Save => self.save_document(false),
                AppAction::SaveAs => self.save_document(true),
                AppAction::ImportCad => {
                    self.host.open_file(&io::IMPORT_FILTER, FilePurpose::CadImport)
                }
                AppAction::ExportCad => {
                    if let Err(e) = self.export_cad() {
                        self.notifications.failure("CAD export", &e);
                    }
                }
                AppAction::LoadEnvironment => {
                    self.host.open_file(&io::ENVIRONMENT_FILTER, FilePurpose::Environment)
                }
                AppAction::ConstructionChanged => self.rebuild_grid(),
                AppAction::TessellationChanged => {
                    let show = self.construction_options.borrow().geometry_options.show_seam_edges;
                    self.document.lock().unwrap().set_seam_edges_visible(show);
                }
                AppAction::CameraChanged => {
                    let view = self.viewer.view_mut(self.view_id).expect("main view");
                    view.set_camera(ui_camera.clone());
                }
                AppAction::LightingChanged => {
                    ui_lighting.apply(&mut self.viewer, self.view_id);
                }
                AppAction::ViewSnapped(toward_eye) => {
                    if self.construction_options.borrow_mut().face_view(toward_eye) {
                        self.rebuild_grid();
                    }
                }
                #[cfg(not(target_os = "emscripten"))]
                AppAction::Quit => return true,
            }
        }
        // A file opened through Open, ImportCad or LoadEnvironment: this frame
        // on native, some frames later in the browser. Read in the callback,
        // which the browser's copy of the file does not outlive.
        let scene = self.scene();
        let received = self.host.receive_opened_file(|purpose, path| match purpose {
            FilePurpose::Document => {
                Received::Document { path: path.to_owned(), result: io::open_document(path) }
            }
            FilePurpose::CadImport => {
                let options = &self.construction_options.borrow().geometry_options;
                let result = io::import_cad_file(&self.document, path, options);
                Received::Handled { operation: "CAD import", result }
            }
            FilePurpose::Environment => {
                let loaded = io::load_environment(&scene, path);
                if loaded.is_ok() {
                    let name = path.file_name().unwrap_or_default().to_string_lossy();
                    self.notifications.info(format!("Lit by {name}"));
                }
                Received::Handled { operation: "Environment load", result: loaded }
            }
        });
        match received {
            Some(Received::Document { path, result: Ok((document, session)) }) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                self.install_document(document, session, Some(path));
                self.notifications.info(format!("Opened {name}"));
            }
            Some(Received::Document { result: Err(e), .. }) => self.notifications.failure("Open", &e),
            Some(Received::Handled { operation, result: Err(e) }) => {
                self.notifications.failure(operation, &e)
            }
            Some(Received::Handled { result: Ok(()), .. }) | None => {}
        }
        // After the actions, so a snap this frame is in place before its view
        // is checked.
        self.check_snapped_view();

        // After the UI and its actions, so a tool they finished — by a panel's
        // Apply, say — cedes back to selection in the same frame.
        let scene = self.scene();
        let selection = self.viewer.selection_mut(self.view_id).expect("main view");
        self.tools.update(&scene, selection);

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
                    self.viewer.context().device(),
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
                let mut encoder = self.viewer.context().device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("egui Encoder") },
                );
                self.render_egui_overlay(&full_output, ppp, &mut encoder, &view);
                self.viewer.context().queue().submit(std::iter::once(encoder.finish()));
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
        let device = self.viewer.context().device();
        let queue = self.viewer.context().queue();

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
