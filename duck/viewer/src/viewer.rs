use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use duck_engine_common::Vector3;
use duck_engine_scene::{NodeFlags, NodeId};
use web_time::Instant;

#[cfg(feature = "streaming")]
use crate::streaming::ViewerStreamClient;

use crate::{
    event::{DeviceEvent, Event, EventContext, EventDispatcher},
    scene::{NodePayload, PositionedCamera, Scene},
    selection::SelectionManager,
    renderer::{Gpu, Renderer, HighlightQuery},
};

/// Surface-free core viewer: owns the GPU handles, renderer, scene, and event
/// handling. Drives rendering into an arbitrary [`wgpu::TextureView`] via
/// [`render_scene_to_view`](Self::render_scene_to_view); the concrete output
/// target (a window surface or an offscreen texture) is owned by a wrapper such
/// as [`SurfacedViewer`].
pub struct Viewer {
    gpu: Gpu,
    renderer: Renderer,
    scene: Arc<Mutex<Scene>>,
    selection: SelectionManager,
    dispatcher: EventDispatcher,
    /// Current cursor position in screen coordinates
    cursor_position: Option<(f32, f32)>,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
    #[cfg(feature = "streaming")]
    stream_client: Option<ViewerStreamClient>,
}

impl Viewer {
    /// Build a viewer around an existing GPU context.
    ///
    /// `color_format` is the format of the target the viewer will render into
    /// (a surface format or an offscreen texture format). Both [`SurfacedViewer`]
    /// and offscreen viewers construct the core through this builder.
    pub fn from_gpu(
        gpu: Gpu,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        let renderer = Renderer::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            color_format,
            width,
            height,
            sample_count,
            has_compute,
        );
        let scene = Arc::new(Mutex::new(Scene::new()));
        let dispatcher = EventDispatcher::new();

        let mut viewer = Self {
            gpu,
            renderer,
            scene,
            selection: SelectionManager::new(),
            dispatcher,
            cursor_position: None,
            last_update_time: None,
            #[cfg(feature = "streaming")]
            stream_client: None,
        };

        // Ensure there is always an active camera in the scene
        viewer.ensure_active_camera();

        viewer
    }

    /// Resize the render target. Updates the renderer's internal targets
    /// (depth/MSAA); the owning wrapper is responsible for resizing any surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize((width, height));
    }

    /// Handle a single event by dispatching it to registered handlers.
    pub fn handle_event(&mut self, event: &Event) {
        if let Event::Device(DeviceEvent::CursorMoved { position }) = event {
            self.cursor_position = Some((position.0 as f32, position.1 as f32));
        }
        if let Event::Device(DeviceEvent::Resized(physical_size)) = event {
            let (w, h) = *physical_size;
            if w > 0 && h > 0 {
                self.resize(w, h);
            }
        }

        let mut ctx = EventContext {
            size: self.renderer.size(),
            cursor_position: &mut self.cursor_position,
            scene: Arc::clone(&self.scene),
            selection: &mut self.selection,
            modifiers: Default::default(), // dispatcher overwrites this in dispatch()
            emit_queue: Vec::new(),
        };
        self.dispatcher.dispatch(event, &mut ctx);
    }

    /// Dispatch an Update event with delta_time since last update.
    ///
    /// Call this once per frame before rendering to enable smooth continuous
    /// operations like WASD movement in walk mode.
    ///
    /// The delta_time is automatically calculated from the time since the
    /// last call to update(). On the first call, a small default delta is used.
    pub fn update(&mut self) {
        let now = Instant::now();
        let delta_time = match self.last_update_time {
            Some(last) => now.duration_since(last).as_secs_f32(),
            None => 1.0 / 60.0, // Assume 60 FPS on first frame
        };
        self.last_update_time = Some(now);

        let event = Event::Device(DeviceEvent::Update { delta_time });
        self.handle_event(&event);

        #[cfg(feature = "streaming")]
        self.poll_stream();
    }

    /// Connect to a streaming server. Replaces any existing connection.
    #[cfg(feature = "streaming")]
    pub fn connect_stream(&mut self, addr: &str) -> anyhow::Result<()> {
        use duck_engine_streaming::SubscribeOptions;
        let camera = self.scene.lock().unwrap().active_camera_positioned(1.0).map(|cam| {
            let fwd = cam.forward();
            duck_engine_streaming::CameraHint {
                position: cam.eye.into(),
                forward: fwd.into(),
                fov_y_rad: cam.fovy.to_radians(),
            }
        });
        let client = ViewerStreamClient::connect(addr, SubscribeOptions { camera, ..Default::default() })?;
        self.stream_client = Some(client);
        Ok(())
    }

    /// Disconnect from the streaming server.
    #[cfg(feature = "streaming")]
    pub fn disconnect_stream(&mut self) {
        self.stream_client = None;
    }

    /// Returns `true` once the initial priority sync from the server is complete.
    #[cfg(feature = "streaming")]
    pub fn stream_sync_complete(&self) -> bool {
        self.stream_client.as_ref().map(|c| c.sync_complete).unwrap_or(false)
    }

    /// Drain pending scene updates from the streaming client. Called every frame from `update`.
    #[cfg(feature = "streaming")]
    fn poll_stream(&mut self) {
        use crate::streaming::PollResult;
        let mut scene = self.scene.lock().unwrap();
        let result = self.stream_client.as_mut().map(|c| c.poll(&mut *scene));
        match result {
            Some(PollResult::Disconnected) => { self.stream_client = None; }
            _ => {}
        }
    }

    /// Get a reference to the active camera.
    ///
    /// Panics if the scene has no active camera node.
    pub fn camera(&self) -> PositionedCamera {
        let (w, h) = self.renderer.size();
        let aspect = if h > 0 { w as f32 / h as f32 } else { 16.0 / 9.0 };
        self.scene.lock().unwrap().active_camera_positioned(aspect).expect("no active camera in scene")
    }

    /// Clones the active camera, passes it to `f` for mutation, then writes it back.
    pub fn with_camera_mut<F: FnOnce(&mut PositionedCamera)>(&mut self, f: F) {
        let mut cam = self.camera();
        f(&mut cam);
        self.set_camera(cam);
    }

    /// Replace the active camera.
    pub fn set_camera(&mut self, camera: PositionedCamera) {
        let mut scene = self.scene.lock().unwrap();
        let id = scene.active_camera().expect("no active camera in scene");
        scene.set_node_transform(id, camera.to_node_transform());
        scene.set_node_payload(id, NodePayload::Camera(camera.projection()));
    }

    /// Get the current viewport size as (width, height)
    pub fn size(&self) -> (u32, u32) {
        self.renderer.size()
    }

    /// Get the render target texture format.
    /// Useful for creating render pipelines that need to match the target format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.renderer.surface_format()
    }

    /// Replace the active rendering workflow.
    pub fn set_workflow(&mut self, workflow: Box<crate::renderer::SceneWorkflow>) {
        self.renderer.set_workflow(workflow);
    }

    /// Create a [`ShadedWorkflow`](crate::renderer::ShadedWorkflow) configured for this viewer.
    pub fn shaded_workflow(&mut self) -> crate::renderer::ShadedWorkflow {
        self.renderer.shaded_workflow()
    }

    /// Create a [`HiddenLineWorkflow`](crate::renderer::HiddenLineWorkflow) configured for this viewer.
    pub fn hidden_line_workflow(&mut self, config: crate::renderer::HiddenLineConfig) -> crate::renderer::HiddenLineWorkflow {
        self.renderer.hidden_line_workflow(config)
    }

    /// Get a reference to the wgpu device
    pub fn device(&self) -> &wgpu::Device {
        &self.gpu.device
    }

    /// Get a reference to the wgpu queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.gpu.queue
    }

    /// Returns a clone of the GPU handle pair, for sharing the device/queue with
    /// other owners (e.g. an offscreen viewer or an egui renderer).
    pub fn gpu(&self) -> Gpu {
        self.gpu.clone()
    }

    /// Returns a clone of the scene Arc, for sharing with other owners (e.g. Document).
    pub fn scene(&self) -> Arc<Mutex<Scene>> {
        Arc::clone(&self.scene)
    }

    /// Replace the viewer's scene Arc.
    ///
    /// This clears all scene-specific GPU resources and selection state
    /// to prevent stale data from persisting across scene changes.
    /// If the incoming scene has no active camera a default one is added.
    pub fn set_scene(&mut self, scene: Arc<Mutex<Scene>>) {
        self.scene = scene;
        self.selection.clear();
        self.renderer.clear_gpu_resources();
        self.ensure_active_camera();
        self.ensure_default_lights();
    }

    /// Clear the scene, removing all geometry, materials, textures, and
    /// associated GPU resources.
    ///
    /// This is the recommended way to reset the viewer before loading
    /// new content. It clears:
    /// - All scene nodes, instances, meshes, materials (except default), textures
    /// - All cached GPU resources (vertex buffers, texture views, material bind groups)
    /// - The current selection
    pub fn clear_scene(&mut self) {
        self.scene.lock().unwrap().clear();
        self.selection.clear();
        self.renderer.clear_gpu_resources();
        self.ensure_active_camera();
    }

    /// Adds a default camera node to the scene if no active camera is set.
    fn ensure_active_camera(&mut self) -> NodeId {
        let mut scene = self.scene.lock().unwrap();
        if let Some(camera_id) = scene.active_camera() {
            return camera_id;
        }
        let cam = PositionedCamera {
            eye: (0.0, 0.1, 0.2).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vector3::unit_y(),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.001,
            zfar: 100.0,
            ortho: false,
        };
        let id = scene.add_node(None, Some("Camera".to_string()), cam.to_node_transform(), NodeFlags::NONE)
            .expect("Failed to add default camera node");
        scene.set_node_payload(id, NodePayload::Camera(cam.projection()));
        scene.set_active_camera(Some(id));
        return id;
    }

    /// Adds default lights as children of the camera if the scene is otherwise unlit.
    fn ensure_default_lights(&mut self) {
        let has_lights = {
            let scene = self.scene.lock().unwrap();
            scene.has_light_nodes() || scene.active_environment_map().is_some()
        };
        if has_lights {
            return;
        }
        let camera = self.ensure_active_camera();
        self.scene.lock().unwrap().set_default_light_nodes(camera);
    }

    /// Get a reference to the selection manager
    pub fn selection(&self) -> &SelectionManager {
        &self.selection
    }

    /// Get a mutable reference to the selection manager
    pub fn selection_mut(&mut self) -> &mut SelectionManager {
        &mut self.selection
    }

    /// Get a reference to the event dispatcher
    pub fn dispatcher(&self) -> &EventDispatcher {
        &self.dispatcher
    }

    /// Get a mutable reference to the event dispatcher
    pub fn dispatcher_mut(&mut self) -> &mut EventDispatcher {
        &mut self.dispatcher
    }

    /// Prepare GPU resources for the scene ahead of rendering. Called by the
    /// owning wrapper before recording render passes.
    pub fn prepare_scene(&mut self) -> Result<(), anyhow::Error> {
        let mut scene = self.scene.lock().unwrap();
        self.renderer.prepare_scene(&mut *scene)
    }

    /// Returns a highlight query for the renderer if outline rendering is enabled.
    fn selection_for_render(selection: &SelectionManager) -> Option<&dyn HighlightQuery> {
        if selection.config().outline_enabled {
            Some(selection)
        } else {
            None
        }
    }

    /// Render the 3D scene to a specific view using a specific encoder.
    ///
    /// This is the low-level API used by all output targets (window surface,
    /// offscreen texture). It does not prepare the scene or submit the encoder;
    /// call [`prepare_scene`](Self::prepare_scene) beforehand and submit the
    /// encoder afterwards.
    pub fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), anyhow::Error> {
        let scene = self.scene.lock().unwrap();
        let highlight = Self::selection_for_render(&self.selection);
        self.renderer.render_scene_to_view(view, encoder, None, &*scene, highlight)
    }

    /// Render the scene from the given camera and read the result back into an
    /// RGBA image (blocking). For headless still-image / thumbnail rendering.
    pub fn render_to_image(&mut self, camera: &PositionedCamera) -> Result<image::RgbaImage, anyhow::Error> {
        let mut scene = self.scene.lock().unwrap();
        let highlight = Self::selection_for_render(&self.selection);
        self.renderer.render_scene_to_image(camera, &mut *scene, highlight)
    }
}

/// A configured window/canvas surface paired with the GPU context created for
/// it.
/// 
/// Owns the surface and its configuration; the [`Gpu`] handle is cloneable
/// and can be shared with other renderers (e.g. an [`OffscreenViewer`] whose
/// texture is displayed inside a UI panel on the same device).
///
/// This is the reusable surface + GPU bootstrap. [`SurfacedViewer`] builds its
/// core on top of it; applications that present through their own compositor
/// (e.g. egui) can hold a `WindowSurface` directly and drive an offscreen
/// viewer with `gpu()`.
pub struct WindowSurface<'a> {
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    sample_count: u32,
    has_compute: bool,
}

impl<'a> WindowSurface<'a> {
    /// Create and configure a surface for the given target, bootstrapping the
    /// wgpu instance, adapter, and device/queue.
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        // Create wgpu instance
        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::util::new_instance_with_webgpu_detection(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        }).await;

        let surface = instance.create_surface(surface_target).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let is_gl_backend = adapter.get_info().backend == wgpu::Backend::Gl;
        let downlevel_flags = adapter.get_downlevel_capabilities().flags;
        let has_compute = downlevel_flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);

        if cfg!(target_arch = "wasm32") {
            if is_gl_backend {
                log::info!("WebGPU not available, falling back to WebGL.");
            } else {
                log::info!("Using WebGPU backend.");
            }
        }

        // Set DUCK_FORCE_WEBGL2_LIMITS to cap any backend to the WebGL2
        // floor, for testing behavior on minimum supported hardware.
        let required_limits = if std::env::var("DUCK_FORCE_WEBGL2_LIMITS").is_ok() {
            log::info!("DUCK_FORCE_WEBGL2_LIMITS set: capping to downlevel WebGL2 limits");
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            adapter.limits()
        };
        log::info!(
            "Requested max_texture_dimension_2d: {}",
            required_limits.max_texture_dimension_2d,
        );

        // Depth formats only guarantee up to 4x MSAA in the WebGPU baseline;
        // higher sample counts (e.g. 8x) require this adapter-specific feature
        // to be enabled at device creation. Request it when available so the
        // MSAA probe below can pick counts above 4.
        let required_features = if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits,
                label: None,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // MSAA must be supported by both the color target and the renderer's
        // depth target, so probe both formats and pick the highest count both
        // allow.
        let sample_count = if downlevel_flags.contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING) {
            use crate::renderer::render_core::GpuTexture;
            let color_flags = adapter.get_texture_format_features(surface_format).flags;
            let depth_flags = adapter.get_texture_format_features(GpuTexture::DEPTH_FORMAT).flags;
            [8, 4, 2, 1]
                .into_iter()
                .find(|&n| {
                    color_flags.sample_count_supported(n) && depth_flags.sample_count_supported(n)
                })
                .unwrap_or(1)
        } else {
            1
        };
        log::info!("Using {sample_count}x MSAA");

        Self {
            surface,
            config,
            gpu: Gpu::new(device, queue),
            sample_count,
            has_compute,
        }
    }

    /// A clone of the shared GPU handle, for building renderers on the same
    /// device/queue (e.g. an [`OffscreenViewer`] or an egui renderer).
    pub fn gpu(&self) -> Gpu {
        self.gpu.clone()
    }

    /// The surface color format (an sRGB format when available).
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The MSAA sample count supported for this surface format.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Whether the adapter supports compute shaders.
    pub fn has_compute(&self) -> bool {
        self.has_compute
    }

    /// Current surface size as (width, height).
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigure the surface to a new size. Ignores zero dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
    }

    /// Acquire the next surface texture to render into and present.
    pub fn acquire(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

/// A [`Viewer`] that owns a window/canvas surface and presents to it.
pub struct SurfacedViewer<'a> {
    surface: WindowSurface<'a>,
    core: Viewer,
}

impl<'a> Deref for SurfacedViewer<'a> {
    type Target = Viewer;
    fn deref(&self) -> &Viewer {
        &self.core
    }
}

impl<'a> DerefMut for SurfacedViewer<'a> {
    fn deref_mut(&mut self) -> &mut Viewer {
        &mut self.core
    }
}

impl<'a> SurfacedViewer<'a> {
    /// Create a new viewer with the given surface target.
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let surface = WindowSurface::new(surface_target, width, height).await;
        Self::from_surface(surface, width, height)
    }

    /// Build a surfaced viewer around an already-configured [`WindowSurface`].
    pub fn from_surface(surface: WindowSurface<'a>, width: u32, height: u32) -> Self {
        let core = Viewer::from_gpu(
            surface.gpu(),
            surface.format(),
            width,
            height,
            surface.sample_count(),
            surface.has_compute(),
        );
        Self { surface, core }
    }

    /// Create a new viewer from a winit Window (native platforms).
    /// The viewer size is automatically determined from the window's inner size.
    #[cfg(feature = "winit-support")]
    pub async fn from_window(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        Self::new(window, size.width, size.height).await
    }

    /// Create a new viewer from an HTML canvas element (WebAssembly).
    /// The viewer size is automatically determined from the canvas dimensions.
    #[cfg(target_arch = "wasm32")]
    pub async fn from_canvas(canvas: web_sys::HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        Self::new(wgpu::SurfaceTarget::Canvas(canvas), width, height).await
    }

    /// Handle a single event, reconfiguring the surface on resize before
    /// delegating to the core viewer.
    pub fn handle_event(&mut self, event: &Event) {
        if let Event::Device(DeviceEvent::Resized(physical_size)) = event {
            let (w, h) = *physical_size;
            self.surface.resize(w, h);
        }
        self.core.handle_event(event);
    }

    /// Render the scene using the default rendering path.
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        let (output, _view, encoder) = self.render_scene()?;
        self.present(encoder, output);
        Ok(())
    }

    /// Prepare and render the 3D scene, returning the surface output, view, and
    /// command encoder for further rendering (overlays, post-processing, etc.).
    ///
    /// Call [`present()`](Self::present) when done to submit commands and display the frame.
    pub fn render_scene(&mut self) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView, wgpu::CommandEncoder), anyhow::Error> {
        self.core.prepare_scene()?;

        let output = self.surface.acquire()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.core.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") },
        );

        self.core.render_scene_to_view(&view, &mut encoder)?;

        Ok((output, view, encoder))
    }

    /// Submit the command encoder and present the surface texture.
    pub fn present(&self, encoder: wgpu::CommandEncoder, output: wgpu::SurfaceTexture) {
        self.core.queue().submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

/// A [`Viewer`] that renders into an owned offscreen texture instead of a
/// surface. The texture is sampleable (`TEXTURE_BINDING`), so consumers can
/// display it inside a UI panel (e.g. `egui::Image` via
/// `egui_wgpu::Renderer::register_native_texture`) or read it back to an image.
pub struct OffscreenViewer {
    color_format: wgpu::TextureFormat,
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    core: Viewer,
}

impl Deref for OffscreenViewer {
    type Target = Viewer;
    fn deref(&self) -> &Viewer {
        &self.core
    }
}

impl DerefMut for OffscreenViewer {
    fn deref_mut(&mut self) -> &mut Viewer {
        &mut self.core
    }
}

impl OffscreenViewer {
    /// Build an offscreen viewer around an existing GPU context.
    ///
    /// Use this to share the device/queue with the rest of an application (e.g.
    /// the device that owns the window surface and the egui renderer) so the
    /// rendered texture can be sampled by that same device. The renderer's
    /// internal MSAA targets use `sample_count`; the owned color texture is the
    /// single-sampled resolve target.
    pub fn from_gpu(
        gpu: Gpu,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        let (color_texture, color_view) =
            Self::create_color(&gpu.device, color_format, width, height);
        let core = Viewer::from_gpu(gpu, color_format, width, height, sample_count, has_compute);
        Self {
            color_format,
            color_texture,
            color_view,
            core,
        }
    }

    /// Create an offscreen viewer with its own headless GPU context.
    ///
    /// Convenience for thumbnails / server-side rendering where no surface or
    /// external device is involved. Uses `Rgba8UnormSrgb` and no MSAA.
    pub async fn headless(width: u32, height: u32) -> anyhow::Result<Self> {
        let (gpu, caps) = Gpu::headless().await?;
        Ok(Self::from_gpu(
            gpu,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            width,
            height,
            1,
            caps.has_compute,
        ))
    }

    fn create_color(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Color Target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Resize the offscreen target, recreating the color texture.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (texture, view) =
            Self::create_color(self.core.device(), self.color_format, width, height);
        self.color_texture = texture;
        self.color_view = view;
        self.core.resize(width, height);
    }

    /// Render the scene into the owned offscreen texture and submit. Does not
    /// present (there is no surface); read the result via [`texture_view`] or
    /// [`Viewer::render_to_image`].
    ///
    /// [`texture_view`]: Self::texture_view
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        self.core.prepare_scene()?;
        let mut encoder = self.core.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Offscreen Render Encoder") },
        );
        // Disjoint field borrows: `&self.color_view` alongside `&mut self.core`.
        self.core.render_scene_to_view(&self.color_view, &mut encoder)?;
        self.core.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// The offscreen color texture being rendered into.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.color_texture
    }

    /// A view of the offscreen color texture, for sampling (e.g. egui).
    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }
}
