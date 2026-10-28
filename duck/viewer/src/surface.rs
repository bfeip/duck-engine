use std::ops::{Deref, DerefMut};

use crate::{
    Viewer,
    compositor::{SrgbTarget, VIEW_FORMAT, srgb_view_format},
    event::{DeviceEvent, Event},
    renderer::{Gpu, GpuOptions, Renderer},
};

/// A configured window/canvas surface paired with the GPU context created for
/// it.
/// 
/// [`SurfacedViewer`] composes this with a viewer core on top of it;
/// applications that present through their own compositor (e.g. egui) can
/// hold a `WindowSurface` directly and drive an offscreen
/// viewer with `gpu()`.
pub struct WindowSurface<'a> {
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    sample_count: u32,
    /// Whether the surface texture allows its sRGB variant as a view format.
    can_reinterpret: bool,
}

impl<'a> WindowSurface<'a> {
    /// Create and configure a surface for the given target, bootstrapping the
    /// wgpu instance, adapter, and device/queue.
    pub async fn new<T>(surface_target: T, width: u32, height: u32, options: GpuOptions) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let instance = options.create_instance().await;

        let surface = instance.create_surface(surface_target).unwrap_or_else(|e| {
            panic!("No surface on backends {:?}: {e}", options.backends)
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "No GPU adapter for this surface among backends {:?}: {e}",
                    options.backends
                )
            });

        let info = adapter.get_info();
        let is_gl_backend = info.backend == wgpu::Backend::Gl;

        log::info!("Using {} adapter: {}", info.backend, info.name);

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

        // Without an sRGB surface format (WebGPU canvases offer none), view
        // the surface at its sRGB variant where the backend allows it.
        let can_reinterpret = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS);
        let view_formats = srgb_view_format(surface_format, can_reinterpret)
            .filter(|&format| format != surface_format)
            .into_iter()
            .collect();

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
            view_formats,
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // The renderer picks the count: it owns the attachments that have to
        // support it.
        let sample_count = Renderer::preferred_sample_count(&adapter, VIEW_FORMAT);

        if sample_count > 1 {
            log::info!("Using {sample_count}x MSAA");
        }
        else {
            log::warn!("MSAA unavailable");
        }

        Self {
            surface,
            config,
            gpu: Gpu::new(&adapter, device, queue),
            sample_count,
            can_reinterpret,
        }
    }

    /// Create a surface on the canvas named by a CSS selector (emscripten).
    #[cfg(all(emscripten, feature = "emscripten-support"))]
    pub async fn from_canvas_selector(
        selector: &str,
        width: u32,
        height: u32,
        options: GpuOptions,
    ) -> Self {
        let target = crate::emscripten_support::CanvasSelector::new(selector);
        Self::new(target, width, height, options).await
    }

    /// A clone of the shared GPU handle, for building renderers on the same
    /// device/queue (e.g. an [`OffscreenViewer`](crate::OffscreenViewer) or
    /// an egui renderer).
    pub fn gpu(&self) -> Gpu {
        self.gpu.clone()
    }

    /// The surface color format (an sRGB format when available).
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The MSAA sample count supported for rendering views on this device.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
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
    target: SrgbTarget,
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
    pub async fn new<T>(surface_target: T, width: u32, height: u32, options: GpuOptions) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let surface = WindowSurface::new(surface_target, width, height, options).await;
        Self::from_surface(surface, width, height)
    }

    /// Build a surfaced viewer around an already-configured [`WindowSurface`].
    pub fn from_surface(surface: WindowSurface<'a>, width: u32, height: u32) -> Self {
        let target = SrgbTarget::new(
            &surface.gpu.device,
            surface.format(),
            surface.can_reinterpret,
            width,
            height,
        );
        let core = Viewer::from_gpu(
            surface.gpu(),
            target.composite_format(),
            width,
            height,
            surface.sample_count(),
        );
        Self { surface, target, core }
    }

    /// Create a new viewer from a winit Window (native platforms).
    /// The viewer size is automatically determined from the window's inner size.
    #[cfg(feature = "winit-support")]
    pub async fn from_window(
        window: std::sync::Arc<winit::window::Window>,
        options: GpuOptions,
    ) -> Self {
        let size = window.inner_size();
        Self::new(window, size.width, size.height, options).await
    }

    /// Create a new viewer from an HTML canvas element (WebAssembly).
    /// The viewer size is automatically determined from the canvas dimensions.
    #[cfg(web)]
    pub async fn from_canvas(canvas: web_sys::HtmlCanvasElement, options: GpuOptions) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        Self::new(wgpu::SurfaceTarget::Canvas(canvas), width, height, options).await
    }

    /// Create a new viewer on the canvas named by a CSS selector (emscripten).
    #[cfg(all(emscripten, feature = "emscripten-support"))]
    pub async fn from_canvas_selector(
        selector: &str,
        width: u32,
        height: u32,
        options: GpuOptions,
    ) -> Self {
        let surface = WindowSurface::from_canvas_selector(selector, width, height, options).await;
        Self::from_surface(surface, width, height)
    }

    /// Handle a single event, reconfiguring the surface on resize before
    /// delegating to the core viewer.
    pub fn handle_event(&mut self, event: &Event) {
        if let Event::Device(DeviceEvent::Resized(physical_size)) = event {
            let (w, h) = *physical_size;
            self.surface.resize(w, h);
            self.target.resize(&self.surface.gpu.device, w, h);
        }
        self.core.handle_event(event);
    }

    /// Render all views using the default rendering path.
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        let (output, _view, encoder) = self.render_scene()?;
        self.present(encoder, output);
        Ok(())
    }

    /// Render all views into the surface, returning the surface output, a view
    /// of it at the surface format, and the command encoder for further
    /// rendering (overlays, post-processing, etc.).
    ///
    /// Call [`present()`](Self::present) when done to submit commands and display the frame.
    pub fn render_scene(&mut self) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView, wgpu::CommandEncoder), anyhow::Error> {
        let output = self.surface.acquire()?;
        let mut encoder = self.core.context().device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") },
        );

        self.target.render(&mut self.core, &output.texture, &mut encoder)?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((output, view, encoder))
    }

    /// Submit the command encoder and present the surface texture.
    pub fn present(&self, encoder: wgpu::CommandEncoder, output: wgpu::SurfaceTexture) {
        self.core.context().queue().submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
