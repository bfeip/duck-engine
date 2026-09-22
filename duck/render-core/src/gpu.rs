/// The GPU handle pair: device + queue.
///
/// Owned by whoever drives rendering; both `wgpu::Device` and `wgpu::Queue`
/// are internally reference-counted, so `Gpu` is cheaply cloneable.
#[derive(Clone)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// What the adapter behind a [`Gpu`] can do, for the decisions that have to be
/// made before any rendering: which attachments to allocate, and which passes
/// can run at all.
#[derive(Clone, Copy, Debug)]
pub struct GpuCapabilities {
    pub has_compute: bool,
    /// Whether a texture can carry a second view format (such as an sRGB target
    /// sampled as raw bytes).
    pub has_view_formats: bool,
    pub backend: wgpu::Backend,
}

impl GpuCapabilities {
    #[must_use]
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let flags = adapter.get_downlevel_capabilities().flags;
        Self {
            has_compute: flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS),
            has_view_formats: flags.contains(wgpu::DownlevelFlags::VIEW_FORMATS),
            backend: adapter.get_info().backend,
        }
    }

    /// Whether a depth attachment can be read in a shader.
    ///
    /// False on GL for two independent reasons: naga's GLSL backend has no
    /// `textureLoad` for depth textures, and multisampled attachments are
    /// allocated as renderbuffers, which cannot be bound at all. Plain MSAA
    /// still works there — only reading depth back does not.
    #[must_use]
    pub const fn samples_depth_textures(&self) -> bool {
        !matches!(self.backend, wgpu::Backend::Gl)
    }
}

/// Knobs for bringing up the wgpu instance behind a [`Gpu`].
#[derive(Clone, Copy, Debug)]
pub struct GpuOptions {
    /// Backends an adapter may be chosen from.
    pub backends: wgpu::Backends,
}

/// The backends normally used on this platform.
fn default_backends() -> wgpu::Backends {
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu::Backends::PRIMARY
    }
    // Emscripten reaches WebGL2 through the GLES backend; there is no WebGPU
    // backend to detect.
    #[cfg(all(target_arch = "wasm32", target_os = "emscripten"))]
    {
        wgpu::Backends::GL
    }
    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    {
        wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
    }
}

impl Default for GpuOptions {
    /// The platform's backends, overridden by the `WGPU_BACKEND` environment
    /// variable when it is set.
    fn default() -> Self {
        Self { backends: default_backends().with_env() }
    }
}

impl GpuOptions {
    /// Restrict adapter selection to the given backends.
    #[must_use]
    pub const fn with_backends(mut self, backends: wgpu::Backends) -> Self {
        self.backends = backends;
        self
    }

    /// Create the wgpu instance these options describe.
    #[allow(clippy::unused_async)]
    pub async fn create_instance(&self) -> wgpu::Instance {
        let descriptor = wgpu::InstanceDescriptor {
            backends: self.backends,
            ..Default::default()
        };

        #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
        {
            wgpu::util::new_instance_with_webgpu_detection(&descriptor).await
        }
        #[cfg(not(all(target_arch = "wasm32", not(target_os = "emscripten"))))]
        {
            wgpu::Instance::new(&descriptor)
        }
    }
}

impl Gpu {
    /// Wrap pre-created device and queue (e.g. created alongside a surface).
    #[must_use] 
    pub const fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
    }

    /// Create a GPU without a surface, for offscreen/headless rendering.
    ///
    /// Creates its own wgpu instance and adapter. Useful for generating still
    /// images, thumbnails, or server-side rendering.
    /// 
    /// # Errors
    /// 
    /// Will return `Err` if initialization of WGPU fails.
    pub async fn headless(options: GpuOptions) -> anyhow::Result<(Self, GpuCapabilities)> {
        let instance = options.create_instance().await;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "No GPU adapter for headless rendering among backends {:?}: {e}",
                    options.backends
                )
            })?;

        let info = adapter.get_info();
        log::info!("Using {} adapter: {}", info.backend, info.name);

        let capabilities = GpuCapabilities::from_adapter(&adapter);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // What the adapter offers, not the WebGPU baseline: a GL
                // adapter can sit below `Limits::default()` and would refuse it.
                required_limits: adapter.limits(),
                label: Some("Headless Renderer"),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await?;

        Ok((Self { device, queue }, capabilities))
    }
}
