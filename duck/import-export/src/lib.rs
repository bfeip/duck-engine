//! Scene import library.
//!
//! Provides format-agnostic loading of scenes with progress reporting.
//! Supports glTF (.glb/.gltf), USD (.usdc/.usda/.usdz), CAD (.step/.iges), and
//! optionally assimp-based formats.
//!
//! # Submodules
//!
//! - [`gltf`] — glTF loading
//! - [`usd`] — USD loading (USDC, USDA, USDZ)
//! - [`assimp`] — Assimp-based loading (feature-gated)
//! # Native
//! Loading runs on a background thread via [`std::thread::spawn`]. The caller
//! polls [`LoadHandle`] each frame for completion and progress.
//!
//! # WASM
//! Loading uses chunked yielding via `wasm_bindgen_futures::spawn_local`,
//! yielding to the browser event loop between loading phases so the page
//! stays responsive.
//!
//! # Examples
//!
//! ```no_run
//! use duck_engine_import_export::{load_async, LoadOptions, SceneSource};
//!
//! let handle = load_async(SceneSource::Path("model.glb".into()), LoadOptions::default());
//!
//! // In your render loop:
//! // if let Some(result) = handle.try_recv() { ... }
//! ```

#[cfg(feature = "assimp")]
pub mod assimp;
#[cfg(feature = "cad")]
pub mod cad;
pub mod gltf;
pub mod importer;
#[cfg(feature = "usd")]
pub mod usd;

pub use importer::{default_importers, detect_importer, Importer};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use thiserror::Error;

use duck_engine_scene::PositionedCamera;
use duck_engine_scene::SceneData;

// ============================================================================
// Type Aliases
// ============================================================================

/// The result type produced by a completed load operation.
type LoadResult = Result<SceneLoadResult, LoadError>;

// ============================================================================
// Types
// ============================================================================

/// Where scene data comes from.
pub enum SceneSource {
    /// Load from a filesystem path. Format is auto-detected from magic bytes,
    /// falling back to file extension.
    Path(PathBuf),
    /// Load from bytes already in memory. Format is auto-detected from magic bytes.
    Bytes(Vec<u8>),
}

/// Options that control scene loading behavior.
pub struct LoadOptions {
    /// Aspect ratio for cameras embedded in glTF files. Ignored for other formats.
    /// Default: 16.0 / 9.0.
    pub aspect: f32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            aspect: 16.0 / 9.0,
        }
    }
}

/// The result of a successful scene load.
pub struct SceneLoadResult {
    /// The loaded scene.
    pub scene: SceneData,
    /// Camera extracted from the file, if present (glTF only).
    pub camera: Option<PositionedCamera>,
    /// Which format was detected and loaded.
    pub format: DetectedFormat,
}

/// The file format that was detected and used for loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectedFormat {
    Gltf,
    #[cfg(feature = "assimp")]
    Assimp,
    #[cfg(feature = "cad")]
    Step,
    #[cfg(feature = "cad")]
    Iges,
    #[cfg(feature = "usd")]
    Usd,
    /// A custom format provided by a user-defined [`Importer`].
    Other(String),
}

/// Errors that can occur during scene loading.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("glTF error: {0}")]
    Gltf(String),

    #[cfg(feature = "assimp")]
    #[error("Assimp error: {0}")]
    Assimp(String),

    #[cfg(feature = "cad")]
    #[error("CAD error: {0}")]
    Cad(String),

    #[cfg(feature = "usd")]
    #[error("USD error: {0}")]
    Usd(String),

    #[error("Format '{0}' is not supported on this platform")]
    UnsupportedPlatform(String),

    #[error("Unknown file format")]
    UnknownFormat,
}

// ============================================================================
// Progress Reporting
// ============================================================================

/// A snapshot of the current progress state.
///
/// Produced by [`SharedProgress::snapshot`] for reading on the caller's thread,
/// or constructed by importers/exporters and pushed via [`ProgressReporter::update`].
#[derive(Clone, Debug, Default)]
pub struct ProgressState {
    /// Human-readable description of the current operation.
    /// e.g. `"Parsing glTF"`, `"Tessellating STEP model"`, `"Loading texture: Skybox"`
    pub description: String,
    /// Overall progress in the range 0.0–1.0.
    /// `None` means indeterminate — show a spinner rather than a progress bar.
    pub progress: Option<f32>,
    /// Item-level stage progress: `(completed, total)`.
    /// `None` when the operation has no meaningful item breakdown.
    pub stage: Option<(u32, u32)>,
}

impl ProgressState {
    /// Initial state before any work has begun.
    pub fn starting() -> Self {
        Self {
            description: "Starting".into(),
            progress: Some(0.0),
            stage: None,
        }
    }
}

/// Push interface for importers and exporters to report progress.
///
/// Call [`update`](Self::update) at meaningful checkpoints — at minimum once
/// when starting and once when complete. The trait is object-safe and
/// `Send + Sync` so it can be called from background threads.
pub trait ProgressReporter: Send + Sync {
    fn update(&self, state: ProgressState);
}

/// Thread-safe shared progress cell. Holds the most recent [`ProgressState`]
/// behind a `Mutex`. `Clone`-able so async tasks and their callers can both
/// hold a reference to the same cell.
///
/// Implements [`ProgressReporter`] (push) and exposes [`snapshot`](Self::snapshot) (pull).
#[derive(Clone, Default)]
pub struct SharedProgress {
    state: Arc<Mutex<ProgressState>>,
}

impl SharedProgress {
    /// Create a new `SharedProgress` in the default (empty) state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the current state for reading. Takes the lock once, clones, and returns.
    pub fn snapshot(&self) -> ProgressState {
        self.state.lock().unwrap().clone()
    }
}

impl ProgressReporter for SharedProgress {
    fn update(&self, state: ProgressState) {
        *self.state.lock().unwrap() = state;
    }
}

/// No-op progress reporter. Use for sync operations where progress tracking is not needed.
pub struct NullProgress;

impl ProgressReporter for NullProgress {
    fn update(&self, _: ProgressState) {}
}

/// Handle to an async operation in progress.
///
/// Poll each frame with [`try_get`](AsyncHandle::try_get) to check for
/// completion, and read [`progress`](AsyncHandle::progress) to display a
/// progress bar.
pub struct AsyncHandle<T> {
    progress: SharedProgress,
    done: Arc<AtomicBool>,
    #[cfg(not(target_arch = "wasm32"))]
    receiver: std::sync::mpsc::Receiver<T>,
    #[cfg(target_arch = "wasm32")]
    result: std::rc::Rc<std::cell::RefCell<Option<T>>>,
}

impl<T> AsyncHandle<T> {
    /// Returns `Some(result)` if the operation has completed, `None` if still in progress.
    ///
    /// Consumes the result on first successful call. Subsequent calls return `None`.
    pub fn try_get(&self) -> Option<T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.receiver.try_recv().ok()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.result.borrow_mut().take()
        }
    }

    /// Returns the shared progress state.
    pub fn progress(&self) -> &SharedProgress {
        &self.progress
    }

    /// Returns true if the operation has completed (success or failure).
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

/// Handle to a loading operation in progress.
pub type LoadHandle = AsyncHandle<LoadResult>;

// ============================================================================
// Sync Loading (core logic)
// ============================================================================

/// Load a scene synchronously. Useful for CLI tools and tests.
///
/// Uses the [default importers](default_importers) for format detection.
pub fn load_sync(source: SceneSource, options: LoadOptions) -> LoadResult {
    let importers = default_importers();
    load_sync_with(source, options, &importers)
}

/// Load a scene synchronously using a custom set of importers.
///
/// The importer list order determines detection priority (first match wins).
pub fn load_sync_with(
    source: SceneSource,
    options: LoadOptions,
    importers: &[Box<dyn Importer>],
) -> LoadResult {
    load_sync_with_progress(source, &options, &NullProgress, importers)
}

fn load_sync_with_progress(
    source: SceneSource,
    options: &LoadOptions,
    progress: &dyn ProgressReporter,
    importers: &[Box<dyn Importer>],
) -> LoadResult {
    progress.update(ProgressState::starting());

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    let importer = detect_importer(&bytes, path_hint.as_deref(), importers)?;
    let result = importer.load(&bytes, path_hint.as_deref(), options, progress)?;

    Ok(result)
}

// ============================================================================
// Async Helpers
// ============================================================================

/// Spawn a sync closure on a background thread, returning a handle to poll for the result.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_async<T, E>(
    progress: SharedProgress,
    f: impl FnOnce(&SharedProgress) -> Result<T, E> + Send + 'static,
) -> AsyncHandle<Result<T, E>>
where
    T: Send + 'static,
    E: Send + 'static,
{
    let progress_clone = progress.clone();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let result = f(&progress_clone);
        done_clone.store(true, Ordering::Release);
        let _ = tx.send(result);
    });

    AsyncHandle {
        progress,
        done,
        receiver: rx,
    }
}

/// Spawn a future as a WASM microtask, returning a handle to poll for the result.
#[cfg(target_arch = "wasm32")]
fn spawn_async_wasm<T, E>(
    progress: SharedProgress,
    fut: impl std::future::Future<Output = Result<T, E>> + 'static,
) -> AsyncHandle<Result<T, E>>
where
    T: 'static,
    E: 'static,
{
    use std::cell::RefCell;
    use std::rc::Rc;

    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let result_cell: Rc<RefCell<Option<Result<T, E>>>> = Rc::new(RefCell::new(None));
    let result_clone = Rc::clone(&result_cell);

    wasm_bindgen_futures::spawn_local(async move {
        let result = fut.await;
        done_clone.store(true, Ordering::Release);
        *result_clone.borrow_mut() = Some(result);
    });

    AsyncHandle {
        progress,
        done,
        result: result_cell,
    }
}

// ============================================================================
// Async Loading
// ============================================================================

/// Start loading a scene asynchronously.
///
/// Uses the [default importers](default_importers) for format detection.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between loading phases.
///
/// Returns a [`LoadHandle`] to poll for completion and progress.
pub fn load_async(source: SceneSource, options: LoadOptions) -> LoadHandle {
    load_async_with(source, options, default_importers())
}

/// Start loading a scene asynchronously using a custom set of importers.
///
/// The importer list order determines detection priority (first match wins).
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between built-in loading phases; custom importers run
/// synchronously within a microtask.
pub fn load_async_with(
    source: SceneSource,
    options: LoadOptions,
    importers: Vec<Box<dyn Importer>>,
) -> LoadHandle {
    #[cfg(not(target_arch = "wasm32"))]
    {
        spawn_async(SharedProgress::new(), move |progress| {
            load_sync_with_progress(source, &options, progress, &importers)
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let progress = SharedProgress::new();
        let p = progress.clone();
        spawn_async_wasm(progress, async move {
            load_chunked_wasm(source, &options, &p, &importers).await
        })
    }
}

/// Yield to the browser event loop. Allows the page to process events
/// (rendering, input) between expensive loading phases.
///
/// Uses `setTimeout(0)` to schedule a macrotask, ensuring the browser gets
/// a chance to render and run `requestAnimationFrame` between yield points.
/// A resolved `Promise` alone only creates a microtask, which runs before
/// the browser paints.
#[cfg(target_arch = "wasm32")]
async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let set_timeout =
            js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout"))
                .expect("setTimeout not found on global");
        let set_timeout: js_sys::Function = set_timeout.into();
        let _ = set_timeout.call2(
            &wasm_bindgen::JsValue::NULL,
            &resolve,
            &wasm_bindgen::JsValue::from(0),
        );
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// WASM chunked loading: runs loading phases with yields between them.
///
/// glTF uses optimized chunked loading with yield points between phases. All
/// other importers fall back to the synchronous `Importer::load` within a
/// single microtask.
#[cfg(target_arch = "wasm32")]
async fn load_chunked_wasm(
    source: SceneSource,
    options: &LoadOptions,
    progress: &SharedProgress,
    importers: &[Box<dyn Importer>],
) -> LoadResult {
    progress.update(ProgressState::starting());

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            // Note: std::fs::read doesn't work on WASM. Callers should use
            // SceneSource::Bytes. This arm exists for API uniformity.
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    yield_to_event_loop().await;

    let importer = detect_importer(&bytes, path_hint.as_deref(), importers)?;

    // Use the optimized chunked path for glTF
    match importer.name() {
        "glTF" => {
            if !bytes.starts_with(b"glTF") {
                // Non-GLB glTF files reference external resources via filesystem
                // paths, which are not accessible on WASM.
                return Err(LoadError::UnsupportedPlatform(
                    "non-GLB glTF (requires filesystem)".into(),
                ));
            }
            load_gltf_chunked(&bytes, options, progress).await
        }
        _ => {
            // Custom/other importers: run synchronously within this microtask
            importer.load(&bytes, path_hint.as_deref(), options, progress)
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn load_gltf_chunked(
    bytes: &[u8],
    options: &LoadOptions,
    progress: &SharedProgress,
) -> LoadResult {
    use self::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf};

    progress.update(ProgressState {
        description: "Parsing glTF".into(),
        progress: Some(0.1),
        stage: None,
    });
    let parsed = parse_gltf(bytes).map_err(|e| LoadError::Gltf(e.to_string()))?;
    yield_to_event_loop().await;

    progress.update(ProgressState {
        description: "Loading assets".into(),
        progress: Some(0.2),
        stage: None,
    });
    let mut scene = SceneData::new();
    let (_material_map, mesh_map) =
        load_gltf_assets(&parsed, &mut scene).map_err(|e| LoadError::Gltf(e.to_string()))?;
    yield_to_event_loop().await;

    progress.update(ProgressState {
        description: "Building scene".into(),
        progress: Some(0.7),
        stage: None,
    });
    let camera = build_gltf_scene(&parsed, &mut scene, &mesh_map, options.aspect)
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

    progress.update(ProgressState {
        description: "Complete".into(),
        progress: Some(1.0),
        stage: None,
    });
    Ok(SceneLoadResult {
        scene,
        camera,
        format: DetectedFormat::Gltf,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Minimal importer used to exercise the format-agnostic plumbing.
    struct TestImporter;

    impl Importer for TestImporter {
        fn name(&self) -> &str { "Test" }
        fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
            bytes.starts_with(b"TEST")
        }
        fn detect_from_extension(&self, ext: &str) -> bool {
            ext == "test"
        }
        fn load(
            &self,
            _bytes: &[u8],
            _path_hint: Option<&Path>,
            _options: &LoadOptions,
            progress: &dyn ProgressReporter,
        ) -> Result<SceneLoadResult, LoadError> {
            progress.update(ProgressState {
                description: "Testing".into(),
                progress: Some(0.5),
                stage: None,
            });
            progress.update(ProgressState {
                description: "Complete".into(),
                progress: Some(1.0),
                stage: None,
            });
            Ok(SceneLoadResult {
                scene: SceneData::new(),
                camera: None,
                format: DetectedFormat::Other("Test".into()),
            })
        }
    }

    #[test]
    fn test_detect_glb() {
        let importers = default_importers();
        let bytes = b"glTF\x02\x00\x00\x00rest";
        let imp = detect_importer(bytes, None, &importers).unwrap();
        assert_eq!(imp.name(), "glTF");
    }

    #[test]
    fn test_detect_gltf_json() {
        let importers = default_importers();
        let bytes = b"  { \"asset\": {} }";
        let imp = detect_importer(bytes, None, &importers).unwrap();
        assert_eq!(imp.name(), "glTF");
    }

    #[test]
    fn test_detect_unknown() {
        let importers = default_importers();
        let bytes = b"\x00\x00\x00\x00";
        assert!(detect_importer(bytes, None, &importers).is_err());
    }

    #[test]
    fn test_detect_from_extension() {
        let importers = default_importers();
        let unknown = b"\x00\x00\x00\x00";

        let imp = detect_importer(unknown, Some(std::path::Path::new("model.glb")), &importers).unwrap();
        assert_eq!(imp.name(), "glTF");

        let imp = detect_importer(unknown, Some(std::path::Path::new("model.gltf")), &importers).unwrap();
        assert_eq!(imp.name(), "glTF");

        #[cfg(feature = "assimp")]
        {
            let imp = detect_importer(unknown, Some(std::path::Path::new("model.obj")), &importers).unwrap();
            assert_eq!(imp.name(), "Assimp");
        }
        #[cfg(not(feature = "assimp"))]
        assert!(detect_importer(unknown, Some(std::path::Path::new("model.obj")), &importers).is_err());
    }

    #[test]
    fn test_custom_importer() {
        let importers: Vec<Box<dyn Importer>> = vec![Box::new(TestImporter)];

        // Detect from bytes
        let imp = detect_importer(b"TEST data", None, &importers).unwrap();
        assert_eq!(imp.name(), "Test");

        // Detect from extension
        let imp = detect_importer(b"\x00\x00\x00\x00", Some(Path::new("file.test")), &importers).unwrap();
        assert_eq!(imp.name(), "Test");

        // Load via load_sync_with
        let result = load_sync_with(
            SceneSource::Bytes(b"TEST data".to_vec()),
            LoadOptions::default(),
            &importers,
        ).unwrap();
        assert_eq!(result.format, DetectedFormat::Other("Test".into()));
    }

    #[test]
    fn test_shared_progress() {
        let progress = SharedProgress::new();

        let snap = progress.snapshot();
        assert_eq!(snap.description, "");
        assert_eq!(snap.progress, None);
        assert_eq!(snap.stage, None);

        progress.update(ProgressState {
            description: "Loading".into(),
            progress: Some(0.5),
            stage: Some((3, 10)),
        });
        let snap = progress.snapshot();
        assert_eq!(snap.description, "Loading");
        assert_eq!(snap.progress, Some(0.5));
        assert_eq!(snap.stage, Some((3, 10)));

        progress.update(ProgressState {
            description: "Complete".into(),
            progress: Some(1.0),
            stage: None,
        });
        let snap = progress.snapshot();
        assert_eq!(snap.progress, Some(1.0));
        assert_eq!(snap.stage, None);
    }

    #[test]
    fn test_null_progress_is_no_op() {
        NullProgress.update(ProgressState {
            description: "test".into(),
            progress: Some(0.5),
            stage: Some((1, 2)),
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_async_success() {
        let importers: Vec<Box<dyn Importer>> = vec![Box::new(TestImporter)];
        let handle = load_async_with(
            SceneSource::Bytes(b"TEST data".to_vec()),
            LoadOptions::default(),
            importers,
        );

        // Poll until done (in a real app this would be each frame)
        loop {
            if let Some(result) = handle.try_get() {
                let result = result.unwrap();
                assert_eq!(result.format, DetectedFormat::Other("Test".into()));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_async_unknown_format() {
        let bytes = vec![0u8; 100];
        let handle = load_async(SceneSource::Bytes(bytes), LoadOptions::default());

        loop {
            if let Some(result) = handle.try_get() {
                assert!(result.is_err());
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }
}
