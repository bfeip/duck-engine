//! Trait-based importer system for scene file formats.
//!
//! Implement [`Importer`] to add support for a new file format. The built-in
//! importers ([`GltfImporter`], [`UsdImporter`], [`AssimpImporter`])
//! cover the standard formats; custom importers can be passed to
//! [`load_sync_with`](crate::load_sync_with) / [`load_async_with`](crate::load_async_with).

use std::path::Path;

use crate::{
    DetectedFormat, LoadError, LoadOptions, ProgressReporter, ProgressState, SceneLoadResult,
};

// ============================================================================
// Importer Trait
// ============================================================================

/// Trait for scene file format importers.
///
/// Implement this trait to add support for loading a new file format.
/// The importer is responsible for format detection, progress reporting,
/// and converting file data into a [`SceneLoadResult`].
///
/// # Format Detection
///
/// When multiple importers are registered, detection proceeds in list order:
/// 1. [`detect_from_bytes`](Importer::detect_from_bytes) is tried on each importer; first `true` wins.
/// 2. If no byte-level match, [`detect_from_extension`](Importer::detect_from_extension) is tried; first `true` wins.
///
/// Use [`default_importers`] to get the built-in set, or supply your own list
/// to [`load_sync_with`](crate::load_sync_with) / [`load_async_with`](crate::load_async_with).
pub trait Importer: Send + Sync {
    /// Human-readable format name (e.g., `"glTF"`, `"USD"`).
    fn name(&self) -> &str;

    /// Try to detect this format from the first bytes of a file.
    ///
    /// When multiple importers match, the first one in the importer list wins.
    fn detect_from_bytes(&self, bytes: &[u8]) -> bool;

    /// Check if this importer handles the given file extension (without dot, lowercase).
    fn detect_from_extension(&self, ext: &str) -> bool;

    /// Load a scene from bytes.
    ///
    /// `path_hint` is provided when the source was a file path, allowing the
    /// importer to resolve external resources (textures, .bin files, etc.)
    /// relative to it.
    ///
    /// Call [`ProgressReporter::update`] at meaningful checkpoints. At minimum,
    /// update once when starting work and once on completion.
    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        options: &LoadOptions,
        progress: &dyn ProgressReporter,
    ) -> Result<SceneLoadResult, LoadError>;
}

// ============================================================================
// Built-in Importers
// ============================================================================

/// Importer for glTF / GLB files.
pub struct GltfImporter;

impl Importer for GltfImporter {
    fn name(&self) -> &str {
        "glTF"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        if bytes.len() < 4 {
            return false;
        }
        // GLB binary magic
        if bytes.starts_with(b"glTF") {
            return true;
        }
        // glTF JSON: starts with '{' after optional whitespace
        let first_non_ws = bytes.iter().position(|&b| !b.is_ascii_whitespace());
        matches!(first_non_ws, Some(pos) if bytes[pos] == b'{')
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("glb") || ext.eq_ignore_ascii_case("gltf")
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        options: &LoadOptions,
        progress: &dyn ProgressReporter,
    ) -> Result<SceneLoadResult, LoadError> {
        use crate::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf, parse_gltf_from_path};
        use duck_engine_scene::Scene;

        progress.update(ProgressState {
            description: "Parsing glTF".into(),
            progress: Some(0.1),
            stage: None,
        });
        let parsed = if let Some(path) = path_hint {
            parse_gltf_from_path(path)
        } else {
            parse_gltf(bytes)
        }
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

        progress.update(ProgressState {
            description: "Loading assets".into(),
            progress: Some(0.2),
            stage: None,
        });
        let mut scene = Scene::new();
        let (_material_map, mesh_map) = load_gltf_assets(&parsed, &mut scene)
            .map_err(|e| LoadError::Gltf(e.to_string()))?;

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
}

/// Importer for USD files (USDC, USDA, USDZ).
#[cfg(feature = "usd")]
pub struct UsdImporter;

#[cfg(feature = "usd")]
impl Importer for UsdImporter {
    fn name(&self) -> &str {
        "USD"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        // USDC binary starts with "PXR-USDC"
        if bytes.len() >= 8 && bytes.starts_with(b"PXR-USDC") {
            return true;
        }
        // USDZ is a ZIP archive starting with "PK"
        bytes.len() >= 2 && bytes[0] == b'P' && bytes[1] == b'K'
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        crate::usd::is_usd_extension(ext)
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &dyn ProgressReporter,
    ) -> Result<SceneLoadResult, LoadError> {
        progress.update(ProgressState {
            description: "Parsing USD".into(),
            progress: None,
            stage: None,
        });

        let result = if let Some(path) = path_hint {
            crate::usd::load_usd_scene_from_path(path)
        } else {
            crate::usd::load_usd_scene_from_bytes(bytes, "scene.usd")
        };

        let usd_result = result.map_err(|e| LoadError::Usd(e.to_string()))?;

        progress.update(ProgressState {
            description: "Complete".into(),
            progress: Some(1.0),
            stage: None,
        });
        Ok(SceneLoadResult {
            scene: usd_result.scene,
            camera: usd_result.camera,
            format: DetectedFormat::Usd,
        })
    }
}

/// Importer for Assimp-supported formats (OBJ, FBX, DAE, etc.).
#[cfg(feature = "assimp")]
pub struct AssimpImporter;

#[cfg(feature = "assimp")]
impl Importer for AssimpImporter {
    fn name(&self) -> &str {
        "Assimp"
    }

    fn detect_from_bytes(&self, _bytes: &[u8]) -> bool {
        // Assimp formats have no single magic byte sequence we can check
        false
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        crate::assimp::is_assimp_extension(ext)
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &dyn ProgressReporter,
    ) -> Result<SceneLoadResult, LoadError> {
        progress.update(ProgressState {
            description: "Importing model".into(),
            progress: None,
            stage: None,
        });

        let result = if let Some(path) = path_hint {
            crate::assimp::load_assimp_scene_from_path(path)
        } else {
            crate::assimp::load_assimp_scene_from_bytes(bytes, "")
        };

        let assimp_result = result.map_err(|e| LoadError::Assimp(e.to_string()))?;

        progress.update(ProgressState {
            description: "Complete".into(),
            progress: Some(1.0),
            stage: None,
        });
        Ok(SceneLoadResult {
            scene: assimp_result.scene,
            camera: assimp_result.camera,
            format: DetectedFormat::Assimp,
        })
    }
}

// ============================================================================
// Importer Registry
// ============================================================================

/// Importer for STEP and IGES CAD files.
///
/// Uses OpenCASCADE via the `duck-engine-cad` crate for tessellation.
#[cfg(feature = "cad")]
pub struct CadImporter;

#[cfg(feature = "cad")]
impl Importer for CadImporter {
    fn name(&self) -> &str {
        "CAD"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        // STEP files begin with the ISO 10303-21 exchange format header.
        // IGES has no reliable magic bytes; it falls back to extension detection.
        bytes.starts_with(b"ISO-10303-21")
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        crate::cad::is_cad_extension(ext)
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &dyn ProgressReporter,
    ) -> Result<SceneLoadResult, LoadError> {
        let is_step = bytes.starts_with(b"ISO-10303-21")
            || path_hint
                .and_then(|p| p.extension()?.to_str())
                .map(crate::cad::is_step_extension)
                .unwrap_or(false);

        let format_name = if is_step { "STEP" } else { "IGES" };
        progress.update(ProgressState {
            description: format!("Tessellating {format_name} model"),
            progress: None,
            stage: None,
        });

        let mut scene = duck_engine_scene::Scene::new();
        let options = crate::cad::CadImportOptions::default();

        let result = if let Some(path) = path_hint {
            if is_step {
                crate::cad::load_step(path, &mut scene, &options)
            } else {
                crate::cad::load_iges(path, &mut scene, &options)
            }
        } else {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| LoadError::Cad("CAD file is not valid UTF-8 text".into()))?;
            if is_step {
                crate::cad::load_step_from_str(text, &mut scene, &options)
            } else {
                crate::cad::load_iges_from_str(text, &mut scene, &options)
            }
        };
        result.map_err(|e| LoadError::Cad(e.to_string()))?; // CadImportResult is intentionally discarded here

        progress.update(ProgressState {
            description: "Complete".into(),
            progress: Some(1.0),
            stage: None,
        });

        let format = if is_step {
            DetectedFormat::Step
        } else {
            DetectedFormat::Iges
        };

        Ok(SceneLoadResult {
            scene,
            camera: None,
            format,
        })
    }
}

/// Returns the default set of built-in importers.
///
/// The order determines detection priority (first match wins):
/// glTF, CAD (if enabled), USD (if enabled), Assimp (if enabled).
pub fn default_importers() -> Vec<Box<dyn Importer>> {
    #[allow(unused_mut)]
    let mut importers: Vec<Box<dyn Importer>> = vec![Box::new(GltfImporter)];
    #[cfg(feature = "cad")]
    importers.push(Box::new(CadImporter));
    #[cfg(feature = "usd")]
    importers.push(Box::new(UsdImporter));
    #[cfg(feature = "assimp")]
    importers.push(Box::new(AssimpImporter));
    importers
}

/// Find the first importer that recognizes the given data.
///
/// Tries magic-byte detection first (in list order), then falls back to
/// file-extension detection if a path hint is available.
pub fn detect_importer<'a>(
    bytes: &[u8],
    path_hint: Option<&Path>,
    importers: &'a [Box<dyn Importer>],
) -> Result<&'a dyn Importer, LoadError> {
    // First pass: magic bytes
    for imp in importers {
        if imp.detect_from_bytes(bytes) {
            return Ok(imp.as_ref());
        }
    }

    // Second pass: file extension fallback
    if let Some(path) = path_hint
        && let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            for imp in importers {
                if imp.detect_from_extension(ext) {
                    return Ok(imp.as_ref());
                }
            }
        }

    Err(LoadError::UnknownFormat)
}
