//! Files to and from the Document: the modeler's own document files, STEP/IGES
//! parts, and the HDR images that light its scene. The platform
//! [`Host`](crate::platform::Host) decides where the files come from and go.
//!
//! A document file is a zip archive: a `document.json` manifest holding the
//! material library, the parts and the [`Session`], one binary B-Rep entry per
//! part, and the environment's HDR image if the scene has one.

use std::ffi::OsString;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use duck_engine_import_export::cad;
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::common::{Plane, WorldUnits};
use duck_engine_scene::{EnvironmentSource, PositionedCamera, Scene};
use duck_engine_viewer::renderer::ibl;
use opencascade::primitives::Shape;
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

use crate::document::{Document, DocumentContents, PartRecord, DOCUMENT_UNITS};
use crate::grid::GridConfig;
use crate::materials::MaterialRecord;
use crate::platform::FileFilter;
use crate::ui::Lighting;

/// The unit OCCT reads STEP and IGES files into, whatever unit a file declares,
/// and writes them in. Shapes convert between it and [`DOCUMENT_UNITS`] on the
/// way in and out.
const FILE_UNITS: WorldUnits = WorldUnits::MILLIMETER;

/// The files [`save_document`] writes and [`open_document`] reads.
pub const DOCUMENT_FILTER: FileFilter = FileFilter { name: "Modeler document", extensions: &["duck"] };

/// The files [`import_cad_file`] reads.
pub const IMPORT_FILTER: FileFilter =
    FileFilter { name: "CAD (STEP/IGES)", extensions: cad::CAD_EXTENSIONS };

/// The files [`export_cad_file`] writes, STEP first.
pub const EXPORT_FILTERS: [FileFilter; 2] = [
    FileFilter { name: "STEP", extensions: &["step", "stp"] },
    FileFilter { name: "IGES", extensions: &["iges", "igs"] },
];

/// The images [`load_environment`] reads.
pub const ENVIRONMENT_FILTER: FileFilter = FileFilter { name: "Radiance HDR", extensions: &["hdr"] };

/// The document format [`save_document`] writes. [`open_document`] reads it
/// and any earlier one.
const FORMAT_VERSION: u32 = 1;

const MANIFEST_ENTRY: &str = "document.json";
const ENVIRONMENT_ENTRY: &str = "environment.hdr";

/// What a document file restores besides the document: how it was viewed and
/// lit, and where it was being modelled.
#[derive(Serialize, Deserialize)]
pub struct Session {
    pub camera: PositionedCamera,
    pub lighting: Lighting,
    /// The construction plane the user set.
    pub construction_plane: Plane,
    pub grid: GridConfig,
}

/// A document file's `document.json`.
#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    materials: Vec<MaterialRecord>,
    parts: Vec<PartEntry>,
    session: Session,
}

#[derive(Serialize, Deserialize)]
struct PartEntry {
    #[serde(flatten)]
    part: PartRecord,
    /// The entry holding the part's B-Rep.
    brep: String,
}

/// Writes `document` and `session` to `path` as a document file. A file
/// already there is replaced only once the new one is complete.
pub fn save_document(path: &Path, document: &Document, session: Session) -> Result<()> {
    let bytes = document_file(document, session)?;
    let mut partial = OsString::from(path.as_os_str());
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    std::fs::write(&partial, bytes).with_context(|| format!("Failed to write {}", partial.display()))?;
    std::fs::rename(&partial, path).with_context(|| format!("Failed to replace {}", path.display()))
}

/// `document` and `session` as the bytes of a document file.
fn document_file(document: &Document, session: Session) -> Result<Vec<u8>> {
    let DocumentContents { materials, parts } = document.contents();
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default();

    let mut entries = Vec::with_capacity(parts.len());
    for (index, (part, shape)) in parts.into_iter().enumerate() {
        let bytes = shape
            .write_brep_bin_to_bytes(false)
            .with_context(|| format!("Failed to write {}", part.name))?;
        let brep = format!("parts/{index}.brep");
        zip.start_file(brep.as_str(), options)?;
        zip.write_all(&bytes)?;
        entries.push(PartEntry { part, brep });
    }

    {
        let scene = document.scene().lock();
        if let Some(map) = scene.active_environment_map().and_then(|id| scene.get_environment_map(id)) {
            zip.start_file(ENVIRONMENT_ENTRY, options)?;
            match map.source() {
                EnvironmentSource::EquirectangularHdr(bytes) => zip.write_all(bytes)?,
                EnvironmentSource::EquirectangularPath(path) => {
                    let bytes = std::fs::read(path)
                        .with_context(|| format!("Failed to read {}", path.display()))?;
                    zip.write_all(&bytes)?;
                }
            }
        }
    }

    let manifest = Manifest { version: FORMAT_VERSION, materials, parts: entries, session };
    zip.start_file(MANIFEST_ENTRY, options)?;
    serde_json::to_writer_pretty(&mut zip, &manifest)?;
    Ok(zip.finish()?.into_inner())
}

/// Reads the document file at `path`: its document, drawn in a scene of its
/// own, and the session to restore with it. Fails as a whole.
pub fn open_document(path: &Path) -> Result<(Document, Session)> {
    let bytes = std::fs::read(path)?;
    let mut archive = ZipArchive::new(Cursor::new(bytes)).context("Not a document file")?;
    let manifest: Manifest = serde_json::from_slice(&read_entry(&mut archive, MANIFEST_ENTRY)?)
        .context("Unreadable document manifest")?;
    anyhow::ensure!(
        manifest.version <= FORMAT_VERSION,
        "The document is format version {}, newer than this modeler reads ({FORMAT_VERSION})",
        manifest.version,
    );

    let parts = manifest
        .parts
        .into_iter()
        .map(|PartEntry { part, brep }| {
            let shape = read_entry(&mut archive, &brep)
                .and_then(|bytes| Ok(Shape::read_brep_bin_from_bytes(&bytes)?))
                .with_context(|| format!("Failed to read {}", part.name))?;
            Ok((part, shape))
        })
        .collect::<Result<_>>()?;
    let contents = DocumentContents { materials: manifest.materials, parts };
    let document = Document::from_contents(Scene::default(), contents)?;
    if archive.index_for_name(ENVIRONMENT_ENTRY).is_some() {
        let image = read_entry(&mut archive, ENVIRONMENT_ENTRY)?;
        set_environment(document.scene(), image).context("Failed to read the environment")?;
    }
    Ok((document, manifest.session))
}

/// The whole of `archive`'s entry `name`, its checksum verified.
fn read_entry(archive: &mut ZipArchive<Cursor<Vec<u8>>>, name: &str) -> Result<Vec<u8>> {
    let mut entry = archive.by_name(name).with_context(|| format!("The document has no {name}"))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Register each leaf part of the STEP/IGES file at `path` in `document`, as
/// one undo step.
pub fn import_cad_file(
    document: &Arc<Mutex<Document>>,
    path: &Path,
    options: &CadTessellationOptions,
) -> Result<()> {
    let parts = cad::load_cad_parts(path, FILE_UNITS.factor_to(DOCUMENT_UNITS))?;
    if parts.is_empty() {
        log::warn!("No parts found in {}", path.display());
        return Ok(());
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Imported").to_string();
    let mut document = document.lock().unwrap();
    // One undo step covers every part in the file.
    let mut document = document.undo_scope(format!("Import {stem}"));
    for (i, part) in parts.into_iter().enumerate() {
        let mut options = options.clone();
        if let Some(color) = part.color {
            options.face_material.set_base_color_factor(color);
            // A part that carries its own color keeps it whatever its geometry
            // class, so drop the sketch appearance for this part.
            options.free_face_material = None;
            options.free_line_material = None;
        }
        let name = part.name.unwrap_or_else(|| format!("{stem} {}", i + 1));
        document.add_part(name, part.shape, part.frame, &options)?;
    }
    Ok(())
}

/// Write all document parts to `path` as one compound, STEP or IGES by its
/// extension.
pub fn export_cad_file(document: &Arc<Mutex<Document>>, path: &Path) -> Result<()> {
    let document = document.lock().unwrap();
    let scale = DOCUMENT_UNITS.factor_to(FILE_UNITS);
    cad::save_cad_shapes(path, document.parts().map(|p| p.shape()), scale)?;
    log::info!("Exported {} part(s) to {}", document.parts().count(), path.display());
    Ok(())
}

/// Light `scene` with the HDR image at `path`, replacing its environment map if
/// it had one.
pub fn load_environment(scene: &Scene, path: &Path) -> Result<()> {
    // The bytes rather than the path: the browser's copy of a picked file is
    // gone once it has been received.
    set_environment(scene, std::fs::read(path)?)
}

/// Light `scene` with the HDR image `bytes`, replacing its environment map if
/// it had one.
fn set_environment(scene: &Scene, bytes: Vec<u8>) -> Result<()> {
    // Decoded once here so a bad image is reported, not left to fail every
    // render.
    ibl::load_hdr_from_bytes(&bytes)?;
    let mut scene = scene.lock();
    if let Some(previous) = scene.active_environment_map() {
        scene.remove_environment_map(previous);
    }
    let id = scene.add_environment_map_from_hdr_data(bytes);
    scene.set_active_environment_map(Some(id));
    Ok(())
}

#[cfg(test)]
mod tests {
    use duck_engine_scene::common::{Point3, RgbaColor, Transform, Vector3};
    use duck_engine_scene::resource::Visibility;
    use duck_engine_scene::{Light, PositionedLight, Projection};
    use duck_engine_viewer::{HeadlightMode, default_headlight_rig};
    use opencascade::primitives::Shape;

    use super::*;

    /// A one-pixel Radiance HDR image.
    const HDR_IMAGE: &[u8] = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\x80\x80\x80\x81";

    /// Write a one-pixel Radiance HDR image named `name` to the temp dir.
    fn hdr_file(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, HDR_IMAGE).unwrap();
        path
    }

    /// A file's parts land in the document as one undo step.
    #[test]
    fn import_registers_parts_as_one_undo_step() {
        let path = std::env::temp_dir().join("duck_modeler_io_import.step");
        cad::save_cad_shapes(&path, [&Shape::box_centered(2.0, 2.0, 2.0)], 1.0).unwrap();
        let document = Arc::new(Mutex::new(Document::new(Scene::default())));

        let result = import_cad_file(&document, &path, &CadTessellationOptions::default());
        let _ = std::fs::remove_file(&path);
        result.unwrap();

        let mut document = document.lock().unwrap();
        assert_eq!(document.parts().count(), 1);
        assert_eq!(document.undo_label(), Some("Import duck_modeler_io_import"));
        document.undo().unwrap();
        assert!(document.is_empty(), "one undo removes the whole import");
    }

    /// The document's lengths are written out as the file's millimetres, and
    /// read back as they were.
    #[test]
    fn export_writes_millimetres_and_import_reads_them_back() {
        let path = std::env::temp_dir().join("duck_modeler_io_round_trip.step");
        let options = CadTessellationOptions::default();
        let document = Arc::new(Mutex::new(Document::new(Scene::default())));
        // 100 mm, in document units.
        let side = 100.0 * FILE_UNITS.factor_to(DOCUMENT_UNITS);
        let cube = Shape::box_centered(side, side, side);
        document.lock().unwrap().add_part("Box", cube, Transform::IDENTITY, &options).unwrap();
        export_cad_file(&document, &path).unwrap();

        let in_file = cad::load_cad_parts(&path, 1.0);
        let imported = Arc::new(Mutex::new(Document::new(Scene::default())));
        let result = import_cad_file(&imported, &path, &options);
        let _ = std::fs::remove_file(&path);
        result.unwrap();

        let in_file = in_file.unwrap();
        let mm_volume = in_file[0].shape.volume();
        assert!((mm_volume - 1e6).abs() < 1e-6 * 1e6, "a 100 mm cube on file, got {mm_volume}");
        let imported = imported.lock().unwrap();
        let volume = imported.parts().next().unwrap().shape().volume();
        let expected = side.powi(3);
        assert!((volume - expected).abs() < 1e-6 * expected, "the cube read back, got {volume}");
    }

    #[test]
    fn loading_an_environment_replaces_the_last() {
        let path = hdr_file("duck_modeler_environment_replace.hdr");
        let scene = Scene::default();
        let first = load_environment(&scene, &path);
        let first_id = scene.active_environment_map();
        let second = load_environment(&scene, &path);
        let _ = std::fs::remove_file(&path);
        first.unwrap();
        second.unwrap();

        let scene = scene.lock();
        assert_eq!(scene.environment_map_count(), 1);
        assert!(scene.active_environment_map().is_some());
        assert_ne!(scene.active_environment_map(), first_id);
    }

    #[test]
    fn a_bad_environment_keeps_the_current_one() {
        let good = hdr_file("duck_modeler_environment_good.hdr");
        let bad = std::env::temp_dir().join("duck_modeler_environment_bad.hdr");
        std::fs::write(&bad, b"not an HDR image").unwrap();
        let scene = Scene::default();
        let loaded = load_environment(&scene, &good);
        let current = scene.active_environment_map();
        let failed = load_environment(&scene, &bad);
        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);

        loaded.unwrap();
        assert!(failed.is_err());
        let scene = scene.lock();
        assert_eq!(scene.environment_map_count(), 1);
        assert_eq!(scene.active_environment_map(), current);
    }

    /// A session unlike the modeler's defaults in every field it saves.
    fn session() -> Session {
        let light = Light::point(RgbaColor { r: 1.0, g: 0.5, b: 0.25, a: 1.0 }, 3.0);
        let placed = Transform { position: Point3::new(10.0, 20.0, 30.0), ..Transform::IDENTITY };
        Session {
            camera: PositionedCamera {
                eye: Point3::new(1.0 / 3.0, 50.0, 75.0),
                target: Point3::new(0.0, 0.1, 0.0),
                up: Vector3::new(0.0, 1.0, 0.0),
                aspect: 1.5,
                projection: Projection::Orthographic { half_height: 12.5, half_depth: 400.0 },
            },
            lighting: Lighting {
                headlight: HeadlightMode::Off,
                headlight_rig: default_headlight_rig(),
                lights: vec![PositionedLight::world(light, placed)],
                environment: Some(0.5),
            },
            construction_plane: Plane::from_point(Vector3::unit_x(), Point3::new(7.0, 0.0, 0.0)),
            grid: GridConfig { visible: false, minor_spacing: 2.5, major_every: 4, ..GridConfig::default() },
        }
    }

    /// A document with a box on Steel, a hidden cylinder, the preset library
    /// and an environment map.
    fn document() -> Document {
        let mut document = Document::new(Scene::default());
        document.add_preset_materials();
        let options = CadTessellationOptions::default();
        let frame = Transform { position: Point3::new(0.0, 0.0, -5.0), ..Transform::IDENTITY };
        let bracket = document.add_part("Bracket", Shape::cube(2.0), frame, &options).unwrap();
        let shaft = Shape::cylinder_radius_height(0.5, 3.0);
        let shaft = document.add_part("Shaft", shaft, Transform::IDENTITY, &options).unwrap();
        let steel = document.materials().iter().find(|m| m.name == "Steel").unwrap().id();
        document.assign_material(&[bracket], steel);
        document.set_part_visibility(shaft, Visibility::Invisible);
        set_environment(document.scene(), HDR_IMAGE.to_vec()).unwrap();
        document
    }

    fn debug(value: &impl std::fmt::Debug) -> String {
        format!("{value:?}")
    }

    #[test]
    fn a_saved_document_opens_as_it_was() {
        let path = std::env::temp_dir().join("duck_modeler_io_document.duck");
        let saved = document();
        save_document(&path, &saved, session()).unwrap();
        let opened = open_document(&path);
        let _ = std::fs::remove_file(&path);
        let (opened, restored) = opened.unwrap();

        let names = |doc: &Document| doc.parts().map(|part| part.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&opened), ["Bracket", "Shaft"]);
        for (before, after) in saved.parts().zip(opened.parts()) {
            assert_eq!(after.frame(), before.frame());
            assert!((after.shape().volume() - before.shape().volume()).abs() < 1e-12);
            assert_eq!(opened.part_visibility(after.id), saved.part_visibility(before.id));
            assert_eq!(opened.part_material(after.id), saved.part_material(before.id));
        }
        assert_eq!(opened.materials().len(), saved.materials().len());
        assert_eq!(opened.undo_label(), None);

        let scene = opened.scene().lock();
        let map = scene.active_environment_map().and_then(|id| scene.get_environment_map(id));
        let image = map.map(|map| debug(map.source()));
        assert_eq!(image, Some(debug(&EnvironmentSource::EquirectangularHdr(HDR_IMAGE.to_vec()))));

        let expected = session();
        assert_eq!(debug(&restored.camera), debug(&expected.camera));
        assert_eq!(restored.lighting.headlight, HeadlightMode::Off);
        assert_eq!(debug(&restored.lighting.headlight_rig), debug(&expected.lighting.headlight_rig));
        assert_eq!(debug(&restored.lighting.lights), debug(&expected.lighting.lights));
        assert_eq!(restored.lighting.environment, Some(0.5));
        assert_eq!(debug(&restored.construction_plane), debug(&expected.construction_plane));
        let grid = |grid: &GridConfig| (grid.visible, grid.size, grid.minor_spacing, grid.major_every);
        assert_eq!(grid(&restored.grid), grid(&expected.grid));
    }

    /// Writes a document file at `path` holding only `manifest`.
    fn manifest_only(path: &Path, manifest: &serde_json::Value) {
        let mut zip = ZipWriter::new(std::fs::File::create(path).unwrap());
        zip.start_file(MANIFEST_ENTRY, SimpleFileOptions::default()).unwrap();
        serde_json::to_writer(&mut zip, manifest).unwrap();
        zip.finish().unwrap();
    }

    /// The manifest of the document file at `path`.
    fn manifest_of(path: &Path) -> serde_json::Value {
        let mut archive = ZipArchive::new(Cursor::new(std::fs::read(path).unwrap())).unwrap();
        serde_json::from_slice(&read_entry(&mut archive, MANIFEST_ENTRY).unwrap()).unwrap()
    }

    #[test]
    fn a_newer_format_is_refused() {
        let path = std::env::temp_dir().join("duck_modeler_io_newer.duck");
        save_document(&path, &Document::new(Scene::default()), session()).unwrap();
        let mut manifest = manifest_of(&path);
        manifest["version"] = (FORMAT_VERSION + 1).into();
        manifest_only(&path, &manifest);
        let opened = open_document(&path);
        let _ = std::fs::remove_file(&path);

        let error = format!("{:#}", opened.err().expect("a newer format is refused"));
        assert!(error.contains("newer"), "{error}");
    }

    #[test]
    fn a_missing_brep_names_its_part() {
        let path = std::env::temp_dir().join("duck_modeler_io_missing.duck");
        save_document(&path, &document(), session()).unwrap();
        let manifest = manifest_of(&path);
        manifest_only(&path, &manifest);
        let opened = open_document(&path);
        let _ = std::fs::remove_file(&path);

        let error = format!("{:#}", opened.err().expect("a missing B-Rep fails the open"));
        assert!(error.contains("Bracket"), "{error}");
    }
}
