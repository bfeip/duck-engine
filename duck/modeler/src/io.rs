//! STEP/IGES files to and from the Document. The platform
//! [`Host`](crate::platform::Host) decides where the files come from and go.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use duck_engine_import_export::cad;
use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::common::WorldUnits;

use crate::document::{Document, DOCUMENT_UNITS};
use crate::platform::FileFilter;

/// The unit OCCT reads STEP and IGES files into, whatever unit a file declares,
/// and writes them in. Shapes convert between it and [`DOCUMENT_UNITS`] on the
/// way in and out.
const FILE_UNITS: WorldUnits = WorldUnits::MILLIMETER;

/// The files [`import_cad_file`] reads.
pub const IMPORT_FILTER: FileFilter =
    FileFilter { name: "CAD (STEP/IGES)", extensions: cad::CAD_EXTENSIONS };

/// The files [`export_cad_file`] writes, STEP first.
pub const EXPORT_FILTERS: [FileFilter; 2] = [
    FileFilter { name: "STEP", extensions: &["step", "stp"] },
    FileFilter { name: "IGES", extensions: &["iges", "igs"] },
];

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
        let mut options = CadTessellationOptions { frame: part.frame, ..options.clone() };
        if let Some(color) = part.color {
            options.face_material.set_base_color_factor(color);
            // A part that carries its own color keeps it whatever its geometry
            // class, so drop the sketch appearance for this part.
            options.free_face_material = None;
            options.free_line_material = None;
        }
        let name = part.name.unwrap_or_else(|| format!("{stem} {}", i + 1));
        document.add_part(name, part.shape, &options)?;
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

#[cfg(test)]
mod tests {
    use duck_engine_scene::Scene;
    use opencascade::primitives::Shape;

    use super::*;

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
        document.lock().unwrap().add_part("Box", cube, &options).unwrap();
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
}
