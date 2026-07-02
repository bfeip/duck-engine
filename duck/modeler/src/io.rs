//! File menu import/export: native file dialogs + Document registration for
//! STEP/IGES files.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use duck_engine_import_export::cad;
use duck_engine_scene::cad::CadTessellationOptions;

use crate::document::Document;

/// The document works in OCCT's native STEP/IGES unit (mm), so shapes pass
/// through import and export without unit scaling.
const UNIT_SCALE: f64 = 1.0;

/// Pick a STEP/IGES file and register each of its leaf parts in `document`.
pub fn import_cad_dialog(
    document: &Arc<Mutex<Document>>,
    options: &CadTessellationOptions,
) -> Result<()> {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("CAD (STEP/IGES)", cad::CAD_EXTENSIONS)
        .pick_file()
    else {
        return Ok(());
    };

    let parts = cad::load_cad_parts(&path, UNIT_SCALE)?;
    if parts.is_empty() {
        log::warn!("No parts found in {}", path.display());
        return Ok(());
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Imported").to_string();
    let mut document = document.lock().unwrap();
    for (i, part) in parts.into_iter().enumerate() {
        let mut options = options.clone();
        if let Some(color) = part.color {
            options.face_material.set_base_color_factor(color);
        }
        let name = part.name.unwrap_or_else(|| format!("{stem} {}", i + 1));
        document.add_part(name, part.shape, &options)?;
    }
    Ok(())
}

/// Pick a destination and write all document parts as one STEP/IGES compound.
pub fn export_cad_dialog(document: &Arc<Mutex<Document>>) -> Result<()> {
    let document = document.lock().unwrap();
    if document.parts().next().is_none() {
        log::warn!("Nothing to export: document has no parts");
        return Ok(());
    }

    let Some(mut path) = rfd::FileDialog::new()
        .add_filter("STEP", &["step", "stp"])
        .add_filter("IGES", &["iges", "igs"])
        .set_file_name("export.step")
        .save_file()
    else {
        return Ok(());
    };
    let known_ext =
        path.extension().and_then(|e| e.to_str()).is_some_and(cad::is_cad_extension);
    if !known_ext {
        path.set_extension("step");
    }

    cad::save_cad_shapes(&path, document.parts().map(|p| &p.shape), UNIT_SCALE)?;
    log::info!("Exported {} part(s) to {}", document.parts().count(), path.display());
    Ok(())
}
