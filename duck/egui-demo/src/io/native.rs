//! Native file I/O: `rfd` filesystem dialogs + path-based loading.


use duck_engine_viewer::import_export;
use duck_engine_viewer::scene::Scene;

use crate::App;

impl App<'_> {
    /// Consume any pending file I/O queued since the last frame.
    ///
    /// The scene loads first: it replaces the whole `SceneData`, which would
    /// discard an environment map applied earlier in the same frame.
    pub(crate) fn process_pending_io(&mut self) {
        if self.pending_scene_load_path.is_some() {
            self.load_scene_file();
        }
        if self.pending_hdr_path.is_some() {
            self.load_hdr_file();
        }
    }

    pub(crate) fn open_hdr_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new().add_filter("HDR", &["hdr"]).pick_file() {
            self.pending_hdr_path = Some(path);
        }
    }

    fn load_hdr_file(&mut self) {
        let Some(path) = self.pending_hdr_path.take() else { return };
        let Some(state) = self.state.as_mut() else { return };
        let path_str = path.display().to_string();
        let scene_arc = state.viewer.scene();
        let mut scene = scene_arc.lock();
        let env_id = scene.add_environment_map_from_hdr_path(&path);
        scene.set_active_environment_map(Some(env_id));
        log::info!("Loaded HDR environment: {}", path_str);
    }

    pub(crate) fn open_scene_file_dialog(&mut self) {
        #[allow(unused_mut)]
        let mut extensions: Vec<&str> = vec!["glb", "gltf"];

        #[cfg(feature = "assimp")]
        extensions.extend_from_slice(import_export::assimp::ASSIMP_EXTENSIONS);

        #[cfg(feature = "usd")]
        extensions.extend_from_slice(import_export::usd::USD_EXTENSIONS);

        #[cfg(feature = "cad")]
        extensions.extend_from_slice(import_export::cad::CAD_EXTENSIONS);

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("3D Scenes", &extensions)
            .pick_file()
        {
            self.pending_scene_load_path = Some(path);
        }
    }

    fn load_scene_file(&mut self) {
        use import_export::{load_sync, SceneSource, LoadOptions};
        let Some(path) = self.pending_scene_load_path.take() else { return };
        let Some(state) = self.state.as_mut() else { return };
        let path_str = path.display().to_string();
        match load_sync(SceneSource::Path(path), LoadOptions::default()) {
            Ok(result) => {
                let bounds = result.scene.bounding().bounds;
                state.viewer.set_scene(Scene::new(result.scene));
                if let Some(camera) = result.camera {
                    state.viewer.set_camera(camera);
                } else if let Some(bounds) = bounds {
                    state.viewer.with_camera_mut(|c| c.fit_to_bounds(&bounds));
                }
                log::info!("Loaded scene: {}", path_str);
            }
            Err(e) => log::error!("Failed to load scene {}: {}", path_str, e),
        }
    }
}
