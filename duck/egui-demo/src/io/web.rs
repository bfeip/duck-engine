//! Web file I/O: `rfd` async browser dialogs + in-memory bytes.
//!
//! Dialogs run on the JS event loop via `spawn_local`; the resulting bytes are
//! stashed in the `App`'s shared buffers and consumed on the next frame.

use std::sync::{Arc, Mutex};

use duck_engine_viewer::import_export;
use duck_engine_viewer::scene::Scene;

use crate::App;

impl App<'_> {
    /// Consume any bytes delivered by the browser file dialogs.
    pub(crate) fn process_pending_io(&mut self) {
        if self.state.is_none() {
            return;
        }
        let scene_bytes = self.pending_scene_bytes.borrow_mut().take();
        // The scene loads first: it replaces the whole `SceneData`, which would
        // discard an environment map applied earlier in the same frame.
        if let Some(bytes) = scene_bytes {
            self.load_scene_bytes(bytes);
        }
        let hdr_bytes = self.pending_hdr_bytes.borrow_mut().take();
        if let Some(bytes) = hdr_bytes {
            self.load_hdr_bytes(bytes);
        }
    }

    pub(crate) fn open_hdr_file_dialog(&mut self) {
        let sink = self.pending_hdr_bytes.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("HDR", &["hdr"])
                .pick_file()
                .await
            {
                *sink.borrow_mut() = Some(file.read().await);
            }
        });
    }

    fn load_hdr_bytes(&mut self, bytes: Vec<u8>) {
        let Some(state) = self.state.as_mut() else { return };
        let scene_arc = state.scene();
        let mut scene = scene_arc.lock();
        let env_id = scene.add_environment_map_from_hdr_data(bytes);
        scene.set_active_environment_map(Some(env_id));
        log::info!("Loaded HDR environment");
    }

    pub(crate) fn open_scene_file_dialog(&mut self) {
        let sink = self.pending_scene_bytes.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("3D Scenes", &["glb", "gltf"])
                .pick_file()
                .await
            {
                *sink.borrow_mut() = Some(file.read().await);
            }
        });
    }

    fn load_scene_bytes(&mut self, bytes: Vec<u8>) {
        use import_export::{load_sync, SceneSource, LoadOptions};
        let Some(state) = self.state.as_mut() else { return };
        match load_sync(SceneSource::Bytes(bytes), LoadOptions::default()) {
            Ok(result) => {
                let bounds = result.scene.bounding().bounds;
                state.viewer.set_view_scene(state.view_id, Scene::new(result.scene));
                let mut view = state.view_mut();
                if let Some(camera) = result.camera {
                    view.set_camera(camera);
                } else if let Some(bounds) = bounds {
                    view.with_camera_mut(|c| c.fit_to_bounds(&bounds));
                }
                log::info!("Loaded scene");
            }
            Err(e) => log::error!("Failed to load scene: {}", e),
        }
    }
}
