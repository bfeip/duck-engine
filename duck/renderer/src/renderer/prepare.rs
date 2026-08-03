use anyhow::Result;

use crate::scene::Scene;

use super::batching::collect_scene_data;
use super::mesh::MeshGpuResources;
use super::scene_bindings::LightsArrayUniform;
use super::texture::create_texture_gpu_resources;
use super::Renderer;

impl Renderer {
    /// Prepare all GPU resources for a scene before rendering.
    ///
    /// Ensures all textures, materials, and meshes have up-to-date GPU resources,
    /// syncs lights, and processes the active environment map. Each subsystem owns
    /// its own generation-synced cache; this just drives them. Call before
    /// `render_scene_to_view()`.
    //
    // TODO: This iterates more or less everything in the scene. For performance in the future,
    // we should keep track of the need for these updates in the scene. I.e. mark things as
    // dirty if they need to be reified.
    pub fn prepare_scene(&mut self, scene: &mut Scene) -> Result<()> {
        // 1. Textures first (materials sample them).
        for texture in scene.textures() {
            self.gpu_textures.try_ensure(texture.id(), texture.generation(), || {
                create_texture_gpu_resources(texture, &self.host.gpu().device, &self.host.gpu().queue)
            })?;
        }

        // 2. Materials (face/line/point), reading uploaded textures.
        self.materials.prepare(&self.host.gpu().device, scene, &self.gpu_textures)?;

        // 3. Meshes.
        // TODO: For wireframe mode, generate line index buffers from triangle primitives
        // using MeshPrimitive::to_line_list(), gated on a wireframe flag, before upload.
        for mesh in scene.meshes() {
            self.gpu_meshes.ensure(mesh.id, mesh.generation(), || {
                MeshGpuResources::new(mesh, &self.host.gpu().device)
            });
        }

        // 4. Lights.
        self.bindings.sync_lights(&self.host.gpu().queue, scene, || {
            let frame_data = collect_scene_data(scene);
            LightsArrayUniform::from_resolved_lights(&frame_data.lights)
        });

        // 5. Environment maps for IBL.
        if let Some(env_id) = scene.active_environment_map()
            && let Some(env_map) = scene.get_environment_map(env_id)
        {
            self.ibl_resources
                .process_environment(&self.host.gpu().device, &self.host.gpu().queue, env_map)?;
        }

        Ok(())
    }
}
