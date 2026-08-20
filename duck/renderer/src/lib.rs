//! wgpu rendering for duck-engine scenes.
//!
//! This crate draws a scene — the GPU-free description defined by the
//! [`scene`] crate — with wgpu. It mirrors scene resources into GPU buffers
//! and textures, keeps the copies in sync through the scene's generation
//! counters, batches instances into draw calls, and runs a configurable
//! sequence of render passes over them. It knows nothing about windows or
//! input: it renders into any `wgpu::TextureView`, or with no display at all
//! straight to an image.
//!
//! Two dependencies are re-exported at the crate root: [`scene`], and
//! [`render_core`] — the scene-agnostic dispatch layer this crate is built on,
//! whose [`Gpu`], [`FrameTargets`], and [`RenderWorkflow`] appear throughout
//! this crate's API.
//!
//! # Context, scene, and view state
//!
//! Render state is split by what it belongs to:
//!
//! - [`RenderContext`] — one per device and target configuration. The GPU
//!   handles, the bind group layouts, and the material pipeline cache with its
//!   shader generator. The target format, MSAA sample count, and compute
//!   availability are baked in at construction; nothing here is per-scene, so
//!   every scene drawn at that configuration shares one set of pipelines.
//! - [`SceneResources`] — one per scene. The GPU caches for meshes, textures,
//!   material bind groups, and processed environment maps, generation-synced
//!   to the scene.
//! - [`Renderer`] — one per view. The frame targets (depth and MSAA
//!   attachments), the active workflow, and this view's camera and lights
//!   uniforms and background color.
//!
//! Any number of renderers draw through one `RenderContext`, and all of them
//! render at its format and sample count. A renderer is not bound to a scene —
//! each render call takes the scene's `SceneResources`, the
//! [`PositionedCamera`](scene::PositionedCamera) to render from, and extra
//! per-view (camera-space) lights to compose after the scene's own.
//!
//! # The frame
//!
//! [`SceneResources::prepare`] brings the GPU caches up to date with the
//! scene. It is destructive — it drains the scene's removal queue — so it must
//! run exactly once per scene per frame, before any renderer over that scene
//! draws. Each view then calls [`Renderer::render_scene_to_view`] with its
//! camera, extra lights, and target texture view, recording into an encoder
//! the caller submits.
//!
//! For rendering without a surface, [`Renderer::render_scene_to_image`]
//! returns the frame as an RGBA image instead. Unlike `render_scene_to_view`
//! it is self-contained: it locks the scene, prepares `shared`, and submits
//! internally. Pair it with [`Gpu::headless`], which acquires a device with no
//! display attached.
//!
//! # Workflows and passes
//!
//! What a renderer draws each frame is decided by its workflow, an ordered
//! sequence of render passes. The default is [`ShadedWorkflow`] — PBR-lit
//! faces, lines and points, selection outlines, overlay geometry —
//! and [`HiddenLineWorkflow`] renders technical-drawing style line work;
//! switch with [`Renderer::set_workflow`].
//!
//! Custom rendering plugs in at two levels. A pass implements
//! [`SceneRenderPass`] and reads the per-frame [`SceneFrame`]: the scene, the
//! collected [`DrawBatch`]es, and the standard bind groups
//! ([`SceneBindingRefs`]), drawn via [`SceneFrame::draw_batch`] or raw wgpu
//! calls. A whole workflow implements
//! [`RenderWorkflow<SceneFrames>`](RenderWorkflow). Custom WESL shaders
//! compile against the engine's shader modules with
//! [`RenderContext::compile_user_wesl`], pipelines come from
//! [`RenderContext::custom_pipeline_builder`], and the bind group
//! conventions those shaders rely on are the constants in [`abi`]. The
//! `gooch` example walks a custom workflow end to end.
//!
//! # Highlights
//!
//! Render calls take an optional [`HighlightQuery`], through which the caller
//! reports which nodes and sub-geometry (faces, edges, points) to highlight
//! and with what [`HighlightConfig`]; the workflow renders node outlines and
//! sub-geometry tints accordingly. The renderer defines only the query trait —
//! what is highlighted, and why, is the caller's concern.
//!
//! # Image-based lighting
//!
//! When the scene has an active environment map, `prepare` processes its HDR
//! source into the textures PBR shading samples, and lit materials pick it up
//! automatically; see [`ibl`]. Processing requires compute shader support
//! (absent on WebGL), reported by [`Gpu`] at acquisition and passed to
//! [`RenderContext::new`].
//!
//! # Example
//!
//! ```no_run
//! use duck_engine_renderer::{Gpu, RenderContext, Renderer, SceneResources};
//! use duck_engine_renderer::scene::{PositionedCamera, Scene, SceneData};
//! use duck_engine_renderer::scene::common::{Point3, RgbaColor, Vector3};
//! use duck_engine_renderer::scene::resource::{
//!     FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! // One RenderContext, one SceneResources per scene, one Renderer per view.
//! let (gpu, caps) = pollster::block_on(Gpu::headless())?;
//! let mut ctx =
//!     RenderContext::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb, 1, caps.has_compute);
//! let mut shared = SceneResources::new(&ctx);
//! let mut renderer = Renderer::new(&mut ctx, 800, 600);
//!
//! // A red sphere.
//! let mut data = SceneData::new();
//! let mesh = data.add_mesh(Mesh::sphere(1.0, 48, 24, PrimitiveType::TriangleList));
//! let material =
//!     data.add_face_material(FaceMaterial::new().with_base_color_factor(RgbaColor::RED));
//! data.add_instance_node(
//!     None, // parent; None creates a root node
//!     Instance::new(mesh).with_face_material(material),
//!     Some("sphere".to_string()),
//!     Default::default(),
//!     NodeFlags::NONE,
//! )?;
//! let mut scene = Scene::new(data);
//!
//! let camera = PositionedCamera {
//!     eye: Point3::new(0.0, 0.0, 3.5),
//!     target: Point3::new(0.0, 0.0, 0.0),
//!     up: Vector3::new(0.0, 1.0, 0.0),
//!     aspect: 800.0 / 600.0,
//!     fovy: 45.0,
//!     znear: 0.1,
//!     zfar: 100.0,
//!     ortho: false,
//! };
//!
//! // Headless one-shot: locks the scene, prepares, renders, reads back.
//! let image =
//!     renderer.render_scene_to_image(&mut ctx, &mut shared, &mut scene, &camera, &[], None)?;
//! # Ok(()) }
//! ```

/// The GPU-free scene description this crate renders.
pub use duck_engine_scene as scene;
/// The scene-agnostic rendering core this crate is built on.
pub use duck_engine_render_core as render_core;

pub(crate) fn rgba_to_wgpu_color(c: scene::common::RgbaColor) -> wgpu::Color {
    wgpu::Color { r: c.r as f64, g: c.g as f64, b: c.b as f64, a: c.a as f64 }
}

pub mod abi;
pub mod ibl;
mod highlight_query;
mod renderer;
mod shaders;

pub use renderer::{
    BatchKey, BatchMaterial, CustomPipelineBuilder, DrawBatch, DrawData, HiddenLineConfig,
    HiddenLineWorkflow, InstanceTransform, RenderContext, Renderer, ResolvedLight,
    SceneBindingRefs, SceneFrame, SceneFrames, SceneRenderPass, SceneResources, SceneWorkflow,
    ShadedWorkflow, SubGeomBatch, instance_buffer_layout, vertex_buffer_layout,
};
pub use highlight_query::{HighlightConfig, HighlightQuery};

// Core dispatch types needed to author custom workflows/passes.
pub use render_core::{FrameTargets, Gpu, RenderWorkflow};
