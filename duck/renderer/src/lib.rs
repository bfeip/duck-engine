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
//! whose [`Gpu`], [`FrameTargets`], [`Pass`] and [`Workflow`] appear throughout
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
//! [`PositionedCamera`](scene::PositionedCamera) to render from, and the
//! [`PositionedLight`](scene::PositionedLight)s to light it with. Lights are
//! not scene resources: the caller supplies the whole list, and the renderer
//! resolves each one against the space it is posed in.
//!
//! # The frame
//!
//! [`SceneResources::prepare`] brings the GPU caches up to date with the
//! scene. It is destructive — it drains the scene's removal queue — so it must
//! run exactly once per scene per frame, before any renderer over that scene
//! draws. Each view then calls [`Renderer::render_scene_to_view`] with its
//! camera, lights, and target texture view, recording into an encoder the
//! caller submits.
//!
//! For rendering without a surface, [`Renderer::render_scene_to_image`]
//! returns the frame as an RGBA image instead. Unlike `render_scene_to_view`
//! it is self-contained: it locks the scene, prepares `shared`, and submits
//! internally. Pair it with [`Gpu::headless`], which acquires a device with no
//! display attached.
//!
//! # Workflows and passes
//!
//! What a renderer draws each frame is decided by its workflow. A
//! [`SceneWorkflow`] is a named, ordered, id-addressable list of passes — not a
//! type per rendering style — so there is one concept to learn, the pass, and
//! one way to compose passes.
//!
//! [`workflow::shaded`] builds the default: PBR-lit faces, lines and points,
//! selection outlines, and overlay geometry. [`workflow::hidden_line`] builds
//! technical-drawing style line work. Install either with
//! [`Renderer::set_workflow`], or hand one to [`Renderer::with_workflow`] at
//! construction.
//!
//! Because a workflow is an editable list, a caller adjusts a stock one rather
//! than rebuilding it. Every built-in pass is public in [`pass`] and registered
//! under a constant in [`pass::ids`]:
//!
//! ```ignore
//! let mut workflow = workflow::shaded(&mut renderer.pass_builder(&mut ctx));
//!
//! // Insert a pass at a named point.
//! workflow.insert_after(pass::ids::FACES, PassId("ghost"), GhostPass::new(&b))?;
//! // Reuse a built-in pass from another workflow.
//! workflow.insert_after(pass::ids::LINES_AND_POINTS, pass::ids::SILHOUETTE, s)?;
//! // Retune one in place — no pipeline is rebuilt.
//! workflow.pass_mut::<FlatColorPass>(pass::ids::SOLID)?.set_color(queue, c);
//! // Or drop one.
//! workflow.remove(pass::ids::SUB_GEOM_HIGHLIGHT);
//! ```
//!
//! [`Renderer::workflow_mut`] does the same to the workflow a renderer is
//! already running.
//!
//! ## Writing a pass
//!
//! A pass implements [`Pass<SceneFrames>`](Pass) and reads the per-frame
//! [`SceneFrame`]: the scene, the collected [`DrawBatch`]es, and the standard
//! bind groups ([`SceneBindingRefs`]). It can draw geometry through its own
//! pipeline with [`SceneFrame::draw_batch`], or with full engine material
//! shading via [`SceneFrame::bind_scene_groups`] and
//! [`SceneFrame::draw_batches`] — the same methods the built-in passes use.
//! [`SceneFrame::surface_pipeline`] and [`SceneFrame::material_bind_group`]
//! expose the shared caches for finer control.
//!
//! Every pass — built-in or user-written — is constructed from a
//! [`PassBuilder`], which carries the device, target configuration, adapter
//! capabilities, shared [`BindGroupLayouts`], and the engine shader library.
//! That common interface is what lets passes from different sources compose in
//! one workflow. Custom WESL shaders compile against the engine's modules with
//! [`PassBuilder::compile_wesl`], pipelines come from
//! [`PassBuilder::pipeline`], and the bind group conventions those shaders rely
//! on are the constants in [`abi`].
//!
//! ## Attachments
//!
//! A pass does not own the attachments it shares. It declares what it needs
//! from [`Pass::target_features`] — the depth buffer, a readable depth buffer,
//! or a named [`AuxTarget`] — the workflow keeps the union across its passes,
//! and the host allocates and resizes them. That is how one pass hands a
//! texture to another: both declare the same [`AuxTarget`] and neither holds
//! the other. It also means a workflow only pays for what its passes use.
//!
//! The `gooch` example builds a wholly custom workflow; `custom_pass` edits a
//! stock one.
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
//! use duck_engine_renderer::{Gpu, GpuOptions, RenderContext, Renderer, SceneResources};
//! use duck_engine_renderer::scene::{PositionedCamera, Projection, Scene, SceneData};
//! use duck_engine_renderer::scene::common::{Point3, RgbaColor, Vector3};
//! use duck_engine_renderer::scene::resource::{
//!     FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! // One RenderContext, one SceneResources per scene, one Renderer per view.
//! let (gpu, caps) = pollster::block_on(Gpu::headless(GpuOptions::default()))?;
//! let mut ctx =
//!     RenderContext::new(gpu, wgpu::TextureFormat::Rgba8UnormSrgb, 1, caps);
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
//!     projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
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
pub mod shaders;

pub use renderer::{
    BatchKey, BatchMaterial, BindGroupLayouts, CustomPipelineBuilder, DrawBatch, DrawData,
    DrawOptions, HiddenLineConfig, InstanceTransform, MaterialTextureSlot, PassBuilder,
    PipelineCacheKey, RenderContext, Renderer, SceneBindingRefs, SceneFrame, SceneFrames,
    ScenePass, SceneResources, SceneWorkflow, SubGeomBatch, SurfaceConfig, TexturePresence,
    instance_buffer_layout, pass, scene_color_format, vertex_buffer_layout, workflow,
};
pub use highlight_query::{HighlightConfig, HighlightQuery};

// Core dispatch types needed to author custom workflows/passes.
pub use render_core::{
    AuxKind, AuxTarget, FrameTargets, Gpu, GpuCapabilities, GpuOptions, Pass, PassId,
    TargetFeatures, Workflow, WorkflowError, WorkflowGuard,
};
