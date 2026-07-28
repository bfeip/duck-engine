//! Scene-agnostic rendering core for duck-engine.
//!
//! This crate contains the parts of the renderer that do not depend on any
//! particular object model: GPU plumbing, render-target management, headless
//! readback, and reusable caching utilities. It deliberately has no dependency
//! on `duck-engine-scene` or `duck-engine-common` — nothing here knows what a
//! scene, mesh, material, or camera is.
//!
//! The standard scene renderer (`duck-engine-renderer`) builds on this crate
//! and re-exports it as `render_core`.

mod gen_cache;
mod gpu;
mod host;
mod pipeline_cache;
mod readback;
mod shader;
mod targets;
mod texture;
mod workflow;

pub use gen_cache::GenCache;
pub use gpu::{Gpu, GpuCapabilities};
pub use host::{RenderHost, RgbaPixels};
pub use pipeline_cache::PipelineCache;
pub use readback::ReadbackTarget;
pub use shader::ShaderLibrary;
pub use targets::{
    FrameTargets, TargetConfig, TargetFeatures, highest_supported_sample_count,
};
pub use texture::{GpuTexture, MaskChannels};
pub use workflow::{FrameFamily, RenderWorkflow};
