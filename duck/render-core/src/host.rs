use std::ops::{Deref, DerefMut};

use crate::{FrameFamily, FrameTargets, Gpu, ReadbackTarget, TargetConfig, Workflow};

/// Tightly-packed RGBA8 pixel data read back from the GPU.
pub struct RgbaPixels {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Owns the frame-agnostic per-frame machinery: the GPU handles, the shared
/// frame targets, the active workflow, and an optional readback target for
/// headless rendering. Knows nothing about what a frame contains.
///
/// The caller builds an `F::Frame<'_>` from its own state each frame and hands
/// it to [`render`](Self::render). Because the frame is built from the
/// *caller's* fields and the host borrows only its own, `&mut host` and the
/// frame coexist without conflict.
pub struct RenderHost<F: FrameFamily> {
    gpu: Gpu,
    targets: FrameTargets,
    workflow: Workflow<F>,
    /// Cached for headless rendering, reused across frames at the same size.
    readback: Option<ReadbackTarget>,
}

impl<F: FrameFamily> RenderHost<F> {
    /// Create a host running `workflow`. The attachments come from the
    /// workflow's passes. See [`Workflow::target_features`].
    #[must_use]
    pub fn new(gpu: Gpu, config: TargetConfig, workflow: Workflow<F>) -> Self {
        let targets = FrameTargets::new(&gpu, config, workflow.target_features().clone());
        Self { gpu, targets, workflow, readback: None }
    }

    #[must_use] 
    pub const fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    #[must_use] 
    pub const fn targets(&self) -> &FrameTargets {
        &self.targets
    }

    #[must_use] 
    pub const fn config(&self) -> TargetConfig {
        self.targets.config()
    }

    /// The active rendering workflow.
    #[must_use]
    pub const fn workflow(&self) -> &Workflow<F> {
        &self.workflow
    }

    /// Edit the active workflow in place — insert, replace or remove a pass, or
    /// retune one via [`Workflow::pass_mut`].
    ///
    /// The returned guard reconciles the frame's attachments with the edited
    /// pass list when it drops, so a workflow can never run against
    /// attachments it did not ask for. Reconciliation is skipped when the edit
    /// changed no structure.
    pub fn workflow_mut(&mut self) -> WorkflowGuard<'_, F> {
        let revision = self.workflow.revision();
        WorkflowGuard { host: self, revision }
    }

    /// Replace the active rendering workflow.
    ///
    /// The new workflow takes effect immediately on the next frame, with
    /// attachments reallocated to match its passes. The previous workflow and
    /// all its GPU resources are dropped.
    pub fn set_workflow(&mut self, workflow: Workflow<F>) {
        self.workflow = workflow;
        self.reconcile_targets();
    }

    /// Reallocate attachments to match the current workflow, then let its
    /// passes rebuild their size-dependent resources.
    ///
    /// Infallible: conflicting declarations are rejected when a pass is
    /// inserted, so by the time a workflow is installed its feature union is
    /// already known good.
    fn reconcile_targets(&mut self) {
        let features = self.workflow.target_features();
        if features != self.targets.features() {
            let features = features.clone();
            self.targets.set_features(&self.gpu, features);
        }
        self.workflow.resize(&self.gpu, &self.targets);
        self.readback = None;
    }

    /// Recreate size-dependent attachments and forward to the workflow.
    /// Ignores zero dimensions.
    pub fn resize(&mut self, size: (u32, u32)) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        self.targets.resize(&self.gpu, size);
        self.workflow.resize(&self.gpu, &self.targets);
        self.readback = None;
    }

    /// Execute the active workflow for one frame. The encoder is not
    /// submitted — the caller is responsible for that.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut F::Frame<'_>,
    ) {
        self.workflow
            .execute(&self.gpu, &self.targets, encoder, view, frame);
    }

    /// Render one frame into an owned offscreen target and read it back.
    ///
    /// Creates and submits its own encoder, then blocks until the GPU work
    /// completes. The readback target is cached and reused across calls at
    /// the same size. Assumes a 4-byte-per-pixel target format.
    /// 
    /// # Errors
    /// 
    /// Will return `Err` if readback from the GPU failed (see [`ReadbackTarget::read`])
    pub fn render_to_rgba(&mut self, frame: &mut F::Frame<'_>) -> anyhow::Result<RgbaPixels> {
        let (width, height) = self.targets.size();

        if self
            .readback
            .as_ref()
            .is_none_or(|r| r.size() != (width, height))
        {
            self.readback = Some(ReadbackTarget::new(
                &self.gpu.device,
                width,
                height,
                self.targets.format(),
            ));
        }
        #[expect(clippy::missing_panics_doc, reason = "infallible")]
        let target = self.readback.take().unwrap();

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Render Encoder"),
            });
        self.render(&mut encoder, target.view(), frame);
        target.encode_copy(&mut encoder);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let data = target.read(&self.gpu.device)?;
        self.readback = Some(target);

        Ok(RgbaPixels { width, height, data })
    }
}

/// Mutable access to a host's [`Workflow`], reconciling the frame's
/// attachments with the pass list on drop.
///
/// Returned by [`RenderHost::workflow_mut`]; derefs to the workflow, so it is
/// used as if it were one.
pub struct WorkflowGuard<'a, F: FrameFamily> {
    host: &'a mut RenderHost<F>,
    /// The workflow's revision when the guard was taken.
    revision: u64,
}

impl<F: FrameFamily> Deref for WorkflowGuard<'_, F> {
    type Target = Workflow<F>;

    fn deref(&self) -> &Self::Target {
        &self.host.workflow
    }
}

impl<F: FrameFamily> DerefMut for WorkflowGuard<'_, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.host.workflow
    }
}

impl<F: FrameFamily> Drop for WorkflowGuard<'_, F> {
    fn drop(&mut self) {
        if self.host.workflow.revision() != self.revision {
            self.host.reconcile_targets();
        }
    }
}
