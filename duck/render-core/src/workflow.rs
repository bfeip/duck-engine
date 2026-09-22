use std::any::Any;

use crate::{AuxConflict, FrameTargets, Gpu, TargetFeatures};

/// A *frame family*: a type-level tag naming the per-frame data type for one
/// rendering stack.
///
/// # Why this trait exists
///
/// A [`Pass`] reads per-frame data that borrows (the scene, bind groups,
/// caches), so that data type carries a lifetime — conceptually `Frame<'a>`. We
/// *also* want passes to be runtime-composable and user-extensible, i.e. held
/// as `Box<dyn Pass<…>>`.
///
/// Those requirements collide. The natural form,
///
/// ```ignore
/// trait Pass { type Frame<'a>; fn execute(&mut self, f: &mut Self::Frame<'_>); }
/// ```
///
/// is not `dyn`-compatible: a trait with a *generic associated type* cannot be
/// made into a trait object, so it could never be boxed. `FrameFamily` lifts the
/// lifetime-parameterized type out into its own trait, leaving `Pass` with a
/// plain type parameter `F` — and `dyn Pass<F>` *is* `dyn`-compatible. This is
/// the standard stable-Rust workaround for "I need a trait object whose
/// associated type has a lifetime."
///
/// # Implementing
///
/// An implementer is a pure type-level token: it is **never constructed** and
/// carries no data, so an uninhabited enum is the natural choice.
/// [`RenderHost<F>`](crate::RenderHost) uses `F` only to name `F::Frame<'_>`;
/// no value of `F` ever exists at runtime.
///
/// ```
/// use duck_engine_render_core::FrameFamily;
///
/// struct MyFrameData<'a> { scene: &'a str }
///
/// enum MyFrames {}
/// impl FrameFamily for MyFrames {
///     type Frame<'a> = MyFrameData<'a>;
/// }
/// // Passes for this stack then implement `Pass<MyFrames>`.
/// ```
pub trait FrameFamily: 'static {
    /// The per-frame data type, parameterized by the frame's borrow lifetime.
    ///
    /// The core never inspects it — its contents are defined entirely by the
    /// rendering stack built on top. A custom stack can put anything here,
    /// including `()` for a frame that borrows nothing.
    type Frame<'a>;
}

/// Downcast support for boxed passes, so [`Workflow::pass_mut`] can hand back a
/// concrete type.
#[doc(hidden)]
pub trait AsAny: 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// One step of a [`Workflow`]: the unit of render customization.
///
/// A pass records GPU work for one frame and owns whatever private resources
/// that takes. Passes may be stateless (zero-size structs) or stateful (holding
/// pipelines, buffers, bind groups); stateful ones implement
/// [`resize`](Self::resize) to rebuild size-dependent resources.
///
/// The type parameter `F` selects the [`FrameFamily`] — i.e. which per-frame
/// data type `execute` receives. Because `F` is a plain type parameter (the
/// lifetime lives on `F::Frame<'a>`, not on this trait), `dyn Pass<F>` is
/// `dyn`-compatible, so passes sharing a frame family compose in one list.
///
/// Passes do **not** own the attachments they share. A pass declares what it
/// needs from [`target_features`](Self::target_features) and the host allocates
/// it, which is how a producing pass hands a texture to a consuming one without
/// either holding the other.
pub trait Pass<F: FrameFamily>: AsAny {
    /// The attachments this pass reads or writes, beyond the final color
    /// target. The workflow keeps the union across all its passes and the host
    /// allocates from it; the default is none.
    fn target_features(&self) -> TargetFeatures {
        TargetFeatures::none()
    }

    /// Whether the pass should run this frame; inactive passes are skipped
    /// entirely. The default is always active.
    fn is_active(&self, _frame: &F::Frame<'_>) -> bool {
        true
    }

    /// Called after a viewport resize, and after the pass list changes. Passes
    /// that own size-dependent resources (textures, bind groups, or pipelines
    /// with baked sample counts) should recreate them here, reading the new
    /// size and sample count from `targets`. The default is a no-op.
    fn resize(&mut self, _gpu: &Gpu, _targets: &FrameTargets) {}

    /// Record this pass's GPU work into `encoder`, drawing to `view`.
    ///
    /// `targets` is passed separately rather than carried inside the frame so
    /// that the host can lend its own attachments while the caller retains an
    /// independently-built frame — the borrow split that lets `&mut host`
    /// coexist with frame data borrowed from the caller's other fields.
    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut F::Frame<'_>,
    );
}

/// A pass's name within its workflow, and the handle used to address it.
///
/// Deliberately a readable `&'static str` rather than a generated id: these
/// name positions in a hand-authored sequence, so they are written as
/// constants next to the passes they identify.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PassId(pub &'static str);

impl std::fmt::Display for PassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Why a workflow edit was rejected.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("no pass named '{0}' in this workflow")]
    UnknownPass(PassId),
    #[error("a pass named '{0}' is already in this workflow")]
    DuplicateId(PassId),
    #[error(transparent)]
    AuxConflict(#[from] AuxConflict),
}

/// A named, ordered, editable sequence of passes: the top-level unit of render
/// customization.
///
/// A [`RenderHost`](crate::RenderHost) runs exactly one workflow per frame.
/// Passes execute in order, each receiving the same frame, and each may skip
/// itself via [`Pass::is_active`].
///
/// Every pass is addressed by a [`PassId`], so a caller can adjust a stock
/// workflow instead of rebuilding one: insert a pass at a known point, swap one
/// out, drop one, or reach in and retune a pass in place with
/// [`pass_mut`](Self::pass_mut).
///
/// The workflow maintains the union of its passes' [`Pass::target_features`],
/// validating each edit as it happens. Because conflicts are caught at insert
/// time, the host can reallocate attachments afterwards without any possibility
/// of failure.
pub struct Workflow<F: FrameFamily> {
    name: String,
    passes: Vec<(PassId, Box<dyn Pass<F>>)>,
    /// Union of every pass's declarations, maintained incrementally.
    features: TargetFeatures,
    /// Bumped by every structural change, so a host can tell an edit that needs
    /// attachments reconciled from one that only retuned a pass in place.
    revision: u64,
}

impl<F: FrameFamily> Workflow<F> {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passes: Vec::new(),
            features: TargetFeatures::none(),
            revision: 0,
        }
    }

    /// Append a pass, for building a workflow in one expression.
    ///
    /// # Panics
    ///
    /// Panics if `id` is already present or the pass's target declarations
    /// conflict with an earlier pass's. Both are mistakes in a literal pass
    /// list, so they fail loudly at construction; use [`push`](Self::push) to
    /// handle them.
    #[must_use]
    pub fn with(mut self, id: PassId, pass: impl Pass<F>) -> Self {
        self.push(id, pass)
            .unwrap_or_else(|e| panic!("building workflow '{}': {e}", self.name));
        self
    }

    /// Append a pass to the end of the sequence.
    ///
    /// # Errors
    ///
    /// [`WorkflowError::DuplicateId`] if `id` is taken, or
    /// [`WorkflowError::AuxConflict`] if the pass declares an auxiliary target
    /// that another pass declared differently.
    pub fn push(&mut self, id: PassId, pass: impl Pass<F>) -> Result<(), WorkflowError> {
        self.insert_at(self.passes.len(), id, Box::new(pass))
    }

    /// Insert a pass immediately before `anchor`.
    ///
    /// # Errors
    ///
    /// As [`push`](Self::push), plus [`WorkflowError::UnknownPass`] if `anchor`
    /// is not in this workflow.
    pub fn insert_before(
        &mut self,
        anchor: PassId,
        id: PassId,
        pass: impl Pass<F>,
    ) -> Result<(), WorkflowError> {
        let at = self.index_of(anchor).ok_or(WorkflowError::UnknownPass(anchor))?;
        self.insert_at(at, id, Box::new(pass))
    }

    /// Insert a pass immediately after `anchor`.
    ///
    /// # Errors
    ///
    /// As [`insert_before`](Self::insert_before).
    pub fn insert_after(
        &mut self,
        anchor: PassId,
        id: PassId,
        pass: impl Pass<F>,
    ) -> Result<(), WorkflowError> {
        let at = self.index_of(anchor).ok_or(WorkflowError::UnknownPass(anchor))?;
        self.insert_at(at + 1, id, Box::new(pass))
    }

    /// Swap out the pass registered under `id`, keeping its position.
    ///
    /// # Errors
    ///
    /// [`WorkflowError::UnknownPass`] if `id` is not present, or
    /// [`WorkflowError::AuxConflict`] if the replacement's declarations
    /// conflict with the rest of the workflow. The workflow is unchanged on
    /// error.
    pub fn replace(&mut self, id: PassId, pass: impl Pass<F>) -> Result<(), WorkflowError> {
        let at = self.index_of(id).ok_or(WorkflowError::UnknownPass(id))?;
        let previous = std::mem::replace(&mut self.passes[at].1, Box::new(pass));
        match self.rebuild_features() {
            Ok(()) => {
                self.revision += 1;
                Ok(())
            }
            Err(e) => {
                self.passes[at].1 = previous;
                // Restoring cannot fail: this is the union that already held.
                let _ = self.rebuild_features();
                Err(e.into())
            }
        }
    }

    /// Remove and return the pass registered under `id`.
    pub fn remove(&mut self, id: PassId) -> Option<Box<dyn Pass<F>>> {
        let at = self.index_of(id)?;
        let (_, pass) = self.passes.remove(at);
        // A removal only relaxes requirements, so this cannot conflict.
        let _ = self.rebuild_features();
        self.revision += 1;
        Some(pass)
    }

    /// The pass registered under `id`, if it is of type `P`.
    #[must_use]
    pub fn pass<P: Pass<F>>(&self, id: PassId) -> Option<&P> {
        self.passes
            .iter()
            // Deref to the trait object deliberately: `AsAny` is blanket-
            // implemented, so calling it on the `Box` would downcast the box.
            .find(|(pid, _)| *pid == id)
            .and_then(|(_, pass)| (**pass).as_any().downcast_ref::<P>())
    }

    /// The pass registered under `id`, if it is of type `P`, for retuning it in
    /// place.
    ///
    /// This is how a live workflow is adjusted without rebuilding it — changing
    /// a color or a threshold costs no pipeline recompilation.
    #[must_use]
    pub fn pass_mut<P: Pass<F>>(&mut self, id: PassId) -> Option<&mut P> {
        self.passes
            .iter_mut()
            .find(|(pid, _)| *pid == id)
            .and_then(|(_, pass)| (**pass).as_any_mut().downcast_mut::<P>())
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The pass ids, in execution order.
    pub fn ids(&self) -> impl Iterator<Item = PassId> + '_ {
        self.passes.iter().map(|(id, _)| *id)
    }

    #[must_use]
    pub fn contains(&self, id: PassId) -> bool {
        self.index_of(id).is_some()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// The attachments this workflow needs: the union of its passes'
    /// declarations.
    #[must_use]
    pub const fn target_features(&self) -> &TargetFeatures {
        &self.features
    }

    /// Counts structural changes — pushes, inserts, replacements, removals.
    /// Retuning a pass through [`pass_mut`](Self::pass_mut) does not bump it.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    fn index_of(&self, id: PassId) -> Option<usize> {
        self.passes.iter().position(|(pid, _)| *pid == id)
    }

    fn insert_at(
        &mut self,
        at: usize,
        id: PassId,
        pass: Box<dyn Pass<F>>,
    ) -> Result<(), WorkflowError> {
        if self.contains(id) {
            return Err(WorkflowError::DuplicateId(id));
        }
        // Validate before inserting, so a rejected edit leaves nothing behind.
        self.features = self.features.clone().union(&pass.target_features())?;
        self.passes.insert(at, (id, pass));
        self.revision += 1;
        Ok(())
    }

    fn rebuild_features(&mut self) -> Result<(), AuxConflict> {
        let mut features = TargetFeatures::none();
        for (_, pass) in &self.passes {
            features = features.union(&pass.target_features())?;
        }
        self.features = features;
        Ok(())
    }

    pub(crate) fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        for (_, pass) in &mut self.passes {
            pass.resize(gpu, targets);
        }
    }

    pub(crate) fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut F::Frame<'_>,
    ) {
        for (_, pass) in &mut self.passes {
            if pass.is_active(frame) {
                pass.execute(gpu, targets, encoder, view, frame);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AuxKind, AuxTarget};

    enum TestFrames {}
    impl FrameFamily for TestFrames {
        type Frame<'a> = ();
    }

    /// A pass that records nothing; only its identity and declarations matter.
    struct Probe {
        features: TargetFeatures,
        tag: u32,
    }

    impl Probe {
        fn new(tag: u32) -> Self {
            Self { features: TargetFeatures::none(), tag }
        }

        fn with_features(tag: u32, features: TargetFeatures) -> Self {
            Self { features, tag }
        }
    }

    impl Pass<TestFrames> for Probe {
        fn target_features(&self) -> TargetFeatures {
            self.features.clone()
        }

        fn execute(
            &mut self,
            _gpu: &Gpu,
            _targets: &FrameTargets,
            _encoder: &mut wgpu::CommandEncoder,
            _view: &wgpu::TextureView,
            _frame: &mut (),
        ) {
        }
    }

    /// A second pass type, so downcasting has something to get wrong.
    struct Other;
    impl Pass<TestFrames> for Other {
        fn execute(
            &mut self,
            _gpu: &Gpu,
            _targets: &FrameTargets,
            _encoder: &mut wgpu::CommandEncoder,
            _view: &wgpu::TextureView,
            _frame: &mut (),
        ) {
        }
    }

    const A: PassId = PassId("a");
    const B: PassId = PassId("b");
    const C: PassId = PassId("c");

    fn mask_target() -> AuxTarget {
        AuxTarget::new("mask", AuxKind::Mask(crate::MaskChannels::Two)).with_resolved().with_sampled()
    }

    fn ids(workflow: &Workflow<TestFrames>) -> Vec<&'static str> {
        workflow.ids().map(|id| id.0).collect()
    }

    #[test]
    fn passes_execute_in_insertion_order() {
        let workflow = Workflow::<TestFrames>::new("test")
            .with(A, Probe::new(1))
            .with(C, Probe::new(3));
        assert_eq!(ids(&workflow), ["a", "c"]);
    }

    #[test]
    fn insert_before_and_after_place_relative_to_an_anchor() {
        let mut workflow = Workflow::<TestFrames>::new("test").with(A, Probe::new(1));

        workflow.insert_after(A, C, Probe::new(3)).unwrap();
        workflow.insert_before(C, B, Probe::new(2)).unwrap();

        assert_eq!(ids(&workflow), ["a", "b", "c"]);
    }

    #[test]
    fn inserting_against_a_missing_anchor_is_rejected() {
        let mut workflow = Workflow::<TestFrames>::new("test").with(A, Probe::new(1));

        let err = workflow.insert_after(B, C, Probe::new(3)).unwrap_err();

        assert!(matches!(err, WorkflowError::UnknownPass(id) if id == B));
        assert_eq!(ids(&workflow), ["a"]);
    }

    #[test]
    fn a_duplicate_id_is_rejected_and_changes_nothing() {
        let mut workflow = Workflow::<TestFrames>::new("test").with(A, Probe::new(1));
        let before = workflow.revision();

        let err = workflow.push(A, Probe::new(2)).unwrap_err();

        assert!(matches!(err, WorkflowError::DuplicateId(id) if id == A));
        assert_eq!(ids(&workflow), ["a"]);
        assert_eq!(workflow.revision(), before);
    }

    #[test]
    fn replace_keeps_position_and_remove_returns_the_pass() {
        let mut workflow = Workflow::<TestFrames>::new("test")
            .with(A, Probe::new(1))
            .with(B, Probe::new(2))
            .with(C, Probe::new(3));

        workflow.replace(B, Probe::new(20)).unwrap();
        assert_eq!(ids(&workflow), ["a", "b", "c"]);
        assert_eq!(workflow.pass::<Probe>(B).unwrap().tag, 20);

        assert!(workflow.remove(B).is_some());
        assert_eq!(ids(&workflow), ["a", "c"]);
        assert!(workflow.remove(B).is_none());
    }

    #[test]
    fn pass_mut_downcasts_to_the_concrete_type() {
        let mut workflow = Workflow::<TestFrames>::new("test")
            .with(A, Probe::new(1))
            .with(B, Other);

        workflow.pass_mut::<Probe>(A).unwrap().tag = 7;

        assert_eq!(workflow.pass::<Probe>(A).unwrap().tag, 7);
        // Right id, wrong type.
        assert!(workflow.pass_mut::<Probe>(B).is_none());
        // Right type, absent id.
        assert!(workflow.pass_mut::<Probe>(C).is_none());
    }

    #[test]
    fn retuning_a_pass_does_not_bump_the_revision() {
        let mut workflow = Workflow::<TestFrames>::new("test").with(A, Probe::new(1));
        let before = workflow.revision();

        workflow.pass_mut::<Probe>(A).unwrap().tag = 7;
        assert_eq!(workflow.revision(), before);

        workflow.push(B, Probe::new(2)).unwrap();
        assert!(workflow.revision() > before);
    }

    #[test]
    fn target_features_are_the_union_of_the_passes() {
        let workflow = Workflow::<TestFrames>::new("test")
            .with(A, Probe::with_features(1, TargetFeatures::depth()))
            .with(
                B,
                Probe::with_features(
                    2,
                    TargetFeatures::none().with_sampled_depth().with_aux(mask_target()),
                ),
            )
            // Declaring the same target again is how a consumer names its input.
            .with(C, Probe::with_features(3, TargetFeatures::none().with_aux(mask_target())));

        let features = workflow.target_features();
        assert!(features.needs_depth());
        assert!(features.needs_sampled_depth());
        assert_eq!(features.aux_targets().len(), 1);
    }

    #[test]
    fn conflicting_aux_declarations_are_rejected_at_insert_time() {
        let mut workflow = Workflow::<TestFrames>::new("test").with(
            A,
            Probe::with_features(1, TargetFeatures::none().with_aux(mask_target())),
        );
        let before = workflow.revision();

        // Same name, different description.
        let clashing = AuxTarget::new("mask", AuxKind::Mask(crate::MaskChannels::One));
        let err = workflow
            .push(B, Probe::with_features(2, TargetFeatures::none().with_aux(clashing)))
            .unwrap_err();

        assert!(matches!(err, WorkflowError::AuxConflict(_)));
        assert_eq!(ids(&workflow), ["a"]);
        assert_eq!(workflow.revision(), before);
        assert_eq!(workflow.target_features().aux_targets(), [mask_target()]);
    }

    #[test]
    fn a_rejected_replacement_restores_the_original_pass() {
        let mut workflow = Workflow::<TestFrames>::new("test")
            .with(A, Probe::with_features(1, TargetFeatures::none().with_aux(mask_target())))
            .with(B, Probe::with_features(2, TargetFeatures::none().with_aux(mask_target())));

        let clashing = AuxTarget::new("mask", AuxKind::Depth);
        let err = workflow
            .replace(B, Probe::with_features(20, TargetFeatures::none().with_aux(clashing)))
            .unwrap_err();

        assert!(matches!(err, WorkflowError::AuxConflict(_)));
        assert_eq!(workflow.pass::<Probe>(B).unwrap().tag, 2);
        assert_eq!(workflow.target_features().aux_targets(), [mask_target()]);
    }

    #[test]
    fn removing_a_pass_relaxes_the_target_union() {
        let mut workflow = Workflow::<TestFrames>::new("test")
            .with(A, Probe::with_features(1, TargetFeatures::depth()))
            .with(B, Probe::with_features(2, TargetFeatures::none().with_aux(mask_target())));

        assert_eq!(workflow.target_features().aux_targets().len(), 1);

        workflow.remove(B);

        assert!(workflow.target_features().needs_depth());
        assert!(workflow.target_features().aux_targets().is_empty());
    }
}
