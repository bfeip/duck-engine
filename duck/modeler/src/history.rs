use std::collections::VecDeque;

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::NodeHandle;
use opencascade::primitives::Shape;

use crate::document::PartId;

/// Maximum number of undoable steps retained; the oldest is evicted beyond this.
const MAX_STEPS: usize = 64;

/// One user-level operation: the deltas recorded while it ran, undone/redone as a unit.
pub struct Step {
    pub label: String,
    pub deltas: Vec<Delta>,
}

/// A single document mutation, captured with enough state to replay in either direction.
pub enum Delta {
    /// A part was added. Undo removes it; redo resurrects the snapshot.
    Added(PartSnapshot),
    /// A part was removed. Undo resurrects the snapshot; redo removes it.
    Removed(PartSnapshot),
    /// A part's shape was replaced in place, its node preserved.
    Reshaped {
        part: PartId,
        before: Shape,
        after: Shape,
        options: CadTessellationOptions,
    },
}

/// Everything needed to remove or resurrect a part with stable ids.
///
/// The node handle owns the part's whole scene chain (node → instance → mesh
/// and materials): removal only detaches the subtree, and this snapshot keeps
/// it alive off-tree until resurrected or evicted from history. Shapes are
/// shallow clones: committed shapes are never mutated in place, so a refcounted
/// handle stays an exact snapshot. The options carry the retessellation
/// parameters, which `remove_part` does not otherwise receive.
pub struct PartSnapshot {
    pub part: PartId,
    pub node: NodeHandle,
    pub name: String,
    pub shape: Shape,
    pub options: CadTessellationOptions,
}

/// Undo/redo stacks plus the in-progress grouping state.
#[derive(Default)]
pub struct History {
    undo: VecDeque<Step>,
    redo: Vec<Step>,
    pending: Option<Step>,
    depth: usize,
}

impl History {
    /// Opens a grouping scope. Nested scopes merge into the outermost one,
    /// whose label wins.
    pub fn begin_group(&mut self, label: impl Into<String>) {
        if self.depth == 0 {
            self.pending = Some(Step { label: label.into(), deltas: Vec::new() });
        }
        self.depth += 1;
    }

    /// Closes a grouping scope; the outermost close commits the pending step.
    pub fn end_group(&mut self) {
        self.depth = self.depth.saturating_sub(1);
        if self.depth > 0 {
            return;
        }
        if let Some(step) = self.pending.take().filter(|s| !s.deltas.is_empty()) {
            self.commit(step);
        }
    }

    /// Records a delta into the open scope, or — with `label` — as a step of its own.
    pub fn record(&mut self, label: &str, delta: Delta) {
        match &mut self.pending {
            Some(step) => step.deltas.push(delta),
            None => self.commit(Step { label: label.to_string(), deltas: vec![delta] }),
        }
    }

    /// Pushes a newly committed step: invalidates redo, evicts beyond the cap.
    fn commit(&mut self, step: Step) {
        self.undo.push_back(step);
        self.redo.clear();
        if self.undo.len() > MAX_STEPS {
            self.undo.pop_front();
        }
    }

    pub fn pop_undo(&mut self) -> Option<Step> {
        self.undo.pop_back()
    }

    pub fn pop_redo(&mut self) -> Option<Step> {
        self.redo.pop()
    }

    /// Parks an undone step on the redo stack.
    pub fn push_redo(&mut self, step: Step) {
        self.redo.push(step);
    }

    /// Returns a redone step to the undo stack (unlike [`record`](Self::record),
    /// this must not invalidate redo).
    pub fn restore_undo(&mut self, step: Step) {
        self.undo.push_back(step);
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo.back().map(|s| s.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|s| s.label.as_str())
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.pending = None;
        self.depth = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(name: &str) -> PartSnapshot {
        PartSnapshot {
            part: PartId::new(),
            node: NodeHandle::unbound(duck_engine_scene::NodeId::new()),
            name: name.to_string(),
            shape: Shape::cube(1.0),
            options: CadTessellationOptions::default(),
        }
    }

    #[test]
    fn record_without_scope_makes_single_delta_steps() {
        let mut history = History::default();
        history.record("Add a", Delta::Added(snapshot("a")));
        history.record("Add b", Delta::Added(snapshot("b")));
        assert_eq!(history.undo_label(), Some("Add b"));
        assert_eq!(history.pop_undo().unwrap().deltas.len(), 1);
        assert_eq!(history.undo_label(), Some("Add a"));
    }

    #[test]
    fn group_collects_deltas_into_one_step() {
        let mut history = History::default();
        history.begin_group("Boolean");
        history.record("Add result", Delta::Added(snapshot("result")));
        history.record("Delete tool", Delta::Removed(snapshot("tool")));
        history.end_group();
        assert_eq!(history.undo_label(), Some("Boolean"));
        let step = history.pop_undo().unwrap();
        assert_eq!(step.deltas.len(), 2);
        assert!(history.pop_undo().is_none());
    }

    #[test]
    fn nested_groups_merge_into_outermost() {
        let mut history = History::default();
        history.begin_group("Outer");
        history.record("x", Delta::Added(snapshot("a")));
        history.begin_group("Inner");
        history.record("x", Delta::Added(snapshot("b")));
        history.end_group();
        history.record("x", Delta::Added(snapshot("c")));
        history.end_group();
        assert_eq!(history.undo_label(), Some("Outer"));
        assert_eq!(history.pop_undo().unwrap().deltas.len(), 3);
    }

    #[test]
    fn empty_group_pushes_no_step() {
        let mut history = History::default();
        history.begin_group("Nothing");
        history.end_group();
        assert!(history.undo_label().is_none());
    }

    #[test]
    fn commit_invalidates_redo_but_restore_does_not() {
        let mut history = History::default();
        history.record("Add a", Delta::Added(snapshot("a")));
        let step = history.pop_undo().unwrap();
        history.push_redo(step);
        assert_eq!(history.redo_label(), Some("Add a"));

        let step = history.pop_redo().unwrap();
        history.restore_undo(step);
        assert_eq!(history.undo_label(), Some("Add a"));
        history.record("Add b", Delta::Added(snapshot("b")));
        assert!(history.redo_label().is_none(), "a new commit must clear redo");
    }

    #[test]
    fn cap_evicts_oldest_step() {
        let mut history = History::default();
        for i in 0..(MAX_STEPS + 2) {
            history.record(&format!("Add {i}"), Delta::Added(snapshot("p")));
        }
        assert_eq!(history.undo_label(), Some(format!("Add {}", MAX_STEPS + 1).as_str()));
        let mut count = 0;
        let mut oldest = None;
        while let Some(step) = history.pop_undo() {
            oldest = Some(step.label);
            count += 1;
        }
        assert_eq!(count, MAX_STEPS);
        assert_eq!(oldest.as_deref(), Some("Add 2"));
    }
}
