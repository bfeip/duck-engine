//! Shared scene handle.
//!
//! [`Scene`] is a cheap, cloneable handle to a [`SceneData`]. Most
//! access goes through the handle's own methods, which acquire and release
//! locks internally, so callers never hold a guard. Compound work that needs a
//! stable `&SceneData` — rendering a frame, importing a file — takes one through
//! [`Scene::lock`].

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};
#[cfg(debug_assertions)]
use std::thread::ThreadId;

use crate::common::{Matrix4, Point3, Quaternion, Transform, Vector3};
use crate::{BoundingResult, EnvironmentMap, EnvironmentMapId, Light, PositionedCamera, SceneData};
use crate::resource::{
    DisplayBehavior, EffectiveVisibility, FaceMaterial, FaceMaterialHandle, FaceMaterialId,
    Instance, InstanceHandle, InstanceId, LineMaterial, LineMaterialHandle, LineMaterialId, Mesh,
    MeshHandle, Node, NodeFlags, NodeHandle, NodeId, NodePayload, PointMaterial,
    PointMaterialHandle, PointMaterialId, Visibility,
};

/// A cheap, cloneable handle to a scene.
///
/// Cloning shares the underlying [`SceneData`], it does not copy it. (Deep
/// copies are still available through `SceneData: Clone`;
/// [`ptr_eq`](Self::ptr_eq) tells shared clones apart from distinct scenes.)
///
/// There are two ways to use one:
///
/// - The handle's own methods ([`add_mesh`](Self::add_mesh),
///   [`set_node_transform`](Self::set_node_transform), ...), each of which locks
///   internally and forms a complete critical section.
/// - [`lock`](Self::lock), which borrows the [`SceneData`] itself for compound
///   work.
///
/// # Deadlocks
///
/// The lock is not reentrant. Handle methods are complete critical sections and
/// compose freely, but **never call a `Scene` method while holding a guard from
/// [`lock`](Self::lock)**. In debug builds that mistake panics with both source
/// locations rather than hanging.
#[derive(Clone)]
pub struct Scene(pub(crate) Arc<SceneLock>);

impl Scene {
    /// Wraps scene data in a new handle.
    ///
    /// Every handle already minted from the data (during import, say) becomes
    /// bound to this scene.
    pub fn new(data: SceneData) -> Self {
        let bind = data.bind().clone();
        let lock = Arc::new(SceneLock::new(data));
        *bind.lock.lock().unwrap_or_else(|e| e.into_inner()) = Arc::downgrade(&lock);
        Self(lock)
    }

    /// Borrows the scene data for a compound operation.
    ///
    /// Prefer the handle's own methods, which lock internally. Reach for a guard
    /// when a single `&SceneData` must stay valid across many reads — rendering a
    /// frame, walking the tree, populating the scene during import.
    ///
    /// The returned guard derefs both shared and mutably. It must not be held
    /// across a call back into this handle, or across anything that might call
    /// back into it (a render pass, an event callback, a UI closure).
    #[track_caller]
    pub fn lock(&self) -> SceneGuard<'_> {
        self.0.lock()
    }

    /// True if both handles refer to the same underlying scene.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Operations that lock internally.
///
/// These are whole operations rather than field accessors: anything that reads
/// and then writes does both inside one critical section, so no other holder can
/// observe an intermediate state. Reads that would otherwise hand out a borrow
/// into the scene return owned clones instead. `Mesh` is deliberately absent —
/// it owns its vertex and index buffers, so mesh access goes through
/// [`lock`](Self::lock).
impl Scene {
    // ========== Camera ==========

    /// Builds a [`PositionedCamera`] for the active camera node at `aspect`.
    #[track_caller]
    pub fn camera(&self, aspect: f32) -> Option<PositionedCamera> {
        self.lock().active_camera_positioned(aspect)
    }

    /// Writes a camera back to the active camera node (pose and projection).
    ///
    /// Does nothing if the scene has no active camera.
    #[track_caller]
    pub fn set_camera(&self, camera: PositionedCamera) {
        let mut scene = self.lock();
        let Some(id) = scene.active_camera() else { return };
        scene.set_node_transform(id, camera.to_node_transform());
        scene.set_node_payload(id, NodePayload::Camera(camera.projection()));
    }

    /// Reads the active camera, passes it to `f`, and writes it back.
    ///
    /// The read and the write share one critical section, so the camera `f`
    /// mutates is the one that gets stored. `f` must not touch this handle.
    #[track_caller]
    pub fn with_camera_mut(&self, aspect: f32, f: impl FnOnce(&mut PositionedCamera)) {
        let mut scene = self.lock();
        let Some(id) = scene.active_camera() else { return };
        let Some(mut camera) = scene.active_camera_positioned(aspect) else { return };
        f(&mut camera);
        scene.set_node_transform(id, camera.to_node_transform());
        scene.set_node_payload(id, NodePayload::Camera(camera.projection()));
    }

    /// The node driving rendering when no explicit camera is passed.
    #[track_caller]
    pub fn active_camera(&self) -> Option<NodeId> {
        self.lock().active_camera()
    }

    /// Sets the active camera node. Pass `None` to clear.
    #[track_caller]
    pub fn set_active_camera(&self, node_id: Option<NodeId>) {
        self.lock().set_active_camera(node_id);
    }

    // ========== Node queries ==========

    /// Clones the node with the given id.
    #[track_caller]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.lock().get_node(id).cloned()
    }

    /// True if the scene contains a node with this id.
    #[track_caller]
    pub fn has_node(&self, id: NodeId) -> bool {
        self.lock().has_node(id)
    }

    /// Number of nodes in the scene.
    #[track_caller]
    pub fn node_count(&self) -> usize {
        self.lock().node_count()
    }

    /// The scene-tree generation; see [`SceneData::node_generation`].
    #[track_caller]
    pub fn node_generation(&self) -> u64 {
        self.lock().node_generation()
    }

    /// Copies the root node ids out of the scene.
    #[track_caller]
    pub fn root_nodes(&self) -> Vec<NodeId> {
        self.lock().root_nodes().collect()
    }

    /// World transform of a node, or `None` if the tree is incomplete.
    #[track_caller]
    pub fn nodes_transform(&self, node_id: NodeId) -> Option<Matrix4> {
        self.lock().nodes_transform(node_id)
    }

    /// World-space bounds of a node's subtree.
    #[track_caller]
    pub fn nodes_bounding(&self, node_id: NodeId) -> BoundingResult {
        self.lock().nodes_bounding(node_id)
    }

    /// World-space bounds of the whole scene.
    #[track_caller]
    pub fn bounding(&self) -> BoundingResult {
        self.lock().bounding()
    }

    /// Visibility a node actually renders with, after inheritance.
    #[track_caller]
    pub fn node_effective_visibility(&self, node_id: NodeId) -> EffectiveVisibility {
        self.lock().node_effective_visibility(node_id)
    }

    // ========== Node mutation ==========

    /// Adds a node under `parent` (`None` for a root) and returns its owning
    /// handle.
    #[track_caller]
    pub fn add_node(
        &self,
        parent: Option<NodeId>,
        name: Option<String>,
        transform: Transform,
        flags: NodeFlags,
    ) -> anyhow::Result<NodeHandle> {
        self.lock().add_node(parent, name, transform, flags)
    }

    /// Adds a node carrying `instance` and returns its owning handle.
    #[track_caller]
    pub fn add_instance_node(
        &self,
        parent: Option<NodeId>,
        instance: Instance,
        name: Option<String>,
        transform: Transform,
        flags: NodeFlags,
    ) -> anyhow::Result<NodeHandle> {
        self.lock().add_instance_node(parent, instance, name, transform, flags)
    }

    /// An owning handle for an existing node.
    #[track_caller]
    pub fn node_handle(&self, id: NodeId) -> Option<NodeHandle> {
        self.lock().node_handle(id)
    }

    /// Sets what a node is (instance, camera, light, …).
    #[track_caller]
    pub fn set_node_payload(&self, node_id: NodeId, payload: NodePayload) {
        self.lock().set_node_payload(node_id, payload);
    }

    /// Sets a node's local transform.
    #[track_caller]
    pub fn set_node_transform(&self, node_id: NodeId, transform: Transform) {
        self.lock().set_node_transform(node_id, transform);
    }

    /// Sets the translation part of a node's local transform.
    #[track_caller]
    pub fn set_node_position(&self, node_id: NodeId, position: Point3) {
        self.lock().set_node_position(node_id, position);
    }

    /// Sets the rotation part of a node's local transform.
    #[track_caller]
    pub fn set_node_rotation(&self, node_id: NodeId, rotation: Quaternion) {
        self.lock().set_node_rotation(node_id, rotation);
    }

    /// Sets the scale part of a node's local transform.
    #[track_caller]
    pub fn set_node_scale(&self, node_id: NodeId, scale: Vector3) {
        self.lock().set_node_scale(node_id, scale);
    }

    /// Sets a node's display behavior (screen-space rendering, layer).
    #[track_caller]
    pub fn set_node_display(&self, node_id: NodeId, display: DisplayBehavior) {
        self.lock().set_node_display(node_id, display);
    }

    /// Sets a node's authored visibility.
    #[track_caller]
    pub fn set_node_visibility(&self, node_id: NodeId, visibility: Visibility) {
        self.lock().set_node_visibility(node_id, visibility);
    }

    /// Detaches a node's subtree; see [`SceneData::remove_node`].
    #[track_caller]
    pub fn remove_node(&self, node_id: NodeId) {
        self.lock().remove_node(node_id);
    }

    /// Moves a node under a new parent; see [`SceneData::reparent_node`].
    #[track_caller]
    pub fn reparent_node(&self, node_id: NodeId, new_parent: Option<NodeId>) -> anyhow::Result<()> {
        self.lock().reparent_node(node_id, new_parent)
    }

    /// Empties the scene.
    #[track_caller]
    pub fn clear(&self) {
        self.lock().clear();
    }

    // ========== Instances and materials ==========

    /// Adds an instance and returns its owning handle.
    #[track_caller]
    pub fn add_instance(&self, instance: Instance) -> InstanceHandle {
        self.lock().add_instance(instance)
    }

    /// Clones the instance with the given id.
    #[track_caller]
    pub fn get_instance(&self, id: InstanceId) -> Option<Instance> {
        self.lock().get_instance(id).cloned()
    }

    /// Adds a mesh and returns its owning handle.
    #[track_caller]
    pub fn add_mesh(&self, mesh: Mesh) -> MeshHandle {
        self.lock().add_mesh(mesh)
    }

    /// Adds a face material and returns its owning handle.
    #[track_caller]
    pub fn add_face_material(&self, material: FaceMaterial) -> FaceMaterialHandle {
        self.lock().add_face_material(material)
    }

    /// Clones the face material with the given id.
    #[track_caller]
    pub fn get_face_material(&self, id: FaceMaterialId) -> Option<FaceMaterial> {
        self.lock().get_face_material(id).cloned()
    }

    /// Adds a line material and returns its owning handle.
    #[track_caller]
    pub fn add_line_material(&self, material: LineMaterial) -> LineMaterialHandle {
        self.lock().add_line_material(material)
    }

    /// Clones the line material with the given id.
    #[track_caller]
    pub fn get_line_material(&self, id: LineMaterialId) -> Option<LineMaterial> {
        self.lock().get_line_material(id).cloned()
    }

    /// Adds a point material and returns its owning handle.
    #[track_caller]
    pub fn add_point_material(&self, material: PointMaterial) -> PointMaterialHandle {
        self.lock().add_point_material(material)
    }

    /// Clones the point material with the given id.
    #[track_caller]
    pub fn get_point_material(&self, id: PointMaterialId) -> Option<PointMaterial> {
        self.lock().get_point_material(id).cloned()
    }

    // ========== Counts ==========

    /// Number of meshes in the scene.
    #[track_caller]
    pub fn mesh_count(&self) -> usize {
        self.lock().mesh_count()
    }

    /// Number of instances in the scene.
    #[track_caller]
    pub fn instance_count(&self) -> usize {
        self.lock().instance_count()
    }

    /// Number of light nodes in the scene.
    #[track_caller]
    pub fn light_count(&self) -> usize {
        self.lock().light_count()
    }

    /// Collects every light in the scene with the attached node carrying it.
    #[track_caller]
    pub fn light_nodes(&self) -> Vec<(NodeId, Light)> {
        let scene = self.lock();
        scene
            .nodes()
            .filter_map(|n| match n.payload() {
                NodePayload::Light(light) if scene.is_node_attached(n.id) => {
                    Some((n.id, light.clone()))
                }
                _ => None,
            })
            .collect()
    }

    // ========== Environment maps ==========

    /// The environment map currently used for IBL, if any.
    #[track_caller]
    pub fn active_environment_map(&self) -> Option<EnvironmentMapId> {
        self.lock().active_environment_map()
    }

    /// Clones the environment map with the given id.
    #[track_caller]
    pub fn get_environment_map(&self, id: EnvironmentMapId) -> Option<EnvironmentMap> {
        self.lock().get_environment_map(id).cloned()
    }

    /// Sets the environment map used for IBL. Pass `None` to disable.
    #[track_caller]
    pub fn set_active_environment_map(&self, id: Option<EnvironmentMapId>) {
        self.lock().set_active_environment_map(id);
    }

    /// Sets an environment map's IBL intensity. No-op if the id is unknown.
    #[track_caller]
    pub fn set_environment_map_intensity(&self, id: EnvironmentMapId, intensity: f32) {
        if let Some(env) = self.lock().get_environment_map_mut(id) {
            env.set_intensity(intensity);
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new(SceneData::new())
    }
}

impl From<SceneData> for Scene {
    fn from(data: SceneData) -> Self {
        Self::new(data)
    }
}

/// The mutex behind a [`Scene`], with poison handling and debug re-entrancy
/// detection folded in.
pub(crate) struct SceneLock {
    data: Mutex<SceneData>,
    /// Thread and source location of the guard currently held, for the debug
    /// re-entrancy panic.
    #[cfg(debug_assertions)]
    held_by: Mutex<Option<(ThreadId, &'static std::panic::Location<'static>)>>,
}

impl SceneLock {
    fn new(data: SceneData) -> Self {
        Self {
            data: Mutex::new(data),
            #[cfg(debug_assertions)]
            held_by: Mutex::new(None),
        }
    }

    /// Acquires the lock, ignoring poisoning.
    ///
    /// A panic while the scene is locked can leave it in a partially-updated
    /// state, but the scene is plain data with no cross-field invariants the
    /// renderer cannot tolerate, and refusing to unlock would turn any panic into
    /// a cascade of secondary panics in every later frame.
    #[track_caller]
    pub(crate) fn lock(&self) -> SceneGuard<'_> {
        #[cfg(debug_assertions)]
        let mut guard = {
            let caller = std::panic::Location::caller();
            let guard = match self.data.try_lock() {
                Ok(guard) => guard,
                Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    // Contention from the thread that already holds the guard is
                    // re-entrancy, and would hang. Contention from another thread
                    // is ordinary and must block, so check who holds it.
                    let held = self.held_by.lock().map_or(None, |h| *h);
                    match held {
                        Some((thread, at)) if thread == std::thread::current().id() => panic!(
                            "scene lock is not reentrant: {caller} tried to lock a scene this \
                             thread already locked at {at}. Drop the guard before calling back \
                             into the handle."
                        ),
                        _ => self.data.lock().unwrap_or_else(|e| e.into_inner()),
                    }
                }
            };
            if let Ok(mut held) = self.held_by.lock() {
                *held = Some((std::thread::current().id(), caller));
            }
            SceneGuard { inner: guard, lock: self }
        };

        #[cfg(not(debug_assertions))]
        let mut guard = SceneGuard {
            inner: self.data.lock().unwrap_or_else(|e| e.into_inner()),
        };

        // Every guard observes a fully-reaped scene.
        guard.reap();
        guard
    }
}

/// Borrowed scene data. See [`Scene::lock`].
///
/// Derefs to [`SceneData`], both shared and mutably. Resources whose last
/// handle has dropped are removed when the guard is acquired and again when it
/// drops, so a guard never observes them.
pub struct SceneGuard<'a> {
    inner: MutexGuard<'a, SceneData>,
    #[cfg(debug_assertions)]
    lock: &'a SceneLock,
}

impl Deref for SceneGuard<'_> {
    type Target = SceneData;

    fn deref(&self) -> &SceneData {
        &self.inner
    }
}

impl DerefMut for SceneGuard<'_> {
    fn deref_mut(&mut self) -> &mut SceneData {
        &mut self.inner
    }
}

impl Drop for SceneGuard<'_> {
    fn drop(&mut self) {
        // Handles dropped while this guard was held are reaped before the
        // scene unlocks.
        self.inner.reap();
        #[cfg(debug_assertions)]
        if let Ok(mut held) = self.lock.held_by.lock() {
            *held = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Transform;
    use crate::resource::NodeFlags;

    #[test]
    fn clone_shares_the_same_scene() {
        let scene = Scene::default();
        let other = scene.clone();

        let id = scene
            .lock()
            .add_node(None, Some("n".into()), Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();

        assert!(other.lock().has_node(id));
        assert!(scene.ptr_eq(&other));
    }

    #[test]
    fn distinct_scenes_do_not_share() {
        let a = Scene::default();
        let b = Scene::default();

        a.lock()
            .add_node(None, Some("n".into()), Transform::IDENTITY, NodeFlags::NONE)
            .unwrap();

        assert_eq!(b.lock().node_count(), 0);
        assert!(!a.ptr_eq(&b));
    }

    #[test]
    fn sequential_locks_do_not_deadlock() {
        let scene = Scene::default();
        for _ in 0..3 {
            assert_eq!(scene.lock().node_count(), 0);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not reentrant")]
    fn reentrant_lock_panics_instead_of_hanging() {
        let scene = Scene::default();
        let _held = scene.lock();
        let _second = scene.lock();
    }

    #[test]
    fn contention_from_another_thread_blocks_rather_than_panicking() {
        use std::sync::mpsc;

        let scene = Scene::default();
        let other = scene.clone();
        let (tx, rx) = mpsc::channel();

        // Hold the guard while a second thread asks for it. That thread must
        // block and then succeed — only same-thread contention is re-entrancy.
        let held = scene.lock();
        let worker = std::thread::spawn(move || {
            tx.send(()).unwrap();
            other.node_count()
        });
        rx.recv().unwrap();
        // Best-effort: give the worker time to actually reach the lock. If it
        // doesn't, the test still passes — it just stops proving anything.
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(held);

        assert_eq!(worker.join().unwrap(), 0);
    }
}
