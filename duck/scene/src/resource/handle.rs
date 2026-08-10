//! Refcounted owning handles for scene resources.
//!
//! A [`Handle`] keeps its resource alive: the scene's own references (an
//! instance's mesh, a node's children) are handles too, so a resource is
//! removed exactly when the last handle drops. Removal is deferred — a drop
//! enqueues the id on the scene's [`SceneBind`] queue, and the queue is drained
//! (with cascade) whenever the scene lock is taken or released. Handle clone
//! and drop never touch the scene lock, so handles may be dropped freely while
//! a [`SceneGuard`](crate::SceneGuard) is held.

use crate::scene_handle::{Scene, SceneLock};
use super::id::{GenericId, Id};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, Weak};

/// Which scene collection a resource lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Mesh,
    Instance,
    Node,
    FaceMaterial,
    LineMaterial,
    PointMaterial,
    Texture,
}

// `SceneResource` must be public (it appears in `Handle` bounds), but only the
// scene's own resource types may implement it — handles to arbitrary types are
// meaningless. Requiring this crate-private parent trait makes implementations
// outside this crate impossible: external code can name `SceneResource` but
// not `Sealed`.
mod sealed {
    pub trait Sealed {}
}

/// A resource type that lives in a scene collection and can be held by a
/// [`Handle`].
///
/// Sealed: implemented exactly for the seven scene resource types below,
/// never externally.
pub trait SceneResource: sealed::Sealed {
    /// The collection this resource lives in.
    const KIND: ResourceKind;
}

macro_rules! impl_scene_resource {
    ($($ty:ty => $kind:ident),* $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl SceneResource for $ty {
            const KIND: ResourceKind = ResourceKind::$kind;
        }
    )*};
}

impl_scene_resource! {
    super::Mesh => Mesh,
    super::Instance => Instance,
    super::Node => Node,
    super::FaceMaterial => FaceMaterial,
    super::LineMaterial => LineMaterial,
    super::PointMaterial => PointMaterial,
    super::Texture => Texture,
}

/// State shared between a `SceneData` and every handle minted from it.
///
/// Pushing onto `dropped` never touches the scene lock, which is what makes
/// handle drops safe under a held guard.
pub(crate) struct SceneBind {
    /// (kind, id) pairs whose last strong handle dropped, awaiting reap.
    pub(crate) dropped: Mutex<Vec<(ResourceKind, GenericId)>>,
    /// The lock wrapping the owning `SceneData`, set by `Scene::new`.
    /// Empty for a standalone `SceneData`.
    pub(crate) lock: Mutex<Weak<SceneLock>>,
}

impl SceneBind {
    pub(crate) fn new() -> Self {
        Self {
            dropped: Mutex::new(Vec::new()),
            lock: Mutex::new(Weak::new()),
        }
    }

    fn scene_lock(&self) -> Option<Arc<SceneLock>> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner()).upgrade()
    }
}

/// One per live resource; the refcount is the `Arc` strong count.
pub(crate) struct HandleCore {
    pub(crate) id: GenericId,
    pub(crate) kind: ResourceKind,
    pub(crate) bind: Weak<SceneBind>,
}

impl Drop for HandleCore {
    // Runs when the last strong handle drops: enqueue for reap.
    fn drop(&mut self) {
        if let Some(bind) = self.bind.upgrade() {
            bind.dropped
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((self.kind, self.id));
        }
    }
}

/// An owning reference to a scene resource.
///
/// Cheap to clone; `Send + Sync`. The resource stays in the scene while any
/// handle for it exists and is removed (with everything it exclusively owns)
/// once the last one drops. Equality and hashing are by resource id.
///
/// # Ownership
///
/// The scene's own references are handles too, so ownership chains:
///
/// - an [`Instance`](super::Instance) owns its mesh and material slots,
/// - a material owns its texture slots,
/// - a [`Node`](super::Node) owns its payload instance and its children,
/// - the scene's root list owns the root nodes.
///
/// Releasing a link releases everything reachable only through it: detaching a
/// node ([`SceneData::remove_node`](crate::SceneData::remove_node)) with no
/// outstanding handles removes its subtree, the instances those nodes carried,
/// and any meshes, materials, and textures nothing else shares. Parent links
/// are plain ids, not handles, so the child→parent direction owns nothing and
/// no cycles form.
pub struct Handle<Kind> {
    core: Arc<HandleCore>,
    _kind: PhantomData<fn() -> Kind>,
}

/// An owning reference to a [`Mesh`](super::Mesh).
pub type MeshHandle = Handle<super::Mesh>;
/// An owning reference to an [`Instance`](super::Instance).
pub type InstanceHandle = Handle<super::Instance>;
/// An owning reference to a [`Node`](super::Node).
pub type NodeHandle = Handle<super::Node>;
/// An owning reference to a [`FaceMaterial`](super::FaceMaterial).
pub type FaceMaterialHandle = Handle<super::FaceMaterial>;
/// An owning reference to a [`LineMaterial`](super::LineMaterial).
pub type LineMaterialHandle = Handle<super::LineMaterial>;
/// An owning reference to a [`PointMaterial`](super::PointMaterial).
pub type PointMaterialHandle = Handle<super::PointMaterial>;
/// An owning reference to a [`Texture`](super::Texture).
pub type TextureHandle = Handle<super::Texture>;

impl<K> Handle<K> {
    pub(crate) fn from_core(core: Arc<HandleCore>) -> Self {
        Self { core, _kind: PhantomData }
    }

    /// The resource's id. Ids are `Copy` and suit map keys; unlike the handle
    /// they carry no ownership.
    pub fn id(&self) -> Id<K> {
        self.core.id.cast()
    }

    /// The scene this handle is bound to, if it has one and the scene is still
    /// alive. Handles minted from a standalone `SceneData` become bound when
    /// the data is wrapped in a [`Scene`].
    pub fn scene(&self) -> Option<Scene> {
        let bind = self.core.bind.upgrade()?;
        Some(Scene(bind.scene_lock()?))
    }

}

impl<K: SceneResource> Handle<K> {
    /// A handle not connected to any scene.
    ///
    /// Carries only the id: it keeps nothing alive and its accessors are inert.
    /// Used when assembling resources outside a scene (deserialization);
    /// inserting the data into a scene replaces unbound handles with live ones.
    pub fn unbound(id: Id<K>) -> Self {
        Self::from_core(Arc::new(HandleCore {
            id: id.erased(),
            kind: K::KIND,
            bind: Weak::new(),
        }))
    }
}

impl<K> Clone for Handle<K> {
    fn clone(&self) -> Self {
        Self { core: self.core.clone(), _kind: PhantomData }
    }
}

impl<K> std::fmt::Debug for Handle<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Handle").field(&self.core.id).finish()
    }
}

impl<K> PartialEq for Handle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.core.id == other.core.id
    }
}

impl<K> Eq for Handle<K> {}

impl<K> std::hash::Hash for Handle<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.core.id.hash(state);
    }
}

#[cfg(feature = "serde")]
impl<K> serde::Serialize for Handle<K> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.id().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, K: SceneResource> serde::Deserialize<'de> for Handle<K> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::unbound(Id::deserialize(deserializer)?))
    }
}

// ========== Typed convenience accessors ==========
//
// These lock the scene internally, so they must not be called while a
// `SceneGuard` is held (the debug re-entrancy panic catches this). On an
// unbound handle they are inert: getters return `None`, mutators no-op.

macro_rules! impl_handle_access {
    ($ty:ty, $get:ident, $get_mut:ident) => {
        impl Handle<$ty> {
            /// A clone of the resource, or `None` if the handle is unbound.
            pub fn get(&self) -> Option<$ty> {
                self.scene()?.lock().$get(self.id()).cloned()
            }

            /// Mutates the resource under the scene lock, returning the
            /// closure's result, or `None` if the handle is unbound.
            pub fn modify<R>(&self, f: impl FnOnce(&mut $ty) -> R) -> Option<R> {
                self.scene()?.lock().$get_mut(self.id()).map(f)
            }
        }
    };
}

impl_handle_access!(super::Instance, get_instance, get_instance_mut);
impl_handle_access!(super::FaceMaterial, get_face_material, get_face_material_mut);
impl_handle_access!(super::LineMaterial, get_line_material, get_line_material_mut);
impl_handle_access!(super::PointMaterial, get_point_material, get_point_material_mut);
impl_handle_access!(super::Texture, get_texture, get_texture_mut);

// `Mesh` deliberately has no `get` — it owns its vertex and index buffers, so
// cloning one out is heavyweight. Mesh access goes through the scene lock.
impl Handle<super::Mesh> {
    /// Mutates the mesh under the scene lock, returning the closure's result,
    /// or `None` if the handle is unbound.
    pub fn modify<R>(&self, f: impl FnOnce(&mut super::Mesh) -> R) -> Option<R> {
        self.scene()?.lock().get_mesh_mut(self.id()).map(f)
    }
}

impl Handle<super::Node> {
    /// A clone of the node, or `None` if the handle is unbound.
    pub fn get(&self) -> Option<super::Node> {
        self.scene()?.lock().get_node(self.id()).cloned()
    }

    /// Sets the node's local transform, invalidating dependent caches.
    pub fn set_transform(&self, transform: crate::common::Transform) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_transform(self.id(), transform);
        }
    }

    /// Sets the node's position, invalidating dependent caches.
    pub fn set_position(&self, position: crate::common::Point3) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_position(self.id(), position);
        }
    }

    /// Sets the node's rotation, invalidating dependent caches.
    pub fn set_rotation(&self, rotation: crate::common::Quaternion) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_rotation(self.id(), rotation);
        }
    }

    /// Sets the node's scale, invalidating dependent caches.
    pub fn set_scale(&self, scale: crate::common::Vector3) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_scale(self.id(), scale);
        }
    }

    /// Sets the node's visibility, propagating to descendants as needed.
    pub fn set_visibility(&self, visibility: super::Visibility) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_visibility(self.id(), visibility);
        }
    }

    /// Sets the node's render-presentation behavior.
    pub fn set_display(&self, display: super::DisplayBehavior) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_display(self.id(), display);
        }
    }

    /// Sets the node's payload, invalidating dependent caches.
    pub fn set_payload(&self, payload: super::NodePayload) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_payload(self.id(), payload);
        }
    }

    /// Sets the node's name.
    pub fn set_name(&self, name: Option<String>) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_name(self.id(), name);
        }
    }

    /// Sets the node's behavior flags.
    pub fn set_flags(&self, flags: super::NodeFlags) {
        if let Some(scene) = self.scene() {
            scene.lock().set_node_flags(self.id(), flags);
        }
    }

    /// Moves the node under a new parent (or to the roots with `None`).
    ///
    /// # Errors
    ///
    /// Fails if the parent is missing or the move would create a cycle.
    pub fn reparent(&self, new_parent: Option<super::NodeId>) -> anyhow::Result<()> {
        match self.scene() {
            Some(scene) => scene.lock().reparent_node(self.id(), new_parent),
            None => anyhow::bail!("handle is not bound to a scene"),
        }
    }

    /// Detaches the node's subtree from the scene tree.
    ///
    /// The node stays alive (this handle owns it) but is no longer rendered or
    /// traversed; [`reparent`](Self::reparent) reattaches it.
    pub fn detach(&self) {
        if let Some(scene) = self.scene() {
            scene.lock().remove_node(self.id());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Transform;
    use crate::resource::{Instance, Mesh, MeshPrimitive, NodeFlags, PrimitiveType, Vertex};
    use crate::SceneData;

    fn test_mesh() -> Mesh {
        let vertices = vec![
            Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
            Vertex { position: [0.5, 1.0, 0.0], tex_coords: [0.5, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
        ];
        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: vec![0, 1, 2],
        }];
        Mesh::from_raw(vertices, primitives)
    }

    #[test]
    fn clone_shares_identity() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let other = mesh.clone();
        assert_eq!(mesh, other);
        assert_eq!(mesh.id(), other.id());
    }

    #[test]
    fn last_drop_reaps_resource() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let id = mesh.id();
        assert_eq!(scene.mesh_count(), 1);

        drop(mesh);
        assert_eq!(scene.mesh_count(), 1); // enqueued, not yet reaped
        scene.reap();
        assert_eq!(scene.mesh_count(), 0);
        assert!(scene.get_mesh(id).is_none());
    }

    #[test]
    fn surviving_clone_keeps_resource() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let keep = mesh.clone();
        drop(mesh);
        scene.reap();
        assert_eq!(scene.mesh_count(), 1);
        drop(keep);
        scene.reap();
        assert_eq!(scene.mesh_count(), 0);
    }

    #[test]
    fn reminted_handle_shares_refcount() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let id = mesh.id();
        let again = scene.mesh_handle(id).unwrap();
        drop(mesh);
        scene.reap();
        assert_eq!(scene.mesh_count(), 1); // `again` still owns it
        drop(again);
        scene.reap();
        assert_eq!(scene.mesh_count(), 0);
    }

    #[test]
    fn revived_resource_survives_pending_drop() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let id = mesh.id();
        drop(mesh); // enqueues
        let revived = scene.mesh_handle(id).unwrap(); // re-mint before reap
        scene.reap();
        assert_eq!(scene.mesh_count(), 1);
        drop(revived);
        scene.reap();
        assert_eq!(scene.mesh_count(), 0);
    }

    #[test]
    fn instance_cascades_to_mesh_and_material() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let mat = scene.add_face_material(crate::resource::FaceMaterial::new());
        let instance = scene.add_instance(Instance::new(mesh).with_face_material(mat));
        // Only the instance handle is held; mesh/material are owned through it.
        assert_eq!(scene.mesh_count(), 1);
        assert_eq!(scene.face_material_count(), 1);

        drop(instance);
        scene.reap();
        assert_eq!(scene.instance_count(), 0);
        assert_eq!(scene.mesh_count(), 0);
        assert_eq!(scene.face_material_count(), 0);
    }

    #[test]
    fn shared_mesh_survives_cascade() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let a = scene.add_instance(Instance::new(mesh.clone()));
        let b = scene.add_instance(Instance::new(mesh));
        drop(a);
        scene.reap();
        assert_eq!(scene.instance_count(), 1);
        assert_eq!(scene.mesh_count(), 1);
        drop(b);
        scene.reap();
        assert_eq!(scene.mesh_count(), 0);
    }

    #[test]
    fn node_removal_cascades_subtree_and_resources() {
        let scene = Scene::default();
        let (root_id, child_id) = {
            let mut data = scene.lock();
            let mesh = data.add_mesh(test_mesh());
            let instance = data.add_instance(Instance::new(mesh));
            let root = data
                .add_node(None, Some("root".into()), Transform::IDENTITY, NodeFlags::NONE)
                .unwrap();
            let child_mesh = data.add_mesh(test_mesh());
            let child = data
                .add_instance_node(
                    Some(root.id()),
                    Instance::new(child_mesh),
                    None,
                    Transform::IDENTITY,
                    NodeFlags::NONE,
                )
                .unwrap();
            data.set_node_payload(root.id(), crate::resource::NodePayload::Instance(instance));
            (root.id(), child.id())
        };
        assert_eq!(scene.lock().node_count(), 2);
        assert_eq!(scene.lock().mesh_count(), 2);

        scene.lock().remove_node(root_id);
        // Reap ran when the guard above dropped.
        let data = scene.lock();
        assert_eq!(data.node_count(), 0);
        assert!(data.get_node(child_id).is_none());
        assert_eq!(data.instance_count(), 0);
        assert_eq!(data.mesh_count(), 0);
    }

    #[test]
    fn external_handle_keeps_detached_node_alive() {
        let scene = Scene::default();
        let node = scene
            .lock()
            .add_node(None, Some("kept".into()), Transform::IDENTITY, NodeFlags::NONE)
            .unwrap();
        scene.lock().remove_node(node.id());
        {
            let data = scene.lock();
            assert_eq!(data.node_count(), 1); // alive but detached
            assert!(!data.is_node_attached(node.id()));
            assert!(data.root_nodes().next().is_none());
        }
        // Reattach to the roots.
        node.reparent(None).unwrap();
        {
            let data = scene.lock();
            assert!(data.is_node_attached(node.id()));
        }
        drop(node);
        assert_eq!(scene.lock().node_count(), 1); // attached: tree owns it
        let root_id = scene.lock().root_nodes().next().unwrap();
        scene.lock().remove_node(root_id);
        assert_eq!(scene.lock().node_count(), 0);
    }

    #[test]
    fn drop_under_guard_reaps_at_guard_drop() {
        let scene = Scene::default();
        let mesh = {
            let mut data = scene.lock();
            data.add_mesh(test_mesh())
        };
        {
            let data = scene.lock();
            drop(mesh); // dropped while the guard is held: must not deadlock
            assert_eq!(data.mesh_count(), 1); // not yet reaped
        }
        assert_eq!(scene.lock().mesh_count(), 0);
    }

    #[test]
    fn cross_thread_drop_reaps_on_next_lock() {
        let scene = Scene::default();
        let mesh = scene.lock().add_mesh(test_mesh());
        std::thread::spawn(move || drop(mesh)).join().unwrap();
        assert_eq!(scene.lock().mesh_count(), 0);
    }

    #[test]
    fn unbound_handle_is_inert() {
        let handle = FaceMaterialHandle::unbound(crate::resource::FaceMaterialId::new());
        assert!(handle.get().is_none());
        assert!(handle.modify(|_| ()).is_none());
        drop(handle); // no scene to enqueue on; must not panic
    }

    #[test]
    fn removals_feed_reports_reaped_resources() {
        let mut scene = SceneData::new();
        let mesh = scene.add_mesh(test_mesh());
        let id = mesh.id();
        drop(mesh);
        scene.reap();
        let removals = scene.take_removals();
        assert!(removals.contains(&(ResourceKind::Mesh, id.erased())));
        assert!(scene.take_removals().is_empty());
    }

    #[test]
    fn handle_mutators_lock_scene() {
        let scene = Scene::default();
        let node = scene
            .lock()
            .add_node(None, None, Transform::IDENTITY, NodeFlags::NONE)
            .unwrap();
        node.set_name(Some("renamed".into()));
        assert_eq!(node.get().unwrap().name.as_deref(), Some("renamed"));
    }
}
