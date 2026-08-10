use super::*;
use duck_engine_common::{Point3, Quaternion, Vector3, Matrix4, SquareMatrix};
use crate::common::{EPSILON, Transform};
use crate::resource::{MeshPrimitive, PrimitiveType, Vertex};

#[test]
fn test_scene_new() {
    let scene = SceneData::new();

    assert_eq!(scene.mesh_count(), 0);
    assert_eq!(scene.instance_count(), 0);
    assert_eq!(scene.node_count(), 0);
    assert_eq!(scene.root_nodes().count(), 0);
}

#[test]
fn test_add_instance() {
    let mut scene = SceneData::new();
    let mesh_id = MeshId::new();
    let mat_id = FaceMaterialId::new();

    let instance = scene.add_instance(
        Instance::new(MeshHandle::unbound(mesh_id))
            .with_face_material(FaceMaterialHandle::unbound(mat_id)),
    );
    assert_eq!(scene.instance_count(), 1);

    let instance = scene.get_instance(instance.id()).unwrap();
    assert_eq!(instance.mesh(), mesh_id);
    assert_eq!(instance.face_material(), Some(mat_id));
}

#[test]
fn test_add_multiple_instances() {
    let mut scene = SceneData::new();

    let id1 = scene.add_instance(Instance::new(MeshHandle::unbound(MeshId::new())));
    let id2 = scene.add_instance(Instance::new(MeshHandle::unbound(MeshId::new())));
    let id3 = scene.add_instance(Instance::new(MeshHandle::unbound(MeshId::new())));

    assert_ne!(id1, id2);
    assert_ne!(id1, id3);
    assert_ne!(id2, id3);
    assert_eq!(scene.instance_count(), 3);
}

#[test]
fn test_add_root_node() {
    let mut scene = SceneData::new();

    let node_id = scene.add_default_node(None, None).unwrap().id();

    assert_eq!(scene.node_count(), 1);
    assert_eq!(scene.root_nodes().count(), 1);
    assert_eq!(scene.root_nodes().next(), Some(node_id));

    let node = scene.get_node(node_id).unwrap();
    assert_eq!(node.parent(), None);
    assert_eq!(node.child_count(), 0);
}

#[test]
fn test_add_child_node() {
    let mut scene = SceneData::new();

    let root = scene.add_default_node(None, None).unwrap().id();
    let child = scene.add_default_node(Some(root), None).unwrap().id();

    assert_eq!(scene.node_count(), 2);
    assert_eq!(scene.root_nodes().count(), 1);

    // Verify parent-child relationship is bidirectional
    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.child_count(), 1);
    assert_eq!(root_node.children().next(), Some(child));

    let child_node = scene.get_node(child).unwrap();
    assert_eq!(child_node.parent(), Some(root));
}

#[test]
fn test_add_multiple_children() {
    let mut scene = SceneData::new();

    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(root), None).unwrap().id();
    let child3 = scene.add_default_node(Some(root), None).unwrap().id();

    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.child_count(), 3);
    assert!(root_node.has_child(child1));
    assert!(root_node.has_child(child2));
    assert!(root_node.has_child(child3));

    // Verify all children point back to root
    assert_eq!(scene.get_node(child1).unwrap().parent(), Some(root));
    assert_eq!(scene.get_node(child2).unwrap().parent(), Some(root));
    assert_eq!(scene.get_node(child3).unwrap().parent(), Some(root));
}

#[test]
fn test_deep_hierarchy() {
    let mut scene = SceneData::new();

    // Create a chain: root -> child1 -> child2 -> child3
    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(child1), None).unwrap().id();
    let child3 = scene.add_default_node(Some(child2), None).unwrap().id();

    // Verify the chain is correct
    assert_eq!(scene.get_node(root).unwrap().parent(), None);
    assert_eq!(scene.get_node(child1).unwrap().parent(), Some(root));
    assert_eq!(scene.get_node(child2).unwrap().parent(), Some(child1));
    assert_eq!(scene.get_node(child3).unwrap().parent(), Some(child2));

    // Verify each node has the correct children
    assert_eq!(scene.get_node(root).unwrap().child_count(), 1);
    assert_eq!(scene.get_node(child1).unwrap().child_count(), 1);
    assert_eq!(scene.get_node(child2).unwrap().child_count(), 1);
    assert_eq!(scene.get_node(child3).unwrap().child_count(), 0);
}

#[test]
fn test_multiple_root_nodes() {
    let mut scene = SceneData::new();

    let root1 = scene.add_default_node(None, None).unwrap().id();
    let root2 = scene.add_default_node(None, None).unwrap().id();
    let root3 = scene.add_default_node(None, None).unwrap().id();

    assert_eq!(scene.root_nodes().count(), 3);
    assert!(scene.root_nodes().any(|id| id == root1));
    assert!(scene.root_nodes().any(|id| id == root2));
    assert!(scene.root_nodes().any(|id| id == root3));

    // Verify all are truly roots
    assert_eq!(scene.get_node(root1).unwrap().parent(), None);
    assert_eq!(scene.get_node(root2).unwrap().parent(), None);
    assert_eq!(scene.get_node(root3).unwrap().parent(), None);
}

#[test]
fn test_complex_tree_structure() {
    let mut scene = SceneData::new();

    // Create a tree:
    //       root
    //      /    \
    //    c1      c2
    //   /  \      \
    //  gc1  gc2   gc3

    let root = scene.add_default_node(None, None).unwrap().id();
    let c1 = scene.add_default_node(Some(root), None).unwrap().id();
    let c2 = scene.add_default_node(Some(root), None).unwrap().id();
    let gc1 = scene.add_default_node(Some(c1), None).unwrap().id();
    let gc2 = scene.add_default_node(Some(c1), None).unwrap().id();
    let gc3 = scene.add_default_node(Some(c2), None).unwrap().id();

    // Verify structure
    assert_eq!(scene.root_nodes().count(), 1);
    assert_eq!(scene.node_count(), 6);

    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.child_count(), 2);

    let c1_node = scene.get_node(c1).unwrap();
    assert_eq!(c1_node.child_count(), 2);

    let c2_node = scene.get_node(c2).unwrap();
    assert_eq!(c2_node.child_count(), 1);

    // Verify leaf nodes have no children
    assert_eq!(scene.get_node(gc1).unwrap().child_count(), 0);
    assert_eq!(scene.get_node(gc2).unwrap().child_count(), 0);
    assert_eq!(scene.get_node(gc3).unwrap().child_count(), 0);
}

#[test]
fn test_add_instance_node() {
    let mut scene = SceneData::new();
    let mesh_id = MeshId::new();
    let mat_id = FaceMaterialId::new();

    let node = scene.add_instance_node(
        None,
        Instance::new(MeshHandle::unbound(mesh_id))
            .with_face_material(FaceMaterialHandle::unbound(mat_id)),
        None,
        Transform::IDENTITY,
        NodeFlags::NONE,
    ).unwrap();

    assert_eq!(scene.node_count(), 1);
    assert_eq!(scene.instance_count(), 1);

    let node = scene.get_node(node.id()).unwrap();
    let instance_id = match node.payload() {
        NodePayload::Instance(h) => h.id(),
        _ => panic!("expected Instance payload"),
    };
    let instance = scene.get_instance(instance_id).unwrap();
    assert_eq!(instance.mesh(), mesh_id);
    assert_eq!(instance.face_material(), Some(mat_id));
}

#[test]
fn test_root_node_identity_transform() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();

    let transform = scene.nodes_transform(root).unwrap();
    let identity = Matrix4::identity();

    // Convert to arrays for comparison
    let t: [[f32; 4]; 4] = transform.into();
    let i: [[f32; 4]; 4] = identity.into();

    for row in 0..4 {
        for col in 0..4 {
            assert!((t[row][col] - i[row][col]).abs() < EPSILON,
                "Transform mismatch at [{row}][{col}]");
        }
    }
}

#[test]
fn test_child_transform_accumulation() {
    let mut scene = SceneData::new();

    // Root at (10, 0, 0)
    let root = scene.add_node(
        None,
        None,
        Transform::from_position(Point3::new(10.0, 0.0, 0.0)),
        NodeFlags::NONE,
    ).unwrap().id();

    // Child at (5, 0, 0) relative to parent
    let child = scene.add_node(
        Some(root),
        None,
        Transform::from_position(Point3::new(5.0, 0.0, 0.0)),
        NodeFlags::NONE,
    ).unwrap().id();

    let child_transform = scene.nodes_transform(child).unwrap();

    // Child should be at (15, 0, 0) in world space
    assert!((child_transform[3][0] - 15.0).abs() < EPSILON);
    assert!((child_transform[3][1] - 0.0).abs() < EPSILON);
    assert!((child_transform[3][2] - 0.0).abs() < EPSILON);
}

#[test]
fn test_transform_with_scale() {
    let mut scene = SceneData::new();

    // Parent with scale 2.0
    let parent = scene.add_node(
        None,
        None,
        Transform::new(
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(2.0, 2.0, 2.0),
        ),
        NodeFlags::NONE,
    ).unwrap().id();

    // Child at (1, 0, 0)
    let child = scene.add_node(
        Some(parent),
        None,
        Transform::from_position(Point3::new(1.0, 0.0, 0.0)),
        NodeFlags::NONE,
    ).unwrap().id();

    let child_transform = scene.nodes_transform(child).unwrap();

    // Child should be at (2, 0, 0) due to parent's scale
    assert!((child_transform[3][0] - 2.0).abs() < EPSILON);
}

#[test]
fn test_transform_caching() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();

    // First computation
    let transform1 = scene.nodes_transform(root).unwrap();

    // Second computation should use cache
    let transform2 = scene.nodes_transform(root).unwrap();

    // Should be identical
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(transform1[i][j], transform2[i][j]);
        }
    }

    // Verify cache is being used
    let node = scene.get_node(root).unwrap();
    assert!(!node.transform_dirty());
}

#[test]
fn test_transform_invalidation_on_change() {
    let mut scene = SceneData::new();
    let node_id = scene.add_default_node(None, None).unwrap().id();

    // Compute transform to populate cache
    let _transform = scene.nodes_transform(node_id).unwrap();

    // Modify the node
    scene.set_node_position(node_id, Point3::new(5.0, 5.0, 5.0));

    // Cache should be dirty
    let node = scene.get_node(node_id).unwrap();
    assert!(node.transform_dirty());
}

#[test]
fn test_tree_consistency_after_multiple_operations() {
    let mut scene = SceneData::new();

    // Build a tree
    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(root), None).unwrap().id();
    let grandchild = scene.add_default_node(Some(child1), None).unwrap().id();

    // Verify every node can be accessed
    assert!(scene.get_node(root).is_some());
    assert!(scene.get_node(child1).is_some());
    assert!(scene.get_node(child2).is_some());
    assert!(scene.get_node(grandchild).is_some());

    // Verify every parent reference is valid
    for node in scene.nodes() {
        if let Some(parent_id) = node.parent() {
            assert!(scene.get_node(parent_id).is_some(),
                "Node references non-existent parent");
        }
    }

    // Verify every child reference is valid
    for node in scene.nodes() {
        for child_id in node.children() {
            assert!(scene.get_node(child_id).is_some(),
                "Node references non-existent child");
        }
    }

    // Verify all root nodes are actually roots
    let roots: Vec<NodeId> = scene.root_nodes().collect();
    for root_id in roots {
        let node = scene.get_node(root_id).unwrap();
        assert!(node.parent().is_none(),
            "Root node has a parent");
    }
}

#[test]
fn test_bidirectional_consistency() {
    let mut scene = SceneData::new();

    let parent = scene.add_default_node(None, None).unwrap().id();
    let child = scene.add_default_node(Some(parent), None).unwrap().id();

    // Parent should list child
    let parent_node = scene.get_node(parent).unwrap();
    assert!(parent_node.has_child(child));

    // Child should reference parent
    let child_node = scene.get_node(child).unwrap();
    assert_eq!(child_node.parent(), Some(parent));
}

#[test]
fn test_no_orphaned_nodes() {
    let mut scene = SceneData::new();

    let root1 = scene.add_default_node(None, None).unwrap().id();
    let root2 = scene.add_default_node(None, None).unwrap().id();
    let _child1 = scene.add_default_node(Some(root1), None).unwrap();
    let _child2 = scene.add_default_node(Some(root2), None).unwrap();

    // All nodes should be reachable from roots
    let mut reachable = std::collections::HashSet::new();

    fn visit_tree(
        scene: &SceneData,
        node_id: NodeId,
        reachable: &mut std::collections::HashSet<NodeId>
    ) {
        reachable.insert(node_id);
        let node = scene.get_node(node_id).unwrap();
        for child_id in node.children() {
            visit_tree(scene, child_id, reachable);
        }
    }

    let roots: Vec<NodeId> = scene.root_nodes().collect();
    for root_id in roots {
        visit_tree(&scene, root_id, &mut reachable);
    }

    // All nodes should be reachable
    assert_eq!(reachable.len(), scene.node_count());
}

#[test]
fn test_large_tree_consistency() {
    let mut scene = SceneData::new();

    let root = scene.add_default_node(None, None).unwrap().id();

    // Create 100 children
    let mut children = Vec::new();
    for _ in 0..100 {
        let child = scene.add_default_node(Some(root), None).unwrap().id();
        children.push(child);
    }

    // Verify all children are registered
    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.child_count(), 100);

    // Verify all children reference the parent
    for &child_id in &children {
        let child_node = scene.get_node(child_id).unwrap();
        assert_eq!(child_node.parent(), Some(root));
    }
}

#[test]
fn test_add_node_with_invalid_parent_fails() {
    let mut scene = SceneData::new();
    let nonexistent = crate::resource::Id::new();

    let result = scene.add_node(
        Some(nonexistent),
        None,
        Transform::IDENTITY,
        NodeFlags::NONE,
    );

    assert!(result.is_err());
    assert_eq!(scene.node_count(), 0);
}

#[test]
fn test_add_default_node_with_invalid_parent_fails() {
    let mut scene = SceneData::new();
    let nonexistent = crate::resource::Id::new();

    let result = scene.add_default_node(Some(nonexistent), None);

    assert!(result.is_err());
    assert_eq!(scene.node_count(), 0);
}

#[test]
fn test_add_instance_node_with_invalid_parent_fails() {
    let mut scene = SceneData::new();
    let nonexistent = NodeId::new();

    let result = scene.add_instance_node(
        Some(nonexistent),
        Instance::new(MeshHandle::unbound(MeshId::new()))
            .with_face_material(FaceMaterialHandle::unbound(FaceMaterialId::new())),
        None,
        Transform::IDENTITY,
        NodeFlags::NONE,
    );

    assert!(result.is_err());
    // No node should be created
    assert_eq!(scene.node_count(), 0);
    // The instance created before node validation has no owner left, so the
    // next reap collects it.
    scene.reap();
    assert_eq!(scene.instance_count(), 0);
}

#[test]
fn test_set_node_visibility_to_invisible() {
    let mut scene = SceneData::new();
    let node = scene.add_default_node(None, None).unwrap().id();

    // Default is visible
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Visible);

    // Set to invisible
    scene.set_node_visibility(node, Visibility::Invisible);
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_set_node_visibility_to_visible() {
    let mut scene = SceneData::new();
    let node = scene.add_default_node(None, None).unwrap().id();

    // Set to invisible first
    scene.set_node_visibility(node, Visibility::Invisible);
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Invisible);

    // Set back to visible
    scene.set_node_visibility(node, Visibility::Visible);
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Visible);
}

#[test]
fn test_visibility_propagation_to_children() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(child1), None).unwrap().id();

    // All default to visible
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(child1).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(child2).unwrap().visibility(), Visibility::Visible);

    // Set root to invisible - should propagate to all descendants
    scene.set_node_visibility(root, Visibility::Invisible);
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child1).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child2).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_visibility_propagation_to_multiple_children() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(root), None).unwrap().id();
    let child3 = scene.add_default_node(Some(root), None).unwrap().id();

    // Set root invisible - all children should become invisible
    scene.set_node_visibility(root, Visibility::Invisible);
    assert_eq!(scene.get_node(child1).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child2).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child3).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_visibility_no_propagation_when_setting_visible() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child = scene.add_default_node(Some(root), None).unwrap().id();

    // Set root invisible (propagates to child)
    scene.set_node_visibility(root, Visibility::Invisible);
    assert_eq!(scene.get_node(child).unwrap().visibility(), Visibility::Invisible);

    // Set root visible - child should stay invisible
    scene.set_node_visibility(root, Visibility::Visible);
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(child).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_effective_visibility_all_visible() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(child1), None).unwrap().id();

    // All nodes visible - effective visibility should be Visible
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Visible);
    assert_eq!(scene.node_effective_visibility(child1), EffectiveVisibility::Visible);
    assert_eq!(scene.node_effective_visibility(child2), EffectiveVisibility::Visible);
}

#[test]
fn test_effective_visibility_node_invisible() {
    let mut scene = SceneData::new();
    let node = scene.add_default_node(None, None).unwrap().id();

    scene.set_node_visibility(node, Visibility::Invisible);
    assert_eq!(scene.node_effective_visibility(node), EffectiveVisibility::Invisible);
}

#[test]
fn test_effective_visibility_mixed() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child1 = scene.add_default_node(Some(root), None).unwrap().id();
    let child2 = scene.add_default_node(Some(root), None).unwrap().id();

    // Make child1 invisible
    scene.set_node_visibility(child1, Visibility::Invisible);

    // Root should have Mixed effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Mixed);
    // child1 is invisible
    assert_eq!(scene.node_effective_visibility(child1), EffectiveVisibility::Invisible);
    // child2 is visible (leaf node, no children)
    assert_eq!(scene.node_effective_visibility(child2), EffectiveVisibility::Visible);
}

#[test]
fn test_effective_visibility_caching() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let _child = scene.add_default_node(Some(root), None).unwrap();

    // First call computes and caches
    let effective1 = scene.node_effective_visibility(root);
    assert_eq!(effective1, EffectiveVisibility::Visible);

    // Node should have cached value
    assert!(!scene.get_node(root).unwrap().effective_visibility_dirty());

    // Second call should use cache
    let effective2 = scene.node_effective_visibility(root);
    assert_eq!(effective2, EffectiveVisibility::Visible);
}

#[test]
fn test_effective_visibility_cache_invalidation_on_child_change() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child = scene.add_default_node(Some(root), None).unwrap().id();

    // Compute and cache root's effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Visible);
    assert!(!scene.get_node(root).unwrap().effective_visibility_dirty());

    // Change child visibility - should invalidate root's cache
    scene.set_node_visibility(child, Visibility::Invisible);
    assert!(scene.get_node(root).unwrap().effective_visibility_dirty());

    // Now root should have Mixed effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Mixed);
}

#[test]
fn test_deep_hierarchy_visibility_propagation() {
    let mut scene = SceneData::new();
    let level0 = scene.add_default_node(None, None).unwrap().id();
    let level1 = scene.add_default_node(Some(level0), None).unwrap().id();
    let level2 = scene.add_default_node(Some(level1), None).unwrap().id();
    let level3 = scene.add_default_node(Some(level2), None).unwrap().id();
    let level4 = scene.add_default_node(Some(level3), None).unwrap().id();

    // Set level2 invisible - should propagate to level3 and level4
    scene.set_node_visibility(level2, Visibility::Invisible);

    assert_eq!(scene.get_node(level0).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(level1).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(level2).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(level3).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(level4).unwrap().visibility(), Visibility::Invisible);

    // Effective visibility
    assert_eq!(scene.node_effective_visibility(level0), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(level1), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(level2), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(level3), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(level4), EffectiveVisibility::Invisible);
}

#[test]
fn test_visibility_with_complex_tree() {
    let mut scene = SceneData::new();

    // Create tree:
    //     root
    //    /  |  \
    //   a   b   c
    //  / \     / \
    // d   e   f   g

    let root = scene.add_default_node(None, None).unwrap().id();
    let a = scene.add_default_node(Some(root), None).unwrap().id();
    let b = scene.add_default_node(Some(root), None).unwrap().id();
    let c = scene.add_default_node(Some(root), None).unwrap().id();
    let d = scene.add_default_node(Some(a), None).unwrap().id();
    let e = scene.add_default_node(Some(a), None).unwrap().id();
    let f = scene.add_default_node(Some(c), None).unwrap().id();
    let g = scene.add_default_node(Some(c), None).unwrap().id();

    // Make 'a' invisible (affects d and e)
    scene.set_node_visibility(a, Visibility::Invisible);

    // Make 'f' invisible
    scene.set_node_visibility(f, Visibility::Invisible);

    // Check explicit visibility
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(a).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(b).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(c).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(d).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(e).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(f).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(g).unwrap().visibility(), Visibility::Visible);

    // Check effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(a), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(b), EffectiveVisibility::Visible);
    assert_eq!(scene.node_effective_visibility(c), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(d), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(e), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(f), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(g), EffectiveVisibility::Visible);
}

#[test]
fn test_visibility_leaf_node_always_visible_or_invisible() {
    let mut scene = SceneData::new();
    let leaf = scene.add_default_node(None, None).unwrap().id();

    // Leaf nodes can only be Visible or Invisible, never Mixed
    assert_eq!(scene.node_effective_visibility(leaf), EffectiveVisibility::Visible);

    scene.set_node_visibility(leaf, Visibility::Invisible);
    assert_eq!(scene.node_effective_visibility(leaf), EffectiveVisibility::Invisible);
}

#[test]
fn test_mesh_translate() {
    let mut mesh = Mesh::sphere(1.0, 8, 4, PrimitiveType::LineList);
    let gen_before = mesh.generation();

    mesh.translate(Vector3::new(5.0, 0.0, 0.0));

    assert!(mesh.generation() > gen_before);
    // All vertices should have x >= 4.0 (radius 1.0 + offset 5.0)
    for v in mesh.vertices() {
        assert!(v.position[0] >= 4.0 - EPSILON);
    }
}

#[test]
fn test_mesh_translated_chaining() {
    let mesh = Mesh::sphere(1.0, 8, 4, PrimitiveType::TriangleList)
        .translated(Vector3::new(0.0, 10.0, 0.0));

    // All vertices should have y offset by 10
    for v in mesh.vertices() {
        assert!(v.position[1] >= 9.0 - EPSILON);
    }
}

#[test]
fn test_mesh_transform_identity() {
    let original = Mesh::cube(1.0, PrimitiveType::TriangleList);
    let original_positions: Vec<[f32; 3]> = original.vertices().iter().map(|v| v.position).collect();

    let mut transformed = original.clone();
    transformed.transform(&Matrix4::identity());

    for (orig, trans) in original_positions.iter().zip(transformed.vertices()) {
        assert!((orig[0] - trans.position[0]).abs() < EPSILON);
        assert!((orig[1] - trans.position[1]).abs() < EPSILON);
        assert!((orig[2] - trans.position[2]).abs() < EPSILON);
    }
}

#[test]
fn test_mesh_transform_translation() {
    let mut mesh = Mesh::cube(1.0, PrimitiveType::TriangleList);
    let offset = Vector3::new(3.0, 4.0, 5.0);
    mesh.transform(&Matrix4::from_translation(offset));

    let bounds = mesh.bounding().unwrap();
    // Cube was [-0.5, 0.5]^3, now should be [2.5, 3.5] x [3.5, 4.5] x [4.5, 5.5]
    assert!((bounds.min.x - 2.5).abs() < EPSILON);
    assert!((bounds.min.y - 3.5).abs() < EPSILON);
    assert!((bounds.min.z - 4.5).abs() < EPSILON);
}

#[test]
fn test_cone_directed_apex_position() {
    let apex = Point3::new(1.0, 2.0, 3.0);
    let direction = Vector3::new(0.0, -1.0, 0.0);
    let mesh = Mesh::cone_directed(apex, direction, 0.5, 2.0, 8, false, PrimitiveType::LineList);

    // At least one vertex should be very close to the apex
    let has_apex_vertex = mesh.vertices().iter().any(|v| {
        let dx = v.position[0] - apex.x;
        let dy = v.position[1] - apex.y;
        let dz = v.position[2] - apex.z;
        (dx * dx + dy * dy + dz * dz).sqrt() < 0.1
    });
    assert!(has_apex_vertex, "No vertex found near the apex");
}

#[test]
fn test_cone_directed_base_along_direction() {
    let apex = Point3::new(0.0, 0.0, 0.0);
    let direction = Vector3::new(0.0, 0.0, 1.0);
    let height = 3.0;
    let mesh = Mesh::cone_directed(apex, direction, 1.0, height, 16, false, PrimitiveType::TriangleList);

    let bounds = mesh.bounding().unwrap();
    // The cone extends from the apex along direction.
    // Bounding box max.z should be past the apex (into positive z).
    assert!(bounds.max.z > 0.0, "Cone should extend in +Z direction");
    // And min.z should be near the apex
    assert!(bounds.min.z < height, "Cone min.z should be less than height");
}

// ============== NodeFlags bounding tests ==============

fn make_unit_mesh() -> Mesh {
    Mesh::from_raw(
        vec![
            Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
            Vertex { position: [0.0, 1.0, 0.0], tex_coords: [0.0, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
        ],
        vec![MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] }],
    )
}

fn add_geometry_node(scene: &mut SceneData, parent: Option<NodeId>, flags: NodeFlags) -> NodeId {
    let mesh = scene.add_mesh(make_unit_mesh());
    let material = scene.add_face_material(FaceMaterial::new());
    let instance = Instance::new(mesh).with_face_material(material);
    scene.add_instance_node(parent, instance, None, Transform::IDENTITY, flags).unwrap().id()
}

#[test]
fn test_does_not_contribute_bounding_returns_no_bounds() {
    let mut scene = SceneData::new();
    let node_id = add_geometry_node(&mut scene, None, NodeFlags::DOES_NOT_CONTRIBUTE_BOUNDING);

    let result = scene.nodes_bounding(node_id);
    assert!(result.bounds.is_none());
    assert!(!result.incomplete);
}

#[test]
fn test_does_not_contribute_bounding_excluded_from_parent() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let _normal = add_geometry_node(&mut scene, Some(root), NodeFlags::NONE);
    let _flagged = add_geometry_node(&mut scene, Some(root), NodeFlags::DOES_NOT_CONTRIBUTE_BOUNDING);

    // Root bounds should come only from the normal child — both have unit geometry
    // at the origin so the bounds should match a single unit mesh.
    let result = scene.nodes_bounding(root);
    assert!(result.bounds.is_some());
    assert!(!result.incomplete);

    // The normal child contributes bounds; the flagged one is excluded.
    // Both use the same unit mesh, so the result matches one unit mesh's bounds.
    let bounds = result.bounds.unwrap();
    assert!((bounds.max.x - 1.0).abs() < EPSILON);
    assert!((bounds.max.y - 1.0).abs() < EPSILON);
}

#[test]
fn test_does_not_contribute_bounding_children_also_excluded() {
    let mut scene = SceneData::new();
    // Flagged parent whose only child has real geometry.
    let flagged_parent = scene.add_node(None, None, Transform::IDENTITY, NodeFlags::DOES_NOT_CONTRIBUTE_BOUNDING).unwrap().id();
    let _child = add_geometry_node(&mut scene, Some(flagged_parent), NodeFlags::NONE);

    let result = scene.nodes_bounding(flagged_parent);
    assert!(result.bounds.is_none());
    assert!(!result.incomplete);
}

#[test]
fn test_normal_node_contributes_bounding() {
    let mut scene = SceneData::new();
    let node_id = add_geometry_node(&mut scene, None, NodeFlags::NONE);

    let result = scene.nodes_bounding(node_id);
    assert!(result.bounds.is_some());
    assert!(!result.incomplete);
}

// ============== Ownership / reap tests ==============

#[test]
fn test_remove_node_cascades_owned_resources() {
    let mut scene = SceneData::new();
    let node_id = add_geometry_node(&mut scene, None, NodeFlags::NONE);
    assert_eq!(scene.mesh_count(), 1);
    assert_eq!(scene.face_material_count(), 1);
    assert_eq!(scene.instance_count(), 1);

    scene.remove_node(node_id);
    assert_eq!(scene.node_count(), 0);
    assert_eq!(scene.instance_count(), 0);
    assert_eq!(scene.mesh_count(), 0);
    assert_eq!(scene.face_material_count(), 0);
}

#[test]
fn test_shared_resources_survive_node_removal() {
    let mut scene = SceneData::new();
    let mesh = scene.add_mesh(make_unit_mesh());
    let a = scene
        .add_instance_node(None, Instance::new(mesh.clone()), None, Transform::IDENTITY, NodeFlags::NONE)
        .unwrap()
        .id();
    let b = scene
        .add_instance_node(None, Instance::new(mesh), None, Transform::IDENTITY, NodeFlags::NONE)
        .unwrap()
        .id();
    // The local `mesh` binding was moved into the second instance; both
    // instances share the mesh.
    assert_eq!(scene.mesh_count(), 1);

    scene.remove_node(a);
    assert_eq!(scene.node_count(), 1);
    assert_eq!(scene.mesh_count(), 1);

    scene.remove_node(b);
    assert_eq!(scene.node_count(), 0);
    assert_eq!(scene.mesh_count(), 0);
}

#[test]
fn test_material_owns_texture() {
    let mut scene = SceneData::new();
    let mesh = scene.add_mesh(make_unit_mesh());
    let texture = scene.add_texture_from_image(image::DynamicImage::new_rgba8(2, 2));
    let material = scene.add_face_material(FaceMaterial::new().with_base_color_texture(texture));
    let node = scene
        .add_instance_node(
            None,
            Instance::new(mesh).with_face_material(material),
            None,
            Transform::IDENTITY,
            NodeFlags::NONE,
        )
        .unwrap()
        .id();
    assert_eq!(scene.texture_count(), 1);

    scene.remove_node(node);
    assert_eq!(scene.face_material_count(), 0);
    assert_eq!(scene.texture_count(), 0);
}

#[test]
fn test_reparent_node_moves_subtree() {
    let mut scene = SceneData::new();
    let a = scene.add_default_node(None, None).unwrap().id();
    let b = scene.add_default_node(None, None).unwrap().id();
    let child = scene.add_default_node(Some(a), None).unwrap().id();

    scene.reparent_node(child, Some(b)).unwrap();
    assert!(!scene.get_node(a).unwrap().has_child(child));
    assert!(scene.get_node(b).unwrap().has_child(child));
    assert_eq!(scene.get_node(child).unwrap().parent(), Some(b));

    // To the roots.
    scene.reparent_node(child, None).unwrap();
    assert_eq!(scene.get_node(child).unwrap().parent(), None);
    assert!(scene.root_nodes().any(|id| id == child));
    assert_eq!(scene.node_count(), 3);
}

#[test]
fn test_reparent_node_rejects_cycles() {
    let mut scene = SceneData::new();
    let root = scene.add_default_node(None, None).unwrap().id();
    let child = scene.add_default_node(Some(root), None).unwrap().id();
    let grandchild = scene.add_default_node(Some(child), None).unwrap().id();

    assert!(scene.reparent_node(root, Some(grandchild)).is_err());
    assert!(scene.reparent_node(root, Some(root)).is_err());
    // Structure unchanged.
    assert!(scene.is_node_attached(grandchild));
    assert_eq!(scene.get_node(grandchild).unwrap().parent(), Some(child));
}

#[test]
fn test_scene_clone_is_independent() {
    let mut scene = SceneData::new();
    let node_id = add_geometry_node(&mut scene, None, NodeFlags::NONE);

    let mut clone = scene.clone();
    assert_eq!(clone.node_count(), 1);
    assert_eq!(clone.mesh_count(), 1);

    // Removing from the original must not affect the clone.
    scene.remove_node(node_id);
    assert_eq!(scene.mesh_count(), 0);
    clone.reap();
    assert_eq!(clone.node_count(), 1);
    assert_eq!(clone.mesh_count(), 1);

    // And vice versa.
    clone.remove_node(node_id);
    assert_eq!(clone.mesh_count(), 0);
}

// ============== SceneData completeness tests ==============

#[test]
fn test_is_complete_fully_populated_scene() {
    let mut scene = SceneData::new();
    let mesh = scene.add_mesh(make_unit_mesh());
    let face = scene.add_face_material(FaceMaterial::new());
    let line = scene.add_line_material(LineMaterial::new(common::RgbaColor::WHITE));
    let instance = Instance::new(mesh).with_face_material(face).with_line_material(line);
    let parent = scene
        .add_instance_node(None, instance, None, Transform::IDENTITY, NodeFlags::NONE)
        .unwrap()
        .id();
    scene.add_default_node(Some(parent), None).unwrap();

    assert!(scene.is_complete());
}

#[test]
fn test_is_complete_missing_mesh() {
    let mut scene = SceneData::new();
    let face = scene.add_face_material(FaceMaterial::new());
    // The mesh is never added to the scene.
    let instance = Instance::new(MeshHandle::unbound(MeshId::new())).with_face_material(face);
    scene
        .add_instance_node(None, instance, None, Transform::IDENTITY, NodeFlags::NONE)
        .unwrap();

    assert!(!scene.is_complete());
}

#[test]
fn test_is_complete_missing_material() {
    let mut scene = SceneData::new();
    let mesh = scene.add_mesh(make_unit_mesh());
    // The face material is never added to the scene.
    let instance =
        Instance::new(mesh).with_face_material(FaceMaterialHandle::unbound(FaceMaterialId::new()));
    scene
        .add_instance_node(None, instance, None, Transform::IDENTITY, NodeFlags::NONE)
        .unwrap();

    assert!(!scene.is_complete());
}

#[test]
fn test_is_complete_dangling_child() {
    let mut scene = SceneData::new();
    // A node whose child was listed but never inserted, as during a partial import.
    let mut node = Node::new(None, Transform::IDENTITY, NodeFlags::NONE);
    node.add_child_unchecked(NodeHandle::unbound(NodeId::new()));
    let _handle = scene.insert_node(node);

    assert!(!scene.is_complete());
}

#[test]
fn test_node_setters_ignore_missing_node() {
    let mut scene = SceneData::new();
    let missing = NodeId::new();
    let generation = scene.node_generation();

    scene.set_node_name(missing, Some("ghost".into()));
    scene.set_node_transform(missing, Transform::IDENTITY);
    scene.set_node_visibility(missing, Visibility::Visible);

    assert_eq!(scene.node_generation(), generation);
    assert_eq!(scene.node_effective_visibility(missing), EffectiveVisibility::Invisible);
}
