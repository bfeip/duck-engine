//! A 3D cursor: a small marker placed at a world point and kept at a
//! constant on-screen size. Owned by the modeler and driven each frame from the
//! active tool's reported target.
//!
//! The marker is a screen-facing, screen-sized textured quad showing an
//! anti-aliased dot. Billboarding and constant-pixel scaling are handled by the
//! renderer via [`DisplayBehavior`], so this module only places the node and
//! lets the render-time presentation do the rest. The dot's color is settable
//! ([`Cursor3d::set_color`]); it tints a white disc texture via the material's
//! base-color factor.

use duck_engine_viewer::common::{Point3, RgbaColor, Transform};
use duck_engine_viewer::scene::resource::{
    AlphaMode, DisplayBehavior, FaceMaterial, FaceMaterialHandle, Instance, MaterialFlags, Mesh,
    NodeFlags, NodeHandle, NodeId, PrimitiveType, RenderLayer, Texture, Visibility,
};
use duck_engine_viewer::scene::{Scene, SceneData};

/// On-screen diameter of the dot, in pixels. The quad's half-width (0.5) is
/// scaled by `screen_size`, so this is also the screen-space size value.
const CURSOR_PIXELS: f32 = 8.0;

/// Default marker color (amber), chosen to read against both the grid and parts.
const MARKER_COLOR: RgbaColor = RgbaColor { r: 1.0, g: 0.78, b: 0.12, a: 1.0 };

/// Edge of the dot texture, in source-texel units, over which alpha ramps from
/// 1 to 0 for anti-aliasing.
const DOT_EDGE_TEXELS: f32 = 1.5;

/// Resolution of the generated dot texture (square).
const DOT_TEXTURE_SIZE: u32 = 64;

/// A marker placed at a world point and kept at a constant pixel size. The node
/// is created lazily on first show; redundant updates are skipped so it is cheap
/// to call every frame.
pub struct Cursor3d {
    /// Owning handle: keeps the marker (and its mesh/material/texture) alive.
    node: Option<NodeHandle>,
    /// Material backing the dot; kept so [`set_color`](Self::set_color) can
    /// retint the white disc.
    material: Option<FaceMaterialHandle>,
    /// Desired dot color; applied to the material as its base-color factor.
    color: RgbaColor,
    /// Last position actually written, to skip redundant writes. Scale is now
    /// owned by the renderer (screen_size), so it is no longer tracked here.
    shown: Option<Point3>,
}

impl Default for Cursor3d {
    fn default() -> Self {
        Self {
            node: None,
            material: None,
            color: MARKER_COLOR,
            shown: None,
        }
    }
}

impl Cursor3d {
    /// Places the cursor at `target`, or hides it when `target` is `None`.
    pub fn update(&mut self, target: Option<Point3>, scene: &Scene) {
        let mut scene = scene.lock();
        let Some(position) = target else {
            if self.shown.take().is_some() {
                if let Some(node) = &self.node
                    && scene.has_node(node.id())
                {
                    scene.set_node_visibility(node.id(), Visibility::Invisible);
                }
            }
            return;
        };

        if self.shown == Some(position) {
            return;
        }

        let node = self.ensure_node(&mut scene);
        scene.set_node_transform(node, Transform::from_position(position));
        scene.set_node_visibility(node, Visibility::Visible);
        self.shown = Some(position);
    }

    /// Sets the dot color. Takes effect immediately if the node already exists,
    /// otherwise it is applied when the node is first created.
    pub fn _set_color(&mut self, color: RgbaColor, scene: &Scene) {
        self.color = color;
        if let Some(material) = &self.material {
            if let Some(mat) = scene.lock().get_face_material_mut(material.id()) {
                mat.set_base_color_factor(color);
            }
        }
    }

    /// Returns the marker node, creating its mesh/material/texture/node on first
    /// use.
    fn ensure_node(&mut self, scene: &mut SceneData) -> NodeId {
        // A marker from a previous scene (after a scene swap) is stale; rebuild.
        if let Some(node) = &self.node
            && !scene.has_node(node.id())
        {
            self.node = None;
            self.material = None;
            self.shown = None;
        }
        if let Some(node) = &self.node {
            return node.id();
        }
        let mesh = scene.add_mesh(Mesh::quad(1.0, 1.0, PrimitiveType::TriangleList));
        let texture = scene.add_texture(build_dot_texture());
        let material = scene.add_face_material(
            FaceMaterial::new()
                .with_base_color_texture(texture)
                // Tint the white disc to the requested color.
                .with_base_color_factor(self.color)
                // Blend the anti-aliased / transparent dot edge.
                .with_alpha_mode(AlphaMode::Blend)
                .with_flags(MaterialFlags::DO_NOT_LIGHT | MaterialFlags::DOUBLE_SIDED),
        );
        let node = scene
            .add_instance_node(
                None,
                Instance::new(mesh).with_face_material(material.clone()),
                Some("3D cursor".to_owned()),
                Transform::IDENTITY,
                // Inert: not selectable (keeps it out of geometry snapping), not
                // exported, and excluded from scene bounds.
                NodeFlags::inert(),
            )
            .expect("Failed to create 3D cursor node");
        // Billboard, constant pixel size, always drawn on top.
        scene.set_node_display(
            node.id(),
            DisplayBehavior {
                screen_facing: true,
                screen_size: Some(CURSOR_PIXELS),
                layer: RenderLayer::Overlay,
            },
        );
        let id = node.id();
        self.material = Some(material);
        self.node = Some(node);
        id
    }
}

/// Builds a white, anti-aliased filled disc on a transparent background. The
/// color is left white so the material's base-color factor can tint it to any
/// hue; only the alpha channel carries the disc shape.
fn build_dot_texture() -> Texture {
    let size = DOT_TEXTURE_SIZE;
    let center = (size as f32 - 1.0) / 2.0;
    // Leave a one-texel margin so the disc edge sits inside the texture.
    let radius = center - 1.0;

    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            // Linear alpha ramp from 1 (inside) to 0 over the edge band.
            let alpha = ((radius - dist) / DOT_EDGE_TEXELS).clamp(0.0, 1.0);
            pixels.extend_from_slice(&[255, 255, 255, (alpha * 255.0).round() as u8]);
        }
    }
    Texture::from_rgba8(size, size, pixels)
}
