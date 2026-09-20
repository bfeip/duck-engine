use crate::{EuclideanSpace, Matrix4, Point3, Quaternion, Vector3};

/// A decomposed 3D transform consisting of position, rotation, and scale (TRS order).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform {
    pub position: Point3,
    pub rotation: Quaternion,
    pub scale: Vector3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        position: Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        rotation: Quaternion {
            s: 1.0,
            v: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
        scale: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    };

    pub fn new(position: Point3, rotation: Quaternion, scale: Vector3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn from_position(position: Point3) -> Self {
        Self {
            position,
            ..Self::IDENTITY
        }
    }

    pub fn from_rotation(rotation: Quaternion) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    pub fn from_scale(scale: Vector3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Computes the 4x4 transformation matrix (Translation * Rotation * Scale).
    pub fn to_matrix(&self) -> Matrix4 {
        let translation = Matrix4::from_translation(self.position.to_vec());
        let rotation = Matrix4::from(self.rotation);
        let scale =
            Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);

        translation * rotation * scale
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SquareMatrix, Point3, Quaternion, Vector3};
    use cgmath::assert_relative_eq;

    #[test]
    fn test_identity() {
        let t = Transform::IDENTITY;
        assert_eq!(t.position, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(t.rotation, Quaternion::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(t.scale, Vector3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_identity_matrix() {
        let matrix = Transform::IDENTITY.to_matrix();
        assert_relative_eq!(matrix, Matrix4::from_value(1.0), epsilon = 1e-6);
    }

    #[test]
    fn test_from_position() {
        let t = Transform::from_position(Point3::new(1.0, 2.0, 3.0));
        assert_eq!(t.position, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(t.rotation, Quaternion::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(t.scale, Vector3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_default() {
        assert_eq!(Transform::default(), Transform::IDENTITY);
    }
}
