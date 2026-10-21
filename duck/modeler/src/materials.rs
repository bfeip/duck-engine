//! The document's material library: named face materials that body parts
//! share, so restyling a material restyles every part it is assigned to.

use duck_engine_scene::common::RgbaColor;
use duck_engine_scene::resource::{FaceMaterial, FaceMaterialHandle, FaceMaterialId, MaterialFlags};

/// The look of a body's faces. Opacity is the color's alpha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Finish {
    pub color: RgbaColor,
    /// 0 for a dielectric, 1 for a metal.
    pub metallic: f32,
    /// 0 for a mirror, 1 for fully diffuse.
    pub roughness: f32,
}

impl Finish {
    pub fn of(material: &FaceMaterial) -> Self {
        Self {
            color: material.base_color_factor(),
            metallic: material.metallic_factor(),
            roughness: material.roughness_factor(),
        }
    }

    /// Writes this finish over `material`'s.
    pub fn apply_to(self, material: &mut FaceMaterial) {
        material.set_base_color_factor(self.color);
        material.set_metallic_factor(self.metallic);
        material.set_roughness_factor(self.roughness);
    }

    /// A new face material with this finish. Double sided: open shells are
    /// bodies too, and sheets are built with arbitrary orientation.
    pub fn material(self) -> FaceMaterial {
        FaceMaterial::new()
            .with_base_color_factor(self.color)
            .with_metallic_factor(self.metallic)
            .with_roughness_factor(self.roughness)
            .with_flags(MaterialFlags::DOUBLE_SIDED)
    }
}

/// The finish new bodies are built with.
pub const DEFAULT_FINISH: Finish = Finish {
    color: RgbaColor { r: 0.34, g: 0.40, b: 0.52, a: 1.0 },
    metallic: 0.0,
    roughness: 0.32,
};

/// The materials a new library starts with, in linear RGB.
pub const PRESETS: [(&str, Finish); 9] = [
    ("Slate", DEFAULT_FINISH),
    ("Steel", metal(0.56, 0.57, 0.58, 0.35)),
    ("Aluminium", metal(0.91, 0.92, 0.92, 0.3)),
    ("Brass", metal(0.91, 0.78, 0.42, 0.3)),
    ("Copper", metal(0.95, 0.64, 0.54, 0.3)),
    ("ABS White", dielectric(0.80, 0.80, 0.78, 1.0, 0.45)),
    ("ABS Black", dielectric(0.03, 0.03, 0.03, 1.0, 0.45)),
    ("Rubber", dielectric(0.02, 0.02, 0.02, 1.0, 0.9)),
    ("Glass", dielectric(0.85, 0.92, 0.95, 0.25, 0.05)),
];

const fn metal(r: f32, g: f32, b: f32, roughness: f32) -> Finish {
    Finish { color: RgbaColor { r, g, b, a: 1.0 }, metallic: 1.0, roughness }
}

const fn dielectric(r: f32, g: f32, b: f32, a: f32, roughness: f32) -> Finish {
    Finish { color: RgbaColor { r, g, b, a }, metallic: 0.0, roughness }
}

/// A named material in the library. Its id is its scene material's.
#[derive(Clone)]
pub struct LibraryMaterial {
    pub name: String,
    handle: FaceMaterialHandle,
}

impl LibraryMaterial {
    pub fn new(name: impl Into<String>, handle: FaceMaterialHandle) -> Self {
        Self { name: name.into(), handle }
    }

    pub fn id(&self) -> FaceMaterialId {
        self.handle.id()
    }

    pub fn handle(&self) -> &FaceMaterialHandle {
        &self.handle
    }
}

/// The library's materials, in the order they are listed.
#[derive(Default)]
pub struct MaterialLibrary {
    materials: Vec<LibraryMaterial>,
}

impl MaterialLibrary {
    pub fn iter(&self) -> impl Iterator<Item = &LibraryMaterial> {
        self.materials.iter()
    }

    pub fn get(&self, id: FaceMaterialId) -> Option<&LibraryMaterial> {
        self.materials.iter().find(|m| m.id() == id)
    }

    pub fn get_mut(&mut self, id: FaceMaterialId) -> Option<&mut LibraryMaterial> {
        self.materials.iter_mut().find(|m| m.id() == id)
    }

    pub fn contains(&self, id: FaceMaterialId) -> bool {
        self.get(id).is_some()
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Lists `material` at `index`, or last if the library is shorter.
    pub fn insert(&mut self, index: usize, material: LibraryMaterial) {
        let index = index.min(self.materials.len());
        self.materials.insert(index, material);
    }

    pub fn push(&mut self, material: LibraryMaterial) {
        self.materials.push(material);
    }

    /// Unlists the material, returning where it was listed.
    pub fn remove(&mut self, id: FaceMaterialId) -> Option<(usize, LibraryMaterial)> {
        let index = self.materials.iter().position(|m| m.id() == id)?;
        Some((index, self.materials.remove(index)))
    }

    pub fn clear(&mut self) {
        self.materials.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_finish_round_trips_through_a_material() {
        for (name, finish) in PRESETS {
            assert_eq!(Finish::of(&finish.material()), finish, "{name}");
        }
    }

    #[test]
    fn applying_a_finish_raises_the_generation() {
        let mut material = DEFAULT_FINISH.material();
        let generation = material.generation();
        PRESETS[1].1.apply_to(&mut material);
        assert!(material.generation() > generation);
        assert_eq!(Finish::of(&material), PRESETS[1].1);
    }

    #[test]
    fn removing_reports_where_the_material_was_listed() {
        let mut library = MaterialLibrary::default();
        let materials: Vec<_> = (0..3)
            .map(|i| LibraryMaterial::new(format!("m{i}"), FaceMaterialHandle::unbound(FaceMaterialId::new())))
            .collect();
        for material in &materials {
            library.push(material.clone());
        }

        let (index, removed) = library.remove(materials[1].id()).expect("listed");
        assert_eq!((index, removed.name.as_str()), (1, "m1"));
        assert!(!library.contains(materials[1].id()));

        library.insert(index, removed);
        let names: Vec<_> = library.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["m0", "m1", "m2"]);
    }
}
