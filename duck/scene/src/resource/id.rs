use std::marker::PhantomData;

/// Opaque unique identifier used for all scene resources.
///
/// Backed by UUID v7 (time-ordered) for stable, globally-unique IDs that merge
/// cleanly across scenes and processes. The underlying representation is an
/// implementation detail — callers use `Id::new()` and `Id::nil()` only.
///
/// The `Kind` type parameter makes ids of different resource kinds
/// non-interchangeable at compile time (e.g. `Id<Mesh>` vs `Id<Node>`). The
/// default `Kind = ()` yields the original untyped `Id` for genuinely-generic
/// sites.
///
// The phantom is `fn() -> Kind`, not `Kind`, on purpose: it is unconditionally
// `Send + Sync + Copy` and carries no ownership of `Kind`. That (a) decouples an
// id's auto-traits from the marker type, and (b) lets a type hold its own id
// (`FaceMaterial { id: Id<FaceMaterial> }`) without creating an auto-trait /
// dropck cycle through the marker. It stays covariant in `Kind`, like
// `PhantomData<Kind>` would.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent, bound = ""))]
pub struct Id<Kind = ()>(
    uuid::Uuid,
    #[cfg_attr(feature = "serde", serde(skip))] PhantomData<fn() -> Kind>,
);

/// An id with its `Kind` parameter dropped.
///
/// Every `Id<Kind>` shares the same underlying UUID representation, so a
/// `GenericId` can stand in wherever the compile-time kind distinction is not
/// needed — e.g. as a map key in code that is generic over resource kind.
/// Obtain one with [`Id::erased`].
pub type GenericId = Id<()>;

impl<Kind> Id<Kind> {
    /// Creates a new globally-unique ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7(), PhantomData)
    }

    /// Returns the nil (all-zeros) ID. Useful as a sentinel or default value.
    pub fn nil() -> Self {
        Self(uuid::Uuid::nil(), PhantomData)
    }

    /// Returns true if this is the nil ID.
    pub fn is_nil(&self) -> bool {
        self.0.is_nil()
    }

    /// Drops the `Kind` parameter, yielding a [`GenericId`].
    ///
    /// Useful for maps and keys that don't need the compile-time distinction,
    /// without compromising the typed public API.
    pub fn erased(self) -> GenericId {
        self.cast()
    }

    /// Re-parameterizes the id to a different `Kind`, preserving the underlying
    /// UUID.
    ///
    /// The kind tag is purely a compile-time label, so this is always sound at
    /// runtime — but it deliberately bypasses the type distinction, so it should
    /// only be used at boundaries that genuinely erase and later restore the kind
    /// (e.g. (de)serialization where a generic id is stored in a table of
    /// contents and re-typed on load). Prefer typed ids everywhere else.
    pub fn cast<U>(self) -> Id<U> {
        Id(self.0, PhantomData)
    }
}

// Hand-written trait impls so `Kind` carries no spurious bounds (deriving would
// inject e.g. `Kind: Clone`). All logic delegates to the inner `Uuid`.

impl<Kind> Clone for Id<Kind> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Kind> Copy for Id<Kind> {}

impl<Kind> PartialEq for Id<Kind> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<Kind> Eq for Id<Kind> {}

impl<Kind> std::hash::Hash for Id<Kind> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<Kind> PartialOrd for Id<Kind> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Kind> Ord for Id<Kind> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<Kind> std::fmt::Debug for Id<Kind> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({})", self.0)
    }
}

impl<Kind> std::fmt::Display for Id<Kind> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
