use std::collections::HashMap;
use std::hash::Hash;

/// A generation-synced GPU resource cache.
///
/// Each entry pairs a resource with the generation counter of the source data
/// it was built from. A resource is stale when the source's current generation
/// differs from the synced one; [`ensure`](Self::ensure) rebuilds it then.
/// This is the engine's standard alternative to boolean dirty flags.
///
/// # Example
///
/// ```
/// use duck_engine_render_core::GenCache;
///
/// struct FakeBuffer(u64);
///
/// let mut cache: GenCache<u32, FakeBuffer> = GenCache::new();
///
/// // First call uploads; second call with the same generation is a no-op.
/// cache.ensure(7, 1, || FakeBuffer(1));
/// cache.ensure(7, 1, || panic!("must not rebuild"));
///
/// // Source mutated (generation bumped): the resource is rebuilt.
/// let buf = cache.ensure(7, 2, || FakeBuffer(2));
/// assert_eq!(buf.0, 2);
/// ```
pub struct GenCache<K, R> {
    entries: HashMap<K, Entry<R>>,
}

struct Entry<R> {
    resource: R,
    synced_generation: u64,
}

impl<K: Copy + Eq + Hash, R> GenCache<K, R> {
    #[must_use] 
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Whether the resource for `key` is missing or was built from a different
    /// generation of the source data.
    pub fn needs_upload(&self, key: K, generation: u64) -> bool {
        match self.entries.get(&key) {
            None => true,
            Some(entry) => entry.synced_generation != generation,
        }
    }

    /// The cached resource for `key`, regardless of generation.
    pub fn get(&self, key: K) -> Option<&R> {
        self.entries.get(&key).map(|e| &e.resource)
    }

    /// Store `resource` as synced to `generation`, replacing any previous entry.
    pub fn insert(&mut self, key: K, resource: R, generation: u64) {
        self.entries.insert(key, Entry { resource, synced_generation: generation });
    }

    /// Get the resource for `key`, building it via `build` if it is missing or
    /// stale relative to `generation`.
    pub fn ensure(&mut self, key: K, generation: u64, build: impl FnOnce() -> R) -> &R {
        if self.needs_upload(key, generation) {
            self.insert(key, build(), generation);
        }
        &self.entries[&key].resource
    }

    /// Fallible variant of [`ensure`](Self::ensure). On build failure the
    /// previous entry (if any) is left untouched.
    /// 
    /// # Errors
    /// 
    /// Will return `Err` if when a upload is needed, `build` returns `Err`.
    pub fn try_ensure(
        &mut self,
        key: K,
        generation: u64,
        build: impl FnOnce() -> anyhow::Result<R>,
    ) -> anyhow::Result<&R> {
        if self.needs_upload(key, generation) {
            self.insert(key, build()?, generation);
        }
        Ok(&self.entries[&key].resource)
    }

    /// Drop the entry for `key`, if any, returning its resource.
    ///
    /// Call when the source for `key` is removed, so its GPU resource is
    /// released instead of lingering until [`clear`](Self::clear).
    pub fn remove(&mut self, key: K) -> Option<R> {
        self.entries.remove(&key).map(|e| e.resource)
    }

    /// Drop all entries.
    ///
    /// Call when the source collection is cleared or replaced, so stale
    /// resources are not reused when new sources recycle the same keys with
    /// matching generation numbers.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<K: Copy + Eq + Hash, R> Default for GenCache<K, R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_key_needs_upload() {
        let cache: GenCache<u32, String> = GenCache::new();
        assert!(cache.needs_upload(1, 0));
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn matching_generation_is_synced() {
        let mut cache = GenCache::new();
        cache.insert(1, "a".to_string(), 5);
        assert!(!cache.needs_upload(1, 5));
        assert!(cache.needs_upload(1, 6));
        assert_eq!(cache.get(1).unwrap(), "a");
    }

    #[test]
    fn ensure_builds_once_per_generation() {
        let mut cache = GenCache::new();
        let mut builds = 0;
        cache.ensure(1, 1, || { builds += 1; "a" });
        cache.ensure(1, 1, || { builds += 1; "b" });
        assert_eq!(builds, 1);
        assert_eq!(*cache.get(1).unwrap(), "a");

        cache.ensure(1, 2, || { builds += 1; "c" });
        assert_eq!(builds, 2);
        assert_eq!(*cache.get(1).unwrap(), "c");
    }

    #[test]
    fn try_ensure_keeps_previous_entry_on_failure() {
        let mut cache = GenCache::new();
        cache.insert(1, "a", 1);
        let result = cache.try_ensure(1, 2, || anyhow::bail!("build failed"));
        assert!(result.is_err());
        assert_eq!(*cache.get(1).unwrap(), "a");
        assert!(cache.needs_upload(1, 2));
    }

    #[test]
    fn remove_drops_single_entry() {
        let mut cache = GenCache::new();
        cache.insert(1, "a", 1);
        cache.insert(2, "b", 1);
        assert_eq!(cache.remove(1), Some("a"));
        assert_eq!(cache.remove(1), None);
        assert!(cache.needs_upload(1, 1));
        assert!(!cache.needs_upload(2, 1));
    }

    #[test]
    fn clear_drops_all_entries() {
        let mut cache = GenCache::new();
        cache.insert(1, "a", 1);
        cache.insert(2, "b", 1);
        cache.clear();
        assert!(cache.needs_upload(1, 1));
        assert!(cache.needs_upload(2, 1));
    }
}
