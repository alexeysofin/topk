use super::ElementCounter;
use ahash::RandomState;
use hashbrown::HashTable;
use std::hash::{BuildHasher, Hash, Hasher};

pub(super) const MAX_ITEMS: usize = u16::MAX as usize + 1;

/// Keys and counters live in stable slots. Only two-byte slot numbers are
/// stored in the lookup table and heap; keys and hashes are not duplicated.
pub(super) struct MonitoredList<T: Eq + Hash> {
    entries: Vec<(T, ElementCounter)>,
    heap: Vec<u16>,
    positions: Vec<u16>,
    lookup: HashTable<u16>,
    hash_builder: RandomState,
}

impl<T: Eq + Hash + Clone> Clone for MonitoredList<T> {
    fn clone(&self) -> Self {
        // Vec::clone drops spare capacity. Preserve the original reservation so
        // a partially filled clone does not later grow beyond its intended k.
        let capacity = self.entries.capacity();
        let mut entries = Vec::with_capacity(capacity);
        entries.extend(self.entries.iter().cloned());
        let mut heap = Vec::with_capacity(capacity);
        heap.extend_from_slice(&self.heap);
        let mut positions = Vec::with_capacity(capacity);
        positions.extend_from_slice(&self.positions);
        Self {
            entries,
            heap,
            positions,
            lookup: self.lookup.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<T: Eq + Hash> MonitoredList<T> {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity <= MAX_ITEMS, "k must be at most {MAX_ITEMS}");
        Self {
            entries: Vec::with_capacity(capacity),
            heap: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            lookup: HashTable::with_capacity(capacity),
            hash_builder: RandomState::default(),
        }
    }

    fn hash_key(hash_builder: &RandomState, key: &T) -> u64 {
        let mut hasher = hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn find_slot(&self, key: &T) -> Option<u16> {
        let hash = Self::hash_key(&self.hash_builder, key);
        self.lookup.find(hash, |&slot| self.entries[usize::from(slot)].0 == *key).copied()
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn get(&self, key: &T) -> Option<(&T, &ElementCounter)> {
        let (key, counter) = &self.entries[usize::from(self.find_slot(key)?)];
        Some((key, counter))
    }

    pub(super) fn contains(&self, key: &T) -> bool {
        self.find_slot(key).is_some()
    }

    pub(super) fn peek(&self) -> Option<(&T, &ElementCounter)> {
        let (key, counter) = &self.entries[usize::from(*self.heap.first()?)];
        Some((key, counter))
    }

    pub(super) fn increase(&mut self, key: &T, count: u64) -> bool {
        let slot = match self.find_slot(key) {
            Some(slot) => usize::from(slot),
            None => return false,
        };
        let previous = self.entries[slot].1;
        self.entries[slot].1.increase(count);
        let position = usize::from(self.positions[slot]);
        if self.entries[slot].1 < previous {
            // Truncation can turn an increment into a smaller count.
            self.sift_up(position);
        } else {
            self.sift_down(position);
        }
        true
    }

    pub(super) fn push(&mut self, key: T, counter: ElementCounter) {
        assert!(self.len() < self.entries.capacity(), "monitored list is full");
        debug_assert!(!self.contains(&key));
        let slot = u16::try_from(self.len()).expect("too many monitored items");
        let hash = Self::hash_key(&self.hash_builder, &key);
        self.entries.push((key, counter));
        let entries = &self.entries;
        let hash_builder = &self.hash_builder;
        self.lookup.insert_unique(hash, slot, |&slot| {
            Self::hash_key(hash_builder, &entries[usize::from(slot)].0)
        });
        self.positions.push(slot);
        self.heap.push(slot);
        self.sift_up(self.heap.len() - 1);
    }

    pub(super) fn replace_min(&mut self, key: T, counter: ElementCounter) -> (T, ElementCounter) {
        debug_assert!(!self.contains(&key));
        let slot = self.heap[0];
        let old_hash = Self::hash_key(&self.hash_builder, &self.entries[usize::from(slot)].0);
        self.lookup.find_entry(old_hash, |&candidate| candidate == slot).unwrap().remove();
        let hash = Self::hash_key(&self.hash_builder, &key);
        let expelled = std::mem::replace(&mut self.entries[usize::from(slot)], (key, counter));
        // Deleted buckets can exhaust growth_left even though the number of
        // live keys stays fixed. Hashbrown would then enlarge the table. Clear
        // its tombstones and reinsert the live slots into the same allocation.
        if self.lookup.capacity() < self.entries.len() {
            self.rebuild_lookup();
        } else {
            let entries = &self.entries;
            let hash_builder = &self.hash_builder;
            self.lookup.insert_unique(hash, slot, |&slot| {
                Self::hash_key(hash_builder, &entries[usize::from(slot)].0)
            });
        }
        self.sift_down(0);
        expelled
    }

    fn rebuild_lookup(&mut self) {
        self.lookup.clear();
        let entries = &self.entries;
        let hash_builder = &self.hash_builder;
        for (slot, (key, _)) in entries.iter().enumerate() {
            let hash = Self::hash_key(hash_builder, key);
            self.lookup.insert_unique(hash, slot as u16, |&slot| {
                Self::hash_key(hash_builder, &entries[usize::from(slot)].0)
            });
        }
    }

    fn less(&self, a: usize, b: usize) -> bool {
        let a_slot = self.heap[a];
        let b_slot = self.heap[b];
        let a_counter = self.entries[usize::from(a_slot)].1;
        let b_counter = self.entries[usize::from(b_slot)].1;
        // Resolve equal counters by stable slot so rebuilding after a serde
        // roundtrip cannot change which equally ranked key is expelled next.
        a_counter < b_counter || (a_counter == b_counter && a_slot < b_slot)
    }

    fn swap_heap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[usize::from(self.heap[a])] = a as u16;
        self.positions[usize::from(self.heap[b])] = b as u16;
    }

    fn sift_up(&mut self, mut position: usize) {
        while position != 0 {
            let parent = (position - 1) / 2;
            if !self.less(position, parent) {
                break;
            }
            self.swap_heap(position, parent);
            position = parent;
        }
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let left = 2 * position + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let child =
                if right < self.heap.len() && self.less(right, left) { right } else { left };
            if !self.less(child, position) {
                break;
            }
            self.swap_heap(position, child);
            position = child;
        }
    }

    /// Call after changing multiple counters through iter_mut.
    pub(super) fn rebuild_heap(&mut self) {
        for position in (0..self.heap.len() / 2).rev() {
            self.sift_down(position);
        }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&T, &ElementCounter)> {
        self.entries.iter().map(|(key, counter)| (key, counter))
    }

    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = (&T, &mut ElementCounter)> {
        self.entries.iter_mut().map(|(key, counter)| (&*key, counter))
    }

    pub(super) fn into_iter(self) -> impl Iterator<Item = (T, ElementCounter)> {
        self.entries.into_iter()
    }

    pub(super) fn into_sorted_vec(mut self) -> Vec<(T, ElementCounter)> {
        self.entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        self.entries
    }

    pub(super) fn clear(&mut self) {
        self.lookup.clear();
        self.heap.clear();
        self.positions.clear();
        self.entries.clear();
    }

    #[cfg(feature = "serde")]
    pub(super) fn capacity(&self) -> usize {
        self.entries.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn assert_heap<T: Eq + Hash>(list: &MonitoredList<T>) {
        assert_eq!(list.entries.len(), list.heap.len());
        assert_eq!(list.entries.len(), list.positions.len());
        assert_eq!(list.entries.len(), list.lookup.len());
        for (position, &slot) in list.heap.iter().enumerate() {
            assert_eq!(usize::from(list.positions[usize::from(slot)]), position);
            assert_eq!(list.find_slot(&list.entries[usize::from(slot)].0), Some(slot));
            if position != 0 {
                assert!(!list.less(position, (position - 1) / 2));
            }
        }
    }

    #[test]
    fn tie_breaking_and_rebuilding() {
        let mut list = MonitoredList::new(4);
        list.push(0, ElementCounter::new(10, 1));
        list.push(1, ElementCounter::new(10, 5));
        list.push(2, ElementCounter::new(20, 0));
        list.push(3, ElementCounter::new(30, 0));
        assert_eq!(*list.peek().unwrap().0, 1);
        list.increase(&1, 100);
        assert_eq!(*list.peek().unwrap().0, 0);
        for (key, counter) in list.iter_mut() {
            counter.estimated_count = 100 - *key;
        }
        list.rebuild_heap();
        assert_eq!(*list.peek().unwrap().0, 3);
        assert_heap(&list);
        let sorted = list.into_sorted_vec();
        assert!(sorted.windows(2).all(|pair| pair[0].1 >= pair[1].1));
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct CollisionKey(u64);

    impl Hash for CollisionKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            0u8.hash(state);
        }
    }

    #[test]
    fn colliding_keys_remain_distinct_after_replacements() {
        let mut list = MonitoredList::new(8);
        for key in 0..8 {
            list.push(CollisionKey(key), ElementCounter::new(key + 1, 0));
        }
        for key in 8..100 {
            let expelled = list.replace_min(CollisionKey(key), ElementCounter::new(key + 1, 0));
            assert!(list.get(&expelled.0).is_none());
            assert_eq!(
                u64::from(list.get(&CollisionKey(key)).unwrap().1.estimated_count()),
                key + 1
            );
            assert_heap(&list);
        }
        assert!(list.increase(&CollisionKey(99), 100));
        assert_heap(&list);
    }

    #[test]
    fn replacing_keys_keeps_lookup_allocation_fixed() {
        // Fill the hash table to its available capacity, forcing deletion
        // cleanup during replacements instead of relying on spare buckets.
        let capacity = MonitoredList::<u64>::new(1000).lookup.capacity();
        let mut list = MonitoredList::new(capacity);
        for key in 0..capacity as u64 {
            list.push(key, ElementCounter::new(key, 0));
        }
        for key in capacity as u64..capacity as u64 + 10_000 {
            list.replace_min(key, ElementCounter::new(key, 0));
            assert!(list.lookup.capacity() <= capacity);
        }
        list.lookup.clear();
        assert_eq!(list.lookup.capacity(), capacity);
        list.rebuild_lookup();
        assert_heap(&list);
    }

    #[test]
    fn clone_and_clear_preserve_reservations() {
        let mut list = MonitoredList::new(1000);
        list.push(1u64, ElementCounter::new(10, 0));
        let mut cloned = list.clone();
        assert_eq!(cloned.entries.capacity(), 1000);
        assert_eq!(cloned.heap.capacity(), 1000);
        assert_eq!(cloned.positions.capacity(), 1000);
        cloned.clear();
        assert!(cloned.get(&1).is_none());
        for key in 0..1000 {
            cloned.push(key, ElementCounter::new(key, 0));
        }
        assert_eq!(cloned.entries.capacity(), 1000);
        assert_eq!(cloned.heap.capacity(), 1000);
        assert_eq!(cloned.positions.capacity(), 1000);
        assert_heap(&cloned);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn largest_u16_slot_and_position_are_supported() {
        let mut list = MonitoredList::new(MAX_ITEMS);
        for key in 0..MAX_ITEMS as u64 {
            list.push(key, ElementCounter::new(key + 1, 0));
        }
        assert_eq!(list.find_slot(&(MAX_ITEMS as u64 - 1)), Some(u16::MAX));
        assert!(list.increase(&0, MAX_ITEMS as u64));
        assert_eq!(list.replace_min(MAX_ITEMS as u64, ElementCounter::new(100_000, 0)).0, 1);
        assert_heap(&list);
    }

    proptest! {
        #[test]
        fn updates_and_evictions_match_a_reference_map(
            updates in prop::collection::vec((0u64..64, 0u64..5000), 1..200)
        ) {
            let mut list = MonitoredList::new(16);
            let mut reference = HashMap::<u64, ElementCounter>::new();
            for (key, count) in updates {
                if let Some(counter) = reference.get_mut(&key) {
                    counter.increase(count);
                    prop_assert!(list.increase(&key, count));
                } else {
                    let counter = ElementCounter::new(count, 0);
                    if list.len() < 16 {
                        list.push(key, counter);
                    } else {
                        let (expelled, old) = list.replace_min(key, counter);
                        prop_assert_eq!(old, *reference.values().min().unwrap());
                        prop_assert_eq!(reference.remove(&expelled), Some(old));
                        prop_assert!(list.get(&expelled).is_none());
                    }
                    reference.insert(key, counter);
                }
                assert_heap(&list);
                prop_assert_eq!(list.peek().unwrap().1, reference.values().min().unwrap());
                for (key, expected) in &reference {
                    prop_assert_eq!(list.get(key).unwrap().1, expected);
                }
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<T: Eq + Hash + serde::Serialize> serde::Serialize for MonitoredList<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)]
        struct Snapshot<'a, T> {
            capacity: usize,
            entries: &'a [(T, ElementCounter)],
        }
        Snapshot { capacity: self.entries.capacity(), entries: &self.entries }.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Eq + Hash + serde::Deserialize<'de>> serde::Deserialize<'de> for MonitoredList<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Snapshot<T> {
            capacity: usize,
            entries: Vec<(T, ElementCounter)>,
        }
        let Snapshot { capacity, mut entries } = Snapshot::<T>::deserialize(deserializer)?;
        if capacity > MAX_ITEMS || entries.len() > capacity {
            return Err(serde::de::Error::custom("invalid monitored capacity"));
        }
        // Keep the configured reservation when restoring a partially filled
        // sketch, and trim spare capacity from the deserializer's temporary Vec.
        if entries.capacity() > capacity {
            entries.shrink_to_fit();
        }
        entries.reserve_exact(capacity - entries.len());
        let mut list = Self {
            entries,
            heap: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            lookup: HashTable::with_capacity(capacity),
            hash_builder: RandomState::default(),
        };
        for (slot, (key, _)) in list.entries.iter().enumerate() {
            let hash = Self::hash_key(&list.hash_builder, key);
            if list.lookup.find(hash, |&slot| list.entries[usize::from(slot)].0 == *key).is_some() {
                return Err(serde::de::Error::custom("duplicate monitored key"));
            }
            let slot = slot as u16;
            list.lookup.insert_unique(hash, slot, |&slot| {
                Self::hash_key(&list.hash_builder, &list.entries[usize::from(slot)].0)
            });
            list.heap.push(slot);
            list.positions.push(slot);
        }
        list.rebuild_heap();
        Ok(list)
    }
}
