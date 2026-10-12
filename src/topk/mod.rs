#[cfg(not(feature = "serde"))]
use ahash::AHasher;
#[cfg(feature = "serde")]
use rustc_stable_hash::StableSipHasher128;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// Default hasher for alpha buckets. Uses stable hash when serde is enabled
/// for cross-platform compatibility; uses fast AHasher otherwise.
#[cfg(feature = "serde")]
pub type DefaultAlphaHasher = StableSipHasher128;

/// Default hasher for alpha buckets. Uses stable hash when serde is enabled
/// for cross-platform compatibility; uses fast AHasher otherwise.
#[cfg(not(feature = "serde"))]
pub type DefaultAlphaHasher = AHasher;

/// A counter for element occurrences with associated error (if present).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElementCounter {
    estimated_count: u64,
    associated_error: u64,
}

impl ElementCounter {
    fn new(estimated_count: u64, associated_error: u64) -> Self {
        ElementCounter { estimated_count, associated_error }
    }

    /// Returns the estimated element occurrence count.
    pub fn estimated_count(&self) -> u64 {
        self.estimated_count
    }

    /// Returns the associated occurrence count error.
    pub fn associated_error(&self) -> u64 {
        self.associated_error
    }
}

impl Ord for ElementCounter {
    fn cmp(&self, other: &Self) -> Ordering {
        self.estimated_count
            .cmp(&other.estimated_count)
            .then(other.associated_error.cmp(&self.associated_error))
    }
}

impl PartialOrd for ElementCounter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

mod monitored;
use monitored::{MonitoredList, MAX_ITEMS};

const ALPHAS_FACTOR: usize = 6;

/// Fixed-size filter storage. Values below a new baseline are rounded up to it,
/// so rebasing can increase estimates but never silently wrap a counter.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct AlphaCounters {
    offset: u64,
    deltas: Vec<u8>,
}

impl AlphaCounters {
    fn new(len: usize) -> Self {
        Self { offset: 0, deltas: vec![0; len] }
    }

    fn len(&self) -> usize {
        self.deltas.len()
    }

    fn get(&self, index: usize) -> u64 {
        self.offset + u64::from(self.deltas[index])
    }

    fn set(&mut self, index: usize, value: u64) {
        let new_offset = self.offset.max(value.saturating_sub(u64::from(u8::MAX)));
        let shift = new_offset - self.offset;
        if shift != 0 {
            for delta in &mut self.deltas {
                // Keep subtraction wide: a large weighted update can move the
                // baseline by much more than 255 in a single operation.
                *delta = u64::from(*delta).saturating_sub(shift) as u8;
            }
            self.offset = new_offset;
        }
        self.deltas[index] = value.saturating_sub(self.offset) as u8;
    }

    fn raise(&mut self, index: usize, value: u64) {
        self.set(index, self.get(index).max(value));
    }

    fn merge(&mut self, other: &Self) {
        // Choose the final baseline first, so rebasing one bucket does not
        // change the inputs used to sum the remaining buckets.
        let old_offset = self.offset;
        let combined_offset = old_offset + other.offset;
        let max_delta = self
            .deltas
            .iter()
            .zip(&other.deltas)
            .map(|(&a, &b)| u64::from(a) + u64::from(b))
            .max()
            .unwrap_or(0);
        let new_offset = combined_offset + max_delta.saturating_sub(u64::from(u8::MAX));
        let shift = new_offset - combined_offset;
        for (delta, &other_delta) in self.deltas.iter_mut().zip(&other.deltas) {
            *delta = (u64::from(*delta) + u64::from(other_delta)).saturating_sub(shift) as u8;
        }
        self.offset = new_offset;
    }

    fn decay(&mut self, factor: f64) {
        let old_offset = self.offset;
        let scaled = |delta: u8| ((old_offset + u64::from(delta)) as f64 * factor) as u64;
        let max_value = self.deltas.iter().map(|&delta| scaled(delta)).max().unwrap_or(0);
        let new_offset = scaled(0).max(max_value.saturating_sub(u64::from(u8::MAX)));
        for delta in &mut self.deltas {
            *delta = scaled(*delta).saturating_sub(new_offset) as u8;
        }
        self.offset = new_offset;
    }

    fn clear(&mut self) {
        self.offset = 0;
        self.deltas.fill(0);
    }
}

/// A filtered space-saving structure containing the current Top-K elements.
///
/// The elements is of type T, which must implement `Eq` and `Hash`.
///
/// The hasher H is used for the alpha bucket hash function. The default is
/// `AHasher` for performance. When the `serde` feature is enabled, the default
/// is `StableSipHasher128` for cross-platform determinism. This reduces
/// performance of common operations by 20–25%.
///
/// The space-saving algorithm guarantees the following:
/// 1. `estimated_count` >= `exact_count`
/// 2. `estimated_count` - `associated_error` <= `exact_count`
///
/// Filter buckets use one byte each plus a shared `u64` offset. When their
/// spread exceeds 255, low bucket values are rounded up to the new offset.
/// This keeps filter storage fixed but can increase estimates and change the
/// returned Top-K. Rebasing takes O(k) time. With serde, the filter is serialized
/// as an offset and byte deltas, rather than the previous array of u64 values.
///
/// Monitored entries use a compact indexed heap with u16 slot numbers and heap
/// positions. The lookup table stores slot numbers without duplicating keys or
/// cached hashes. Capacity is limited to 65,536 monitored entries.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FilteredSpaceSaving<T: Eq + Hash, H: Hasher + Default = DefaultAlphaHasher> {
    k: usize,
    monitored_list: MonitoredList<T>,
    alphas: AlphaCounters,
    count: u64,
    #[cfg_attr(feature = "serde", serde(skip))]
    _hasher: PhantomData<H>,
}

/// Whether an item is monitored immediately after insertion. Later insertions
/// can still expel it; this is not a guarantee about the final Top-K.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InsertResult<T> {
    /// An already monitored item's count increased.
    Updated,
    /// The item did not enter the monitored list.
    Rejected,
    /// The item filled a free slot.
    Inserted,
    /// The item entered the list and expelled the returned key.
    Replaced { expelled: T },
}

#[cfg(feature = "serde")]
impl<'de, T: Eq + Hash + serde::Deserialize<'de>, H: Hasher + Default> serde::Deserialize<'de>
    for FilteredSpaceSaving<T, H>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Snapshot<T: Eq + Hash> {
            k: usize,
            monitored_list: MonitoredList<T>,
            alphas: AlphaCounters,
            count: u64,
        }
        let snapshot = Snapshot::<T>::deserialize(deserializer)?;
        if snapshot.k > MAX_ITEMS
            || snapshot.monitored_list.capacity() != snapshot.k
            || snapshot.alphas.len() != ALPHAS_FACTOR * snapshot.k
        {
            return Err(serde::de::Error::custom("inconsistent sketch capacity"));
        }
        let max_delta = snapshot.alphas.deltas.iter().copied().max().unwrap_or(0);
        if snapshot.alphas.offset.checked_add(u64::from(max_delta)).is_none() {
            return Err(serde::de::Error::custom("invalid alpha offset"));
        }
        Ok(Self {
            k: snapshot.k,
            monitored_list: snapshot.monitored_list,
            alphas: snapshot.alphas,
            count: snapshot.count,
            _hasher: PhantomData,
        })
    }
}

// `new` implicitly forces the use of DefaultAlphaHasher for convenience and
// backwards compatibility.
impl<T: Eq + Hash> FilteredSpaceSaving<T, DefaultAlphaHasher> {
    /// Creates an empty filtered space-saving structure with pre-allocated space for `k` elements.
    ///
    /// Panics if `k` exceeds 65,536, the capacity of the compact slot indices.
    ///
    /// Uses `DefaultAlphaHasher` for alpha bucket hashing, which is `AHasher` by default
    /// for performance, or `StableSipHasher128` when the `serde` feature is enabled for
    /// cross-platform determinism.
    pub fn new(k: usize) -> Self {
        Self::with_hasher(k)
    }
}

// Other methods inherit their default from the `FilteredSpaceSaving` struct or
// from context. We provide `with_hasher` to override the default hasher.
// This is the same pattern used by Rust's `HashMap`.
impl<T: Eq + Hash, H: Hasher + Default> FilteredSpaceSaving<T, H> {
    /// Creates an empty filtered space-saving structure with a custom hasher.
    ///
    /// Panics if `k` exceeds 65,536. A zero-capacity sketch rejects all inserts.
    pub fn with_hasher(k: usize) -> Self {
        assert!(k <= MAX_ITEMS, "k must be at most {MAX_ITEMS}");
        Self {
            k,
            monitored_list: MonitoredList::new(k),
            alphas: AlphaCounters::new(ALPHAS_FACTOR * k),
            count: 0,
            _hasher: PhantomData,
        }
    }

    /// Inserts the item `x` for `count` times.
    ///
    /// Returns its monitoring status after this update and any expelled item.
    ///
    /// Computes in **O(log(k))** time, or **O(k)** when the filter is rebased.
    pub fn insert(&mut self, x: T, count: u64) -> InsertResult<T> {
        self.count += count;

        if self.k == 0 {
            return InsertResult::Rejected;
        }
        if self.monitored_list.increase(&x, count) {
            return InsertResult::Updated;
        }
        if self.monitored_list.len() < self.k {
            self.monitored_list.push(x, ElementCounter::new(count, 0));
            return InsertResult::Inserted;
        }

        let x_hash = self.alpha_hash(&x);
        let (min_elem, min_counter) = self.monitored_list.peek().unwrap();
        if self.alphas.get(x_hash) + count < min_counter.estimated_count {
            self.alphas.set(x_hash, self.alphas.get(x_hash) + count);
            return InsertResult::Rejected;
        }

        let m_hash = self.alpha_hash(min_elem);
        self.alphas.raise(m_hash, min_counter.estimated_count);

        let (expelled, _) = self.monitored_list.replace_min(
            x,
            ElementCounter::new(self.alphas.get(x_hash) + count, self.alphas.get(x_hash)),
        );
        InsertResult::Replaced { expelled }
    }

    /// Estimates the occurrences of the item `x` if the item is in the Top-K approximation.
    ///
    /// Otherwise, `None` is returned.
    ///
    /// Computes in **O(1)** time.
    pub fn get(&self, x: &T) -> Option<ElementCounter> {
        self.monitored_list.get(x).map(|(_, counter)| *counter)
    }

    /// Estimates the occurrences of the item `x`.
    ///
    /// If the item is in the Top-K approximation, the approximation is returned.
    ///
    /// Otherwise, a rough approximation is returned (`error` == `estimate`).
    ///
    /// Computes in **O(1)** time.
    pub fn estimate(&self, x: &T) -> ElementCounter {
        self.get(x).unwrap_or_else(|| {
            let count = if self.k == 0 { self.count } else { self.alphas.get(self.alpha_hash(&x)) };
            ElementCounter::new(count, count)
        })
    }

    /// Merges with `other` filtered space-saving approximation.
    ///
    /// Require `T` to implement `Clone`.
    ///
    /// Merging of different `k` values will result in an `InvalidMergeError`.
    ///
    /// ref: <https://ieeexplore.ieee.org/document/8438445>
    ///
    /// Computes in **O(k*log(k))** time, or **O(k^2)** if repeated filter
    /// updates require rebasing.
    pub fn merge(&mut self, other: &FilteredSpaceSaving<T, H>) -> Result<(), InvalidMergeError>
    where
        T: Clone,
    {
        if self.k != other.k {
            return Err(InvalidMergeError { expect: self.k, actual: other.k });
        }
        self.count += other.count;
        for (key, value) in self.monitored_list.iter_mut() {
            if let Some((_, e)) = other.monitored_list.get(key) {
                value.estimated_count += e.estimated_count;
                value.associated_error += e.associated_error;
            } else {
                let k_hash = other.alpha_hash(key);
                let a2 = other.alphas.get(k_hash);
                value.estimated_count += a2;
                value.associated_error += a2;
            }
        }
        self.monitored_list.rebuild_heap();
        for (key, value) in other.monitored_list.iter() {
            if self.monitored_list.contains(key) {
                continue;
            }
            let k_hash = self.alpha_hash(key);
            let a1 = self.alphas.get(k_hash);
            let e = ElementCounter::new(value.estimated_count + a1, value.associated_error + a1);
            if self.monitored_list.len() < self.k {
                // We have fewer than k items, so add a new item.
                self.monitored_list.push(key.clone(), e);
            } else if self.monitored_list.peek().map_or(false, |(_, m)| *m < e) {
                // We want to evict an item and replace it with this one, but we
                // need to update the error accordingly.
                //
                // We use `max` here because the error for the popped key can't
                // be any larger than `estimated_count`. But we _do_ need to
                // update the `alphas` array because it may contain counted
                // items that were counted purely in `monitored_list` and never
                // in `alphas`.
                let popped = self.monitored_list.replace_min(key.clone(), e);
                let p_hash = self.alpha_hash(&popped.0);
                self.alphas.raise(p_hash, popped.1.estimated_count);
            } else {
                // This item is not in the Top-K and cannot replace any existing
                // item, but we still need to track the error.
                //
                // We use `max` here because the worst possible error is the
                // larger of the errors of the two inputs, not the sum.
                self.alphas.raise(k_hash, e.estimated_count);
            }
        }
        self.alphas.merge(&other.alphas);
        Ok(())
    }

    /// Decays the counters, equivalent to multiplying all counters by the factor.
    ///
    /// Computes in **O(k)** time.
    pub fn decay(&mut self, factor: f64) {
        for (_, value) in self.monitored_list.iter_mut() {
            value.estimated_count = (value.estimated_count as f64 * factor) as u64;
            value.associated_error = (value.associated_error as f64 * factor) as u64;
        }
        self.monitored_list.rebuild_heap();
        self.alphas.decay(factor);
        self.count = (self.count as f64 * factor) as u64
    }

    /// Clears the counter, resetting count to 0.
    ///
    /// Note that this method has no effect on the `k` of the counter.
    pub fn clear(&mut self) {
        self.monitored_list.clear();
        self.alphas.clear();
        self.count = 0;
    }

    /// Returns an iterator in arbitrary order over the Top-K items.
    pub fn iter(&self) -> impl Iterator<Item = (&T, &ElementCounter)> {
        self.monitored_list.iter()
    }

    /// Consumes the `FilteredSpaceSaving` and return an iterator in arbitrary order over the Top-K items.
    pub fn into_iter(self) -> impl Iterator<Item = (T, ElementCounter)> {
        self.monitored_list.into_iter()
    }

    /// Consumes the `FilteredSpaceSaving` and return a `Vec` with Top-K items and counters in descending order (top items first).
    ///
    /// Computes in **O(k*log(k))** time.
    pub fn into_sorted_vec(self) -> Vec<(T, ElementCounter)> {
        self.monitored_list.into_sorted_vec()
    }

    /// Consumes the `FilteredSpaceSaving` and return a `DoubleEndedIterator` with Top-K items and counters in descending order (top items first).
    ///
    /// Computes in **O(k*log(k))** time.
    pub fn into_sorted_iter(self) -> impl DoubleEndedIterator<Item = (T, ElementCounter)> {
        self.into_sorted_vec().into_iter()
    }

    /// Returns count of all seen items (sum of all inserted `count`).
    ///
    /// Computes in **O(1)** time.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the `k` value of the counter.
    pub fn k(&self) -> usize {
        self.k
    }

    fn alpha_hash(&self, x: &T) -> usize {
        let mut hasher = H::default();
        x.hash(&mut hasher);
        (hasher.finish() as u128 * self.alphas.len() as u128 >> 64) as usize
    }
}

#[derive(Debug, Clone)]
pub struct InvalidMergeError {
    expect: usize,
    actual: usize,
}

impl Display for InvalidMergeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "expected merge with same k {}, got {}", self.expect, self.actual)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_insert_reports_membership_and_expulsion() {
        let mut fss = FilteredSpaceSaving::new(1);
        assert_eq!(fss.insert("heavy", 1000), InsertResult::Inserted);
        assert_eq!(fss.insert("heavy", 10), InsertResult::Updated);
        assert_eq!(fss.insert("noise", 1), InsertResult::Rejected);
        assert!(fss.get(&"heavy").is_some());
        assert!(fss.get(&"noise").is_none());
        assert_eq!(fss.insert("replacement", 2000), InsertResult::Replaced { expelled: "heavy" });
        assert!(fss.get(&"heavy").is_none());
        assert!(fss.get(&"replacement").is_some());
        assert_eq!(fss.insert("replacement", 1), InsertResult::Updated);
    }

    #[test]
    fn test_zero_capacity_sketch() {
        let mut fss = FilteredSpaceSaving::new(0);
        assert_eq!(fss.insert(1, 100), InsertResult::Rejected);
        assert_eq!(fss.estimate(&1), ElementCounter::new(100, 100));
        assert_eq!(fss.iter().count(), 0);
        let mut other = FilteredSpaceSaving::new(0);
        other.insert(2, 10);
        fss.merge(&other).unwrap();
        assert_eq!(fss.estimate(&2), ElementCounter::new(110, 110));
        fss.clear();
        assert_eq!(fss.count(), 0);
    }

    #[test]
    #[should_panic(expected = "k must be at most")]
    fn test_capacity_exceeding_u16_indices_is_rejected() {
        let _ = FilteredSpaceSaving::<u64>::new(MAX_ITEMS + 1);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_rebuilds_index_and_heap_for_future_updates() {
        let mut fss = FilteredSpaceSaving::new(10);
        for key in 0..3u64 {
            fss.insert(key, 100 - key);
        }
        let serialized = serde_json::to_string(&fss).unwrap();
        let mut restored: FilteredSpaceSaving<u64> = serde_json::from_str(&serialized).unwrap();
        for key in 0..100 {
            assert_eq!(fss.insert(key, 1000), restored.insert(key, 1000));
        }
        assert_eq!(fss.into_sorted_vec(), restored.into_sorted_vec());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_rejects_inconsistent_storage() {
        let mut fss = FilteredSpaceSaving::new(2);
        fss.insert(1u64, 100);
        fss.insert(2, 200);
        let snapshot = serde_json::to_value(&fss).unwrap();
        let mut duplicate = snapshot.clone();
        duplicate["monitored_list"]["entries"][1] =
            duplicate["monitored_list"]["entries"][0].clone();
        assert!(serde_json::from_value::<FilteredSpaceSaving<u64>>(duplicate).is_err());
        let mut mismatched = snapshot.clone();
        mismatched["monitored_list"]["capacity"] = 3.into();
        assert!(serde_json::from_value::<FilteredSpaceSaving<u64>>(mismatched).is_err());
        let mut invalid_offset = snapshot;
        invalid_offset["alphas"]["offset"] = u64::MAX.into();
        invalid_offset["alphas"]["deltas"][0] = 1.into();
        assert!(serde_json::from_value::<FilteredSpaceSaving<u64>>(invalid_offset).is_err());
    }

    #[test]
    fn test_alpha_rebase_and_clipping() {
        let mut alphas = AlphaCounters::new(3);
        alphas.set(0, 200);
        alphas.set(1, 210);
        alphas.set(2, 300);
        assert_eq!(alphas.offset, 45);
        assert_eq!((0..3).map(|i| alphas.get(i)).collect::<Vec<_>>(), vec![200, 210, 300]);

        alphas.set(2, 600);
        assert_eq!(alphas.offset, 345);
        assert_eq!(alphas.deltas, vec![0, 0, 255]);
        assert_eq!((0..3).map(|i| alphas.get(i)).collect::<Vec<_>>(), vec![345, 345, 600]);

        // A write below the baseline is rounded up, not wrapped.
        alphas.set(0, 10);
        assert_eq!(alphas.get(0), 345);
        alphas.set(0, u64::MAX);
        assert_eq!(alphas.get(0), u64::MAX);
        assert_eq!(alphas.get(1), u64::MAX - 255);
    }

    #[test]
    fn test_alpha_merge_different_offsets() {
        let mut left = AlphaCounters { offset: 1000, deltas: vec![0, 200, 255] };
        let right = AlphaCounters { offset: 2000, deltas: vec![0, 100, 255] };
        let mut reversed = right.clone();
        reversed.merge(&left);
        let allocation = left.deltas.as_ptr();
        left.merge(&right);
        assert_eq!(left.deltas.as_ptr(), allocation);
        assert_eq!(left.offset, 3255);
        assert_eq!(left.deltas, vec![0, 45, 255]);
        assert_eq!(left.offset, reversed.offset);
        assert_eq!(left.deltas, reversed.deltas);
    }

    #[test]
    fn test_alpha_decay_rounding_and_clipping() {
        let mut alphas = AlphaCounters { offset: 301, deltas: vec![1, 3, 255] };
        let allocation = alphas.deltas.as_ptr();
        alphas.decay(0.5);
        assert_eq!((0..3).map(|i| alphas.get(i)).collect::<Vec<_>>(), vec![151, 152, 278]);
        alphas.decay(4.0);
        assert_eq!(alphas.offset, 857);
        assert_eq!((0..3).map(|i| alphas.get(i)).collect::<Vec<_>>(), vec![857, 857, 1112]);
        assert_eq!(alphas.deltas.as_ptr(), allocation);
        alphas.clear();
        assert_eq!(alphas.offset, 0);
        assert_eq!(alphas.deltas, vec![0, 0, 0]);
        assert_eq!(alphas.deltas.as_ptr(), allocation);
    }

    #[test]
    fn test_alpha_storage_stays_one_byte_per_bucket() {
        let mut alphas = AlphaCounters::new(600);
        let allocation = alphas.deltas.as_ptr();
        alphas.set(0, 1_000_000);
        alphas.set(599, 2_000_000);
        assert_eq!(alphas.deltas.as_ptr(), allocation);
        assert_eq!(alphas.deltas.capacity(), 600);
        assert_eq!(
            std::mem::size_of_val(alphas.deltas.as_slice()) + std::mem::size_of_val(&alphas.offset),
            608
        );
    }

    #[test]
    fn test_large_stream_bounds_after_rebasing_and_merge() {
        let mut left = FilteredSpaceSaving::new(8);
        let mut right = FilteredSpaceSaving::new(8);
        let mut exact = HashMap::new();
        for i in 0..2000u64 {
            let key = i % 37;
            let weight = if key < 3 { 1000 } else { 100 };
            if i % 2 == 0 {
                left.insert(key, weight);
            } else {
                right.insert(key, weight);
            }
            *exact.entry(key).or_insert(0u64) += weight;
        }
        assert!(left.alphas.offset > 0);
        assert!(right.alphas.offset > 0);
        left.merge(&right).unwrap();
        // Also exercise eviction after a merge with an elevated baseline.
        for key in 37..74 {
            left.insert(key, 2000);
            *exact.entry(key).or_insert(0) += 2000;
        }
        assert_eq!(left.count(), exact.values().sum::<u64>());
        for key in 0..3 {
            assert!(left.get(&key).is_some(), "lost frequent key {key}");
        }
        for (key, count) in exact {
            let estimate = left.estimate(&key);
            assert!(estimate.estimated_count() >= count);
            assert!(estimate.estimated_count() - estimate.associated_error() <= count);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip_with_nonzero_offset() {
        let mut fss = FilteredSpaceSaving::new(2);
        for i in 0..100u64 {
            fss.insert(i % 7, 1000);
        }
        assert!(fss.alphas.offset > 0);
        let serialized = serde_json::to_string(&fss).unwrap();
        let restored: FilteredSpaceSaving<u64> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(fss.alphas.offset, restored.alphas.offset);
        assert_eq!(fss.alphas.deltas, restored.alphas.deltas);
        for key in 0..10 {
            assert_eq!(fss.estimate(&key), restored.estimate(&key));
        }
        fss.clear();
        assert_eq!(fss.alphas.offset, 0);
        assert_eq!(fss.count(), 0);
        assert_eq!(fss.estimate(&0), ElementCounter::new(0, 0));
    }

    #[test]
    fn test_topk() {
        let mut fss = FilteredSpaceSaving::new(3);
        fss.insert("1", 10);
        fss.insert("2", 20);
        fss.insert("3", 2);
        fss.insert("4", 1);
        fss.insert("4", 3);
        fss.insert("5", 5);
        let result = fss.into_sorted_vec();
        assert_eq!(result[0].0, "2");
        assert_eq!(result[0].1, ElementCounter::new(20, 0));
        assert_eq!(result[1].0, "1");
        assert_eq!(result[1].1, ElementCounter::new(10, 0));
        assert_eq!(result[2].0, "5");
        assert!(result[2].1.estimated_count >= 5);
    }

    #[test]
    fn test_merge() {
        // Lemma 3 Guarantees are verified by property tests
        // here we use large k to have stable result
        let mut fss1 = FilteredSpaceSaving::new(5);
        fss1.insert("1", 10);
        fss1.insert("2", 20);
        fss1.insert("3", 2);
        fss1.insert("4", 1);
        fss1.insert("4", 3);
        fss1.insert("5", 5);
        let mut fss2 = FilteredSpaceSaving::new(5);
        fss2.insert("1", 10);
        fss2.insert("2", 20);
        fss2.insert("3", 20);
        fss2.insert("4", 10);
        let total_count = fss1.count + fss2.count;
        fss2.merge(&fss1).unwrap();
        assert_eq!(fss2.count, total_count);
        let result = fss2.into_sorted_vec();
        assert_eq!(result[0].0, "2");
        assert_eq!(result[0].1, ElementCounter::new(40, 0));
        assert_eq!(result[1].0, "3");
        assert!(result[1].1.estimated_count + result[1].1.associated_error >= 22);
        assert_eq!(result[2].0, "1");
        assert_eq!(result[2].1, ElementCounter::new(20, 0));
    }

    #[test]
    fn test_merge_bad() {
        let mut fss1 = FilteredSpaceSaving::new(1);
        fss1.insert(0, 0);
        let mut fss2 = FilteredSpaceSaving::new(2);
        fss2.insert(0, 0);
        assert!(fss1.merge(&fss2).is_err());
    }

    #[test]
    fn test_clear() {
        let mut fss = FilteredSpaceSaving::new(3);
        fss.insert("1", 10);
        fss.insert("2", 20);
        fss.insert("3", 2);
        fss.insert("4", 1);
        fss.insert("4", 3);
        fss.insert("5", 5);
        fss.clear();
        assert_eq!(fss.k, 3);
        assert_eq!(fss.count, 0);
        assert_eq!(fss.alphas.offset, 0);
        for i in 0..fss.alphas.len() {
            assert_eq!(fss.alphas.get(i), 0);
        }
        assert_eq!(fss.monitored_list.len(), 0);
    }

    fn topk_of<'a>(fss: &'a FilteredSpaceSaving<&str>) -> Vec<(&'a str, u64)> {
        let mut result = Vec::with_capacity(fss.monitored_list.len());
        result.extend(fss.monitored_list.iter());
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.iter().map(|(&name, &counter)| (name, counter.estimated_count())).collect::<Vec<_>>()
    }

    #[test]
    fn test_decay() {
        let mut fss = FilteredSpaceSaving::new(5);
        fss.insert("a", 10);
        fss.insert("b", 20);
        fss.insert("c", 6);
        fss.insert("d", 8);
        fss.insert("a", 2);

        assert_eq!(fss.count(), 46);
        assert_eq!(topk_of(&fss), vec![("b", 20), ("a", 12), ("d", 8), ("c", 6)]);

        fss.decay(0.5);
        assert_eq!(fss.count(), 23);
        assert_eq!(topk_of(&fss), vec![("b", 10), ("a", 6), ("d", 4), ("c", 3)]);

        // make sure we did not lose the alphas of items outside the top-k
        assert_eq!(fss.estimate(&"c").estimated_count(), 3);
        fss.insert("c", 2);
        assert_eq!(fss.estimate(&"c").estimated_count(), 5);
        assert_eq!(topk_of(&fss), vec![("b", 10), ("a", 6), ("c", 5), ("d", 4)]);
    }

    #[test]
    fn test_merge_into_empty() {
        let mut empty = FilteredSpaceSaving::new(100);
        let mut other = FilteredSpaceSaving::new(100);

        for i in 0..50 {
            other.insert(i, (i % 10 + 1) as u64);
        }

        empty.merge(&other).unwrap();
        assert_eq!(empty.iter().count(), 50);
    }

    #[test]
    fn test_merge_underfull() {
        let mut a = FilteredSpaceSaving::new(100);
        let mut b = FilteredSpaceSaving::new(100);

        for i in 0..30 {
            a.insert(i, 10);
        }
        // Lower counts than a's minimum
        for i in 30..60 {
            b.insert(i, 5);
        }

        a.merge(&b).unwrap();
        assert_eq!(a.iter().count(), 60); // Should have all 60 items
    }

    // =======================================================================
    // Property tests
    //
    // These tests generate hundreds of individual test cases, and check that
    // certain algorithmic invariants hold. If a failure is found, proptest
    // will try to "shrink" the failure to find a minimal example, which can
    // then be added as a regular test case.
    //
    // These tests verify the Lemma 3 invariants from the original
    // Space-Saving paper (Metwally et al., 2005).
    //
    // ### Lemma 3 Guarantees
    //
    // For any monitored element with true frequency `f` and estimated count
    // `c` with error `ε`:
    //
    // - `0 ≤ ε ≤ min` (error bounded by minimum count in structure)
    // - `f ≤ c ≤ f + min` (estimated count never underestimates)
    //
    // Testable properties:
    //
    // 1. `estimated_count >= exact_count` (never underestimates)
    // 2. `estimated_count - associated_error <= exact_count` (error bound is
    //     conservative)

    /// Weighted key distribution approximating Zipf's law (common English
    /// words)
    fn weighted_key() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            100 => Just("the"),
            70 => Just("of"),
            60 => Just("and"),
            55 => Just("to"),
            50 => Just("a"),
            45 => Just("in"),
            35 => Just("is"),
            30 => Just("it"),
            25 => Just("for"),
            20 => Just("that"),
        ]
    }

    /// (key, count) pair for insertion
    fn weighted_insertion() -> impl Strategy<Value = (&'static str, u64)> {
        (weighted_key(), 0u64..=2000)
    }

    proptest! {
        #[test]
        fn test_alpha_updates_never_lower_other_buckets(
            updates in prop::collection::vec((0usize..12, 0u64..1_000_000), 1..100)
        ) {
            let mut alphas = AlphaCounters::new(12);
            for (index, value) in updates {
                let before: Vec<_> = (0..12).map(|i| alphas.get(i)).collect();
                alphas.set(index, value);
                prop_assert!(alphas.get(index) >= value);
                for (i, previous) in before.into_iter().enumerate() {
                    if i != index {
                        prop_assert!(alphas.get(i) >= previous);
                    }
                }
            }
        }

        #[test]
        fn test_alpha_merge_never_underestimates_sums(
            offsets in (0u64..1_000_000, 0u64..1_000_000),
            deltas in prop::collection::vec((any::<u8>(), any::<u8>()), 1..100)
        ) {
            let mut left = AlphaCounters { offset: offsets.0, deltas: deltas.iter().map(|&(a, _)| a).collect() };
            let right = AlphaCounters { offset: offsets.1, deltas: deltas.iter().map(|&(_, b)| b).collect() };
            let sums: Vec<_> = (0..left.len()).map(|i| left.get(i) + right.get(i)).collect();
            left.merge(&right);
            for (i, sum) in sums.into_iter().enumerate() {
                prop_assert!(left.get(i) >= sum);
            }
        }

        /// Verify Lemma 3 invariants hold after insert operations
        #[test]
        fn test_insert_lemma3_invariants(
            k in 1usize..=12,
            insertions in prop::collection::vec(weighted_insertion(), 0..25)
        ) {
            let mut fss = FilteredSpaceSaving::new(k);
            let mut exact_counts: HashMap<&str, u64> = HashMap::new();

            for (key, count) in insertions {
                fss.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }

            // Check Lemma 3 for ALL items via estimate()
            for (key, &exact) in &exact_counts {
                let estimate = fss.estimate(key);
                let estimated = estimate.estimated_count();
                let error = estimate.associated_error();

                // Invariant 1: Never underestimates
                prop_assert!(estimated >= exact,
                    "Underestimate: key={}, est={}, err={}, exact={} ", key, estimated, error, exact);

                // Invariant 2: Error bound is conservative
                // For evicted items (estimated == error): 0 <= exact (trivially true)
                // For monitored items: meaningful bound
                prop_assert!(estimated - error <= exact,
                    "Error bound violated: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);
            }
        }

        /// Verify Lemma 3 invariants hold after merge operations
        #[test]
        fn test_merge_lemma3_invariants(
            k in 1usize..=12,
            insertions_a in prop::collection::vec(weighted_insertion(), 0..25),
            insertions_b in prop::collection::vec(weighted_insertion(), 0..25),
        ) {
            let mut fss_a = FilteredSpaceSaving::new(k);
            let mut fss_b = FilteredSpaceSaving::new(k);
            let mut exact_counts: HashMap<&str, u64> = HashMap::new();

            for (key, count) in insertions_a {
                fss_a.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }
            for (key, count) in insertions_b {
                fss_b.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }

            fss_a.merge(&fss_b).unwrap();

            // Check Lemma 3 for ALL items via estimate()
            // This catches underfull merge bug: wrongly-dropped items
            // will have estimate() return low alpha, failing estimated >= exact
            for (key, &exact) in &exact_counts {
                let estimate = fss_a.estimate(key);
                let estimated = estimate.estimated_count();
                let error = estimate.associated_error();

                prop_assert!(estimated >= exact,
                    "Merge underestimate: key={}, est={}, err={}, exact={}", key, estimated, error, exact);

                prop_assert!(estimated - error <= exact,
                    "Merge error bound violated: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);
            }
        }

        /// Verify Lemma 3 invariants hold after serde roundtrip
        #[cfg(feature = "serde")]
        #[test]
        fn test_serde_roundtrip_and_lemma3_invariants(
            k in 1usize..=12,
            insertions in prop::collection::vec(weighted_insertion(), 0..25)
        ) {
            let mut fss = FilteredSpaceSaving::new(k);
            let mut exact_counts: HashMap<&str, u64> = HashMap::new();

            for (key, count) in insertions {
                fss.insert(key, count);
                *exact_counts.entry(key).or_default() += count;
            }

            // Serialize and deserialize
            let serialized = serde_json::to_string(&fss).unwrap();
            let deserialized: FilteredSpaceSaving<&str> = serde_json::from_str(&serialized).unwrap();

            // Verify basic properties preserved
            prop_assert_eq!(fss.count(), deserialized.count());
            prop_assert_eq!(fss.k(), deserialized.k());

            // Check Lemma 3 for ALL items via estimate() after roundtrip
            for (key, &exact) in &exact_counts {
                let orig = fss.estimate(key);
                let restored = deserialized.estimate(key);

                // Estimates should match exactly before and after
                prop_assert_eq!(orig.estimated_count(), restored.estimated_count(),
                    "Serde roundtrip changed estimate: key={}", key);
                prop_assert_eq!(orig.associated_error(), restored.associated_error(),
                    "Serde roundtrip changed error: key={}", key);

                // Lemma 3 invariants should hold. This is almost certainly
                // reundant with the insert tests, and exact dump/load tests
                // above, but we write it out explicitly for documentation
                // purposes, and to be extremely careful.
                let estimated = restored.estimated_count();
                let error = restored.associated_error();

                prop_assert!(estimated >= exact,
                    "Serde roundtrip underestimate: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);

                prop_assert!(estimated - error <= exact,
                    "Serde roundtrip error bound violated: key={}, est={}, err={}, exact={}",
                    key, estimated, error, exact);
            }
        }
    }
}
