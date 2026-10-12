# TopK

TopK algorithm implementation in Rust.

This crate currently provides the [Filtered Space-Saving algorithm](https://doi.org/10.1016/j.ins.2010.08.024).

Version numbers follow the [semver](https://semver.org/) convention.

The filter stores `6 * k` one-byte counters and a shared 64-bit offset: `6 * k + 8`
bytes of counter data, compared with `48 * k` bytes for full 64-bit filter counters.
This excludes the vector header, allocator overhead, and monitored items.
When filter values span more than 255, the offset advances and low values are
rounded up to it. This can increase count estimates and change the returned top-k.
Rebasing scans the filter in place and does not allocate additional counter storage.

Monitored items use a compact indexed heap: a single array of keys and counters,
two arrays of `u16` heap indices, and a hash table containing only `u16` slot
indices. Capacity is limited to 65,536 items. Lookup is expected O(1), minimum
access is O(1), and heap updates are O(log k). Keys are compared for equality,
so hash collisions do not merge distinct items. Owned key buffers, such as String
contents, are additional to the sketch's storage.

`insert` returns `InsertResult::Updated`, `Rejected`, `Inserted`, or
`Replaced { expelled }`, reporting membership immediately after the insertion.
Later insertions may still expel the item.

With the `serde` feature, `alphas` is serialized as an object containing `offset`
and byte `deltas`, and `monitored_list` contains a `capacity` and an `entries`
array. Heap indices and the lookup table are rebuilt on load. Snapshots using
the previous array formats must be migrated before deserialization.

### Example

```rust
let mut topk = FilteredSpaceSaving::new(3);
topk.insert("1", 10);
topk.insert("2", 20);
topk.insert("3", 1);
topk.insert("4", 2);
let topk_result = topk.into_sorted_vec();
assert_eq!(topk_result.len(), 3);
assert_eq!(topk_result[0].0, "2");
```

merging space-saving results is supported:

```rust
let mut fss1 = FilteredSpaceSaving::new(3);
fss1.insert("1", 10);
fss1.insert("2", 20);
fss1.insert("3", 2);
fss1.insert("4", 1);
fss1.insert("4", 3);
fss1.insert("5", 3);
let mut fss2 = FilteredSpaceSaving::new(3);
fss2.insert("1", 10);
fss2.insert("2", 20);
fss2.insert("3", 20);
fss2.insert("4", 10);
fss1.merge(&fss2).unwrap();
let result = fss1.into_sorted_vec();
assert_eq!(result[0].0, "2");
```
