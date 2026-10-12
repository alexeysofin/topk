//! Measure the sketch's requested allocations, including owned key buffers,
//! separately from process RSS and the test harness's existing allocations.
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use topk::FilteredSpaceSaving;

struct TrackingAllocator;

thread_local! {
    // Each test owns and drops its sketch on the same thread. Separate counters
    // prevent parallel tests and test-harness threads from affecting readings.
    static LIVE_BYTES: Cell<isize> = const { Cell::new(0) };
}

fn track_bytes(change: isize) {
    // Thread-local storage may be unavailable during thread teardown.
    let _ = LIVE_BYTES.try_with(|bytes| bytes.set(bytes.get() + change));
}

fn live_bytes() -> isize {
    LIVE_BYTES.with(Cell::get)
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            track_bytes(layout.size() as isize);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            track_bytes(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track_bytes(-(layout.size() as isize));
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            track_bytes(new_size as isize - layout.size() as isize);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn thousand_u64_keys_fit_in_33_kb_without_growth() {
    // Initialize thread-local hashing state outside the measured interval.
    drop(FilteredSpaceSaving::<u64>::new(1));
    let before = live_bytes();
    let mut sketch = FilteredSpaceSaving::new(1000);
    let reserved_bytes = live_bytes() - before;
    for key in 0..1000u64 {
        sketch.insert(key, 1);
    }
    assert_eq!(live_bytes() - before, reserved_bytes);
    for key in 1000..10_000 {
        sketch.insert(key, 1000);
    }
    assert_eq!(sketch.iter().count(), 1000);
    assert_eq!(live_bytes() - before, reserved_bytes);
    let total_bytes = reserved_bytes as usize + std::mem::size_of_val(&sketch);
    assert!(total_bytes <= 33_000, "sketch storage was {total_bytes} bytes");
    sketch.clear();
    assert_eq!(live_bytes() - before, reserved_bytes);
    drop(sketch);
    assert_eq!(live_bytes(), before);
    println!(
        "k=1000, u64 keys: {reserved_bytes} heap bytes, {total_bytes} bytes including the struct"
    );
}

#[test]
fn thousand_random_24_byte_strings_print_sketch_size() {
    // Generate input before measuring. Only the sketch's copies of the strings
    // are included in its memory usage, not the separate input collection.
    let keys = proptest::collection::vec("[a-zA-Z0-9]{24}", 1000)
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current();
    drop(FilteredSpaceSaving::<String>::new(1));
    let before = live_bytes();
    let mut sketch = FilteredSpaceSaving::new(1000);
    let reserved_bytes = live_bytes() - before;
    for key in &keys {
        assert_eq!(key.len(), 24);
        sketch.insert(key.clone(), 1);
    }
    assert_eq!(sketch.iter().count(), 1000);
    let string_bytes: usize = sketch.iter().map(|(key, _)| key.capacity()).sum();
    assert_eq!(string_bytes, 24_000);
    let heap_bytes = live_bytes() - before;
    assert_eq!(heap_bytes, reserved_bytes + string_bytes as isize);
    let total_bytes = heap_bytes as usize + std::mem::size_of_val(&sketch);
    assert!(total_bytes <= 73_000, "sketch storage was {total_bytes} bytes");
    sketch.clear();
    assert_eq!(live_bytes() - before, reserved_bytes);
    drop(sketch);
    assert_eq!(live_bytes(), before);
    println!(
        "k=1000, random 24-byte String keys: {heap_bytes} heap bytes, {total_bytes} bytes including string buffers and the struct ({:.2} KiB)",
        total_bytes as f64 / 1024.0
    );
}
