//! Measure the sketch's requested allocations independently of key buffers,
//! process RSS, and the test harness's existing allocations.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use topk::FilteredSpaceSaving;

struct TrackingAllocator;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE_BYTES.fetch_add(new_size as isize - layout.size() as isize, Ordering::SeqCst);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn thousand_u64_keys_fit_in_41_kb_without_growth() {
    // Initialize thread-local hashing state outside the measured interval.
    drop(FilteredSpaceSaving::<u64>::new(1));
    let before = LIVE_BYTES.load(Ordering::SeqCst);
    let mut sketch = FilteredSpaceSaving::new(1000);
    let reserved_bytes = LIVE_BYTES.load(Ordering::SeqCst) - before;
    for key in 0..1000u64 {
        sketch.insert(key, 1);
    }
    assert_eq!(LIVE_BYTES.load(Ordering::SeqCst) - before, reserved_bytes);
    for key in 1000..10_000 {
        sketch.insert(key, 1000);
    }
    assert_eq!(sketch.iter().count(), 1000);
    assert_eq!(LIVE_BYTES.load(Ordering::SeqCst) - before, reserved_bytes);
    let total_bytes = reserved_bytes as usize + std::mem::size_of_val(&sketch);
    assert!(total_bytes <= 41_000, "sketch storage was {total_bytes} bytes");
    sketch.clear();
    assert_eq!(LIVE_BYTES.load(Ordering::SeqCst) - before, reserved_bytes);
    drop(sketch);
    assert_eq!(LIVE_BYTES.load(Ordering::SeqCst), before);
    println!(
        "k=1000, u64 keys: {reserved_bytes} heap bytes, {total_bytes} bytes including the struct"
    );
}
