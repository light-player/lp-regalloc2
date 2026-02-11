//! Chunked vector for allocation-sensitive contexts (e.g. embedded).
//!
//! Allocates in small chunks instead of one large contiguous block to reduce
//! OOM risk from heap fragmentation when compiling shaders on constrained heaps.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

/// Chunk size: keeps each allocation small (~1KB for 16-byte elements).
/// Matches Cranelift ChunkedVec tuning that fixed similar OOM on ESP32.
const CHUNK_SIZE: usize = 64;

/// A vector backed by multiple smaller allocations.
///
/// Uses `ceil(len/CHUNK_SIZE)` chunks instead of one large Vec to reduce
/// peak allocation size and improve success on fragmented heaps.
#[derive(Debug)]
pub struct ChunkedVec<T> {
    chunks: Vec<Vec<T>>,
    len: usize,
}

impl<T: Clone> ChunkedVec<T> {
    /// Create with given length, filling with `default`.
    pub fn with_capacity_and_default(len: usize, default: T) -> Self {
        let mut chunks = Vec::new();
        let mut remaining = len;
        while remaining > 0 {
            let chunk_len = remaining.min(CHUNK_SIZE);
            chunks.push(vec![default.clone(); chunk_len]);
            remaining -= chunk_len;
        }
        Self { chunks, len }
    }
}

impl<T> ChunkedVec<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn chunk_and_offset(&self, i: usize) -> (usize, usize) {
        debug_assert!(i < self.len);
        (i / CHUNK_SIZE, i % CHUNK_SIZE)
    }

    pub fn get(&self, i: usize) -> &T {
        let (ci, o) = self.chunk_and_offset(i);
        &self.chunks[ci][o]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut T {
        let (ci, o) = self.chunk_and_offset(i);
        &mut self.chunks[ci][o]
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.chunks.iter().flat_map(|c| c.iter())
    }
}

impl<T> Index<usize> for ChunkedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        self.get(i)
    }
}

impl<T> IndexMut<usize> for ChunkedVec<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        self.get_mut(i)
    }
}
