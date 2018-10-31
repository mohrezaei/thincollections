// Copyright 2018 Mohammad Rezaei.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Portions copyright The Rust Project Developers. Licensed under
// the MIT License.

use util::*;
use thin_hasher::*;

use std::{
    alloc::{self, Layout},
    mem, ptr, marker,
};
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::fmt::{self, Debug};

pub struct ClaMap<K: Eq + Hash + Debug, V: Debug, H: BuildHasher> {
    hasher: H,
    table_blocks: usize,
    occupied: usize,
    flagged_blocks: usize, // a block is flagged if it ever gets full
    max_occupied: usize,
    block_kv_count: i8,
    block_v_offset: i8,
    table: *mut u64,
    _marker: marker::PhantomData<(K, V)>,
}

#[derive(PartialEq)]
enum BucketState {
    Full,
    NoSpace,
    Empty,
    Removed,
}

impl BucketState {
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        *self == BucketState::Full
    }
}

impl<K: Eq + Hash + Debug, V: Debug> ClaMap<K, V, OneFieldHasherBuilder> {
    pub fn new() -> Self {
        let (count, v_start) = calculate_sizes::<K, V>();
        ClaMap {
            table_blocks: 0,
            occupied: 0,
            flagged_blocks: 0,
            max_occupied: 0,
            block_kv_count: count,
            block_v_offset: v_start,
            table: ptr::null_mut(),
            hasher: OneFieldHasherBuilder::new(),
            _marker: marker::PhantomData,
        }
    }
}

impl<K: Eq + Hash + Debug, V: Debug, H: BuildHasher> ClaMap<K, V, H> {
    pub fn len(&self) -> usize {
        self.occupied
    }

    fn key_at(ptr: *mut u8, index: i8) -> *mut K {
        unsafe {
            ptr.offset(mem::size_of::<K>() as isize * index as isize) as *mut K
        }
    }

    fn value_at(&self, ptr: *mut u8, index: i8) -> *mut V {
        unsafe {
            ptr.offset(self.block_v_offset as isize + mem::size_of::<V>() as isize * index as isize) as *mut V
        }
    }

    fn increment_control(block_ptr: *mut u8, max: i8) -> bool
    {
        unsafe {
            let control_ptr = block_ptr.add(63);
            let already_flagged = (*control_ptr & 0x80) != 0;
            let mut v = (*control_ptr & 0x7F) + 1;
            if v == max as u8 {
                v |= 0x80;
            }
            v |= *control_ptr & 0x80;
            *control_ptr = v;
            return v & 0x80 != 0 && !already_flagged;
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.table_blocks == 0 {
            self.allocate_table();
        }
        let (block_ptr, index, bucket_state) = self.probe(&key);
        let flagged: bool;
        unsafe {
            if bucket_state.is_full() {
                return Some(mem::replace(&mut (*self.value_at(block_ptr, index)), value));
            }
            ptr::write(<ClaMap<K, V, H>>::key_at(block_ptr, index), key);
            ptr::write(self.value_at(block_ptr, index), value);
            flagged = <ClaMap<K, V, H>>::increment_control(block_ptr, self.block_kv_count);
        }
        self.occupied += 1;
        if flagged {
            self.flagged_blocks += 1;
        }
        if self.occupied >= self.max_occupied || self.flagged_blocks >= (self.table_blocks >> 1) {
            self.rehash();
        }
        return None;
    }

    #[inline]
    fn hash(&self, key: &K) -> u64 {
        let mut state = self.hasher.build_hasher();
        (*key).hash(&mut state);
        state.finish()
    }

    #[inline]
    fn hash_and_mask(&self, key: &K) -> (u64, isize) {
        let hash = self.hash(key);
        (hash, self.mask(hash) as isize)
    }

    #[inline(always)]
    fn mask(&self, hash: u64) -> u64 {
        hash & ((self.table_blocks - 1) as u64)
    }

    // 4 things can happen:
    // 1: we find the key, we'll return the index and BucketState::FULL
    // 2: we don't find the key, the block is not flagged, the block is not full,
    //      we'll return the index, BucketState::EMPTY
    // 3: we don't find the key, the block is flagged, the block is not full,
    //      we'll return the negative index, BucketState::REMOVED
    // 4: we don't find the key, the block is full (and therefore flagged),
    //      we'll return -1, BucketState::NO_SPACE
    fn search_block(&self, block_ptr: *mut u8, key: &K) -> (i8, BucketState) {
        unsafe {
            let control_ptr = block_ptr.offset(63);
            let count = (*control_ptr & 0x7F) as i8;
            let mut ptr: *mut K = block_ptr as *mut K;
            let mut index = 0;
            while index < count {
                if *key == *ptr {
                    return (index, BucketState::Full); // case 1
                }
                index += 1;
                ptr = ptr.add(1);
            }
            if count == self.block_kv_count {
                return (-1, BucketState::NoSpace); // case 4
            }
            let flagged = *control_ptr & 0x80 != 0;
            if flagged {
                return (-count, BucketState::Removed); // case 3
            }
            return (count, BucketState::Empty); // case 2
        }
    }

    fn probe(&self, key: &K) -> (*mut u8, i8, BucketState) {
        let (hash, block_index) = self.hash_and_mask(key);
        unsafe {
            let block_ptr: *mut u8 = self.table.offset(block_index << 3) as *mut u8;
            let (index, state): (i8, BucketState) = self.search_block(block_ptr, key);
            if index >= 0 {
                return (block_ptr, index, state);
            }
            self.probe2(key, hash, block_ptr, index, state)
        }
    }

    fn spread_two_and_mask(&self, hash: u64) -> isize {
        self.mask(spread_two(hash)) as isize
    }

    fn probe2(&self, key: &K, hash: u64, original_block_ptr: *mut u8, original_index: i8,
              original_state: BucketState) -> (*mut u8, i8, BucketState) {
        let block_index = self.spread_two_and_mask(hash);
        unsafe {
            let block_ptr: *mut u8 = self.table.offset(block_index << 3) as *mut u8;
            let (index, state): (i8, BucketState) = self.search_block(block_ptr, key);
            if index >= 0 {
                if state == BucketState::Full {
                    return (block_ptr, index, state);
                }
                if state == BucketState::Empty {
                    if original_state == BucketState::Removed {
                        return (original_block_ptr, original_index, original_state);
                    }
                    return (block_ptr, index, state);
                }
            }
            if original_state == BucketState::Removed {
                return self.probe3(key, hash, original_block_ptr, original_index, original_state);
            }
            return self.probe3(key, hash, block_ptr, index, state);
        }
    }

    fn probe3(&self, key: &K, hash: u64, mut original_block_ptr: *mut u8, mut original_index: i8,
              mut original_state: BucketState) -> (*mut u8, i8, BucketState) {
        let mut next_index = spread_one(hash) as isize;
        let spread_two = spread_two(hash).rotate_right(32) | 1;

        loop {
            unsafe {
                next_index = self.mask((next_index as u64).wrapping_add(spread_two)) as isize;
                let block_ptr: *mut u8 = self.table.offset(next_index << 3) as *mut u8;
                let (index, state): (i8, BucketState) = self.search_block(block_ptr, key);
                if index >= 0 {
                    if state == BucketState::Full {
                        return (block_ptr, index, state);
                    }
                    if state == BucketState::Empty {
                        if original_state == BucketState::Removed {
                            return (original_block_ptr, original_index, original_state);
                        }
                        return (block_ptr, index, state);
                    }
                }
                if state == BucketState::Removed && original_state != BucketState::Removed {
                    original_block_ptr = block_ptr;
                    original_state = state;
                    original_index = index;
                }
            }
        }
    }

    fn allocate_table(&mut self) {
        let (num_blocks, max_occupied) = self.block_for_capcity(8);
        self.max_occupied = max_occupied;
        self.table = <ClaMap<K, V, H>>::allocate_table_for_blocks(num_blocks);
        self.table_blocks = num_blocks;
    }

    fn allocate_table_for_blocks(blocks: usize) -> *mut u64 {
        unsafe {
            let layout = Layout::from_size_align(64 * blocks, 64).unwrap();
            alloc::alloc_zeroed(layout) as *mut u64
        }
    }

    fn block_for_capcity(&self, capacity: usize) -> (usize, usize) {
        let count_minus_one = self.block_kv_count - 1;
        let mut num_blocks = ceil_pow2(capacity as u64 / (count_minus_one as u64)) as usize;
        if count_minus_one as usize * num_blocks < capacity { num_blocks <<= 1; }
        (num_blocks, num_blocks * (count_minus_one as usize))
    }

    fn rehash(&mut self) {
        let x = self.table_blocks << 1;
        self.rehash_for_blocks(x);
    }

    fn rehash_for_blocks(&mut self, new_block_count: usize) {
        let old_table = self.table;
        let old_block_count = self.table_blocks;
        self.table = <ClaMap<K, V, H>>::allocate_table_for_blocks(new_block_count);
        self.table_blocks = new_block_count;
        self.flagged_blocks = 0;
        self.max_occupied = new_block_count * ((self.block_kv_count - 1) as usize);
        unsafe {
            let mut ptr: *mut u64 = old_table as *mut u64;
            let table_end = old_table.add(old_block_count << 3);
            while ptr < table_end {
                let block_ptr = ptr as *mut u8;
                let control_ptr = block_ptr.offset(63);
                let count = (*control_ptr & 0x7F) as i8;
                let mut kptr: *mut K = block_ptr as *mut K;
                let mut vptr: *mut V = block_ptr.offset(self.block_v_offset as isize) as *mut V;
                let mut i = 0;
                while i < count {
                    let (insert_block_ptr, index, _bucket_state) = self.probe(&(*kptr));
                    ptr::copy_nonoverlapping(kptr, <ClaMap<K, V, H>>::key_at(insert_block_ptr, index), 1);
                    ptr::copy_nonoverlapping(vptr, self.value_at(insert_block_ptr, index), 1);
                    let flagged = <ClaMap<K, V, H>>::increment_control(insert_block_ptr, self.block_kv_count);
                    if flagged {
                        self.flagged_blocks += 1;
                    }
                    i += 1;
                    kptr = kptr.add(1);
                    vptr = vptr.add(1);
                }
                ptr = ptr.add(8);
            }
            let layout = Layout::from_size_align(64 * old_block_count, 64).unwrap();
            alloc::dealloc(old_table as *mut u8, layout);
        }
    }
}

impl<K: Eq + Hash + Debug, V: Debug, H: BuildHasher> Drop for ClaMap<K, V, H> {
    fn drop(&mut self) {
        if self.table_blocks > 0 {
            unsafe {
                if self.occupied > 0 {
                    let mut ptr: *mut u64 = self.table;
                    let table_end = self.table.add(self.table_blocks << 3);
                    while ptr < table_end {
                        let block_ptr = ptr as *mut u8;
                        let control_ptr = block_ptr.offset(63);
                        let count = (*control_ptr & 0x7F) as i8;
                        let mut kptr: *mut K = block_ptr as *mut K;
                        let mut i = 0;
                        while i < count {
                            ptr::drop_in_place(kptr);
                            i += 1;
                            kptr = kptr.add(1);
                        }
                        if mem::size_of::<V>() > 0 {
                            i = 0;
                            let mut vptr = block_ptr.offset(self.block_v_offset as isize) as *mut V;
                            while i < count {
                                ptr::drop_in_place(vptr);
                                i += 1;
                                vptr = vptr.add(1);
                            }
                        }
                        ptr = ptr.add(8);
                    }
                }
                let layout = Layout::from_size_align(self.table_blocks * 64, 64).unwrap();
                alloc::dealloc(self.table as *mut u8, layout);
            }
        }
    }
}

impl<K, V, S> Debug for ClaMap<K, V, S>
    where K: Eq + Hash + Debug,
          V: Debug,
          S: BuildHasher
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug_map = f.debug_map();
        if self.table_blocks > 0 && self.occupied > 0 {
            unsafe {
                let mut ptr: *mut u64 = self.table;
                let table_end = self.table.add(self.table_blocks << 3);
                while ptr < table_end {
                    let block_ptr = ptr as *mut u8;
                    let control_ptr = block_ptr.offset(63);
                    let count = (*control_ptr & 0x7F) as i8;
                    let mut kptr: *mut K = block_ptr as *mut K;
                    let mut vptr = block_ptr.offset(self.block_v_offset as isize) as *mut V;
                    let mut i = 0;
                    while i < count {
                        debug_map.entry(&(*kptr), &(*vptr));
                        i += 1;
                        kptr = kptr.add(1);
                        vptr = vptr.add(1);
                    }
                    ptr = ptr.add(8);
                }
            }
        }
        debug_map.finish()
    }
}

impl<K, V, S> ClaMap<K, V, S>
    where K: Eq + Hash + Debug,
          V: Debug,
          S: BuildHasher
{
    pub fn debug(&self) {
        println!("occupied {}, table_blocks {}, flagged_blocks {}", self.occupied, self.table_blocks, self.flagged_blocks);
        if self.table_blocks > 0 && self.occupied > 0 {
            unsafe {
                let mut ptr: *mut u64 = self.table;
                let table_end = self.table.add(self.table_blocks << 3);
                while ptr < table_end {
                    let block_ptr = ptr as *mut u8;
                    let control_ptr = block_ptr.offset(63);
                    let count = (*control_ptr & 0x7F) as i8;
                    let mut kptr: *mut K = block_ptr as *mut K;
                    let mut vptr = block_ptr.offset(self.block_v_offset as isize) as *mut V;
                    let mut i = 0;
                    while i < count {
                        println!("[{:?},{:?}]", &(*kptr), &(*vptr));
                        i += 1;
                        kptr = kptr.add(1);
                        vptr = vptr.add(1);
                    }
                    ptr = ptr.add(8);
                }
            }
        }
    }
}

pub fn calculate_sizes<K, V>() -> (i8, i8) {
    let k_size = mem::size_of::<K>();
    let v_size = mem::size_of::<V>();
    let mut nominal_count = 63 as usize / (k_size + v_size);
    if nominal_count == 0 { panic!("Key-value size is too large!"); }
    if v_size > 0 {
        let v_align = mem::align_of::<V>();
        let mut v_align_offset_count;
        loop {
            v_align_offset_count = k_size * nominal_count / v_align;
            if v_align_offset_count * v_align < k_size * nominal_count {
                v_align_offset_count += 1;
            }
            if v_size * nominal_count + v_align_offset_count * v_align > 63 {
                nominal_count -= 1;
            } else { break; };
        }
        return (nominal_count as i8, (v_align_offset_count * v_align) as i8);
    }
    (nominal_count as i8, 0)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_tests() {
        let (count, v_start) = calculate_sizes::<u32, i32>();
        assert_eq!((7, 28), (count, v_start));

        let x = calculate_sizes::<i8, u64>();
        assert_eq!((6, 8), x);

        let x = calculate_sizes::<u8, ()>();
        assert_eq!((63, 0), x);

        let x = calculate_sizes::<i32, u64>();
        assert_eq!((4, 16), x);

        let x = calculate_sizes::<u64, i32>();
        assert_eq!((5, 40), x);

        let x = calculate_sizes::<u64, u8>();
        assert_eq!((7, 56), x);
    }

    #[test]
    #[should_panic]
    fn allocator_panic_test() {
        calculate_sizes::<(u64, u64, u64, u64), (i64, i64, i64, i64)>();
    }
}

