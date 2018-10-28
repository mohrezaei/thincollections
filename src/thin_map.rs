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

//! # `ThinMap`: a fast map with lower memory usage for small key values.
//! Unlike `std::collections::HashMap`, `ThinMap` is optimized for small key values.
//! It's generally 2-5x faster (see the benchmarks).
//! 
//! It uses less memory because it doesn't store the hash value
//! (64bit x load factor per entry) for every entry
//! in the map. It's also faster for several reasons:
//! - It uses less memory, and accessing memory is expensive.
//! - It uses an adaptive, non-linear, cache line aware collision resolution, which allows it
//!     to use a simpler hashing algorithm.
//! - Unlike `std::collections::HashMap`, inserts and removes do not cause element movement.
//!

use thin_sentinel::*;
use thin_hasher::*;
use util::*;

use std::{
    alloc::{self, Layout},
    mem, ptr, marker,
};
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::fmt::{self, Debug};
use std::cmp;
use std::ops::Index;
use std::iter::FromIterator;
use std::iter::FusedIterator;

/// A fast, low memory replacement for HashMap.
///
/// Keys must implement `ThinSentinel`, which is already implemented for all primitives.
/// Ideally, `mem::size_of::<(K,V)>()` should be 18 bytes or less. The key size is the
/// critical factor for performance, and generally should be 64 bits or less. The map will
/// work fine for larger V sizes, but it will start to lose its advantage over `HashMap`.
/// Keys that have a `Drop` impl have not been tested and should be avoided (it's theoretically
/// possible to have such keys with a proper implementation of `ThinSentinel`, but it's hard).
pub struct ThinMap<K: ThinSentinel + Eq + Hash, V, H: BuildHasher = OneFieldHasherBuilder> {
    hasher: H,
    table_size: usize,
    occupied: usize,
    sentinels: usize,
    occupied_sentinels: u8,
    table: *mut (K, V),
    _marker: marker::PhantomData<(K, V)>,
}

pub enum Entry<'a, K: 'a, V: 'a> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    ptr: &'a mut (K, V),
    occupied_sentinels: &'a mut u8,
    occupied: &'a mut usize,
    sentinels: &'a mut usize,
}

pub struct VacantEntry<'a, K: 'a, V: 'a> {
    ptr: &'a mut (K, V),
    key: K,
    occupied_sentinels: &'a mut u8,
    occupied: &'a mut usize,
    sentinels: &'a mut usize,
}

#[doc(hidden)]
pub struct Drain<'a, K: 'a + ThinSentinel + Eq, V: 'a> {
    sentinel_zero_ptr: *mut (K, V),
    sentinel_one_ptr: *mut (K, V),
    cur: *mut (K, V),
    end: *mut (K, V),
    occupied: &'a mut usize,
    sentinels: &'a mut usize,
    occupied_sentinels: &'a mut u8,
    _marker: marker::PhantomData<(&'a K, &'a V)>,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct Iter<'a, K: 'a, V: 'a> {
    sentinel_zero_ptr: *mut (K, V),
    sentinel_one_ptr: *mut (K, V),
    todo: usize,
    cur: *mut (K, V),
    end: *mut (K, V),
    _marker: marker::PhantomData<(&'a K, &'a V)>,
}

impl<'a, K: 'a, V: 'a> ExactSizeIterator for Iter<'a, K, V>
    where K: ThinSentinel + Eq
{
    #[inline]
    fn len(&self) -> usize {
        self.todo
    }
}

impl<'a, K: 'a, V: 'a> FusedIterator for Iter<'a, K, V>
    where K: ThinSentinel + Eq {}

impl<'a, K: 'a, V: 'a> Iterator for Iter<'a, K, V>
    where K: ThinSentinel + Eq
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        unsafe {
            if !self.sentinel_zero_ptr.is_null() {
                let r = Some((&(*self.sentinel_zero_ptr).0, &(*self.sentinel_zero_ptr).1));
                self.todo -= 1;
                self.sentinel_zero_ptr = ptr::null_mut();
                return r;
            }
            if !self.sentinel_one_ptr.is_null() {
                let r = Some((&(*self.sentinel_one_ptr).0, &(*self.sentinel_one_ptr).1));
                self.todo -= 1;
                self.sentinel_one_ptr = ptr::null_mut();
                return r;
            }
        }
        unsafe {
            while self.cur < self.end {
                if K::thin_sentinel_zero() != (*self.cur).0 && K::thin_sentinel_one() != (*self.cur).0 {
                    let r = Some((&(*self.cur).0, &(*self.cur).1));
                    self.todo -= 1;
                    self.cur = self.cur.add(1);
                    return r;
                }
                self.cur = self.cur.add(1);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.todo, Some(self.todo))
    }
}

impl<'a, K: 'a, V: 'a> Drop for Drain<'a, K, V>
    where K: ThinSentinel + Eq,
{
    fn drop(&mut self) {
        self.for_each(drop);
    }
}


impl<'a, K, V> ExactSizeIterator for Drain<'a, K, V>
    where K: ThinSentinel + Eq
{
    #[inline]
    fn len(&self) -> usize {
        *self.occupied + (*self.occupied_sentinels as usize)
    }
}

impl<'a, K, V> FusedIterator for Drain<'a, K, V>
    where K: ThinSentinel + Eq {}

impl<'a, K, V> Iterator for Drain<'a, K, V>
    where K: ThinSentinel + Eq
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        unsafe {
            if !self.sentinel_zero_ptr.is_null() {
                *self.occupied_sentinels -= 1;
                let r = Some(ptr::read(self.sentinel_zero_ptr));
                overwrite_k(self.sentinel_zero_ptr, K::thin_sentinel_one());
                self.sentinel_zero_ptr = ptr::null_mut();
                return r;
            }
            if !self.sentinel_one_ptr.is_null() {
                *self.occupied_sentinels -= 1;
                let r = Some(ptr::read(self.sentinel_one_ptr));
                overwrite_k(self.sentinel_one_ptr, K::thin_sentinel_zero());
                self.sentinel_one_ptr = ptr::null_mut();
                return r;
            }
        }
        unsafe {
            while self.cur < self.end {
                if K::thin_sentinel_zero() != (*self.cur).0 && K::thin_sentinel_one() != (*self.cur).0 {
                    let r = Some(ptr::read(self.cur));
                    overwrite_k(self.cur, K::thin_sentinel_one());
                    self.cur = self.cur.add(1);
                    *self.occupied -= 1;
                    *self.sentinels += 1;
                    return r;
                }
                self.cur = self.cur.add(1);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = (*self.occupied_sentinels as usize) + *self.occupied;
        (r, Some(r))
    }
}

#[doc(hidden)]
pub struct IterMut<'a, K: 'a, V: 'a> {
    sentinel_zero_ptr: *mut (K, V),
    sentinel_one_ptr: *mut (K, V),
    todo: usize,
    cur: *mut (K, V),
    end: *mut (K, V),
    _marker: marker::PhantomData<(&'a K, &'a mut V)>,
}

impl<'a, K: 'a, V: 'a> ExactSizeIterator for IterMut<'a, K, V>
    where K: ThinSentinel + Eq
{
    #[inline]
    fn len(&self) -> usize {
        self.todo
    }
}

impl<'a, K: 'a, V: 'a> FusedIterator for IterMut<'a, K, V>
    where K: ThinSentinel + Eq {}

impl<'a, K: 'a, V: 'a> Iterator for IterMut<'a, K, V>
    where K: ThinSentinel + Eq
{
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        unsafe {
            if !self.sentinel_zero_ptr.is_null() {
                let r = Some((&(*self.sentinel_zero_ptr).0, &mut (*self.sentinel_zero_ptr).1));
                self.todo -= 1;
                self.sentinel_zero_ptr = ptr::null_mut();
                return r;
            }
            if !self.sentinel_one_ptr.is_null() {
                let r = Some((&(*self.sentinel_one_ptr).0, &mut (*self.sentinel_one_ptr).1));
                self.todo -= 1;
                self.sentinel_one_ptr = ptr::null_mut();
                return r;
            }
        }
        unsafe {
            while self.cur < self.end {
                if K::thin_sentinel_zero() != (*self.cur).0 && K::thin_sentinel_one() != (*self.cur).0 {
                    let r = Some((&(*self.cur).0, &mut (*self.cur).1));
                    self.todo -= 1;
                    self.cur = self.cur.add(1);
                    return r;
                }
                self.cur = self.cur.add(1);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.todo, Some(self.todo))
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct Keys<'a, K: 'a, V: 'a>
{
    inner: Iter<'a, K, V>
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V>
    where K: ThinSentinel + Eq
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for Keys<'a, K, V>
    where K: ThinSentinel + Eq {}

impl<'a, K, V> Iterator for Keys<'a, K, V>
    where K: ThinSentinel + Eq
{
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        let o: Option<(&K, &V)> = self.inner.next();
        if o.is_some() {
            let x: (&K, &V) = o.unwrap();
            return Some(x.0);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct Values<'a, K: 'a, V: 'a>
{
    inner: Iter<'a, K, V>
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V>
    where K: ThinSentinel + Eq,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for Values<'a, K, V>
    where K: ThinSentinel + Eq {}

impl<'a, K, V> Iterator for Values<'a, K, V>
    where K: ThinSentinel + Eq,
{
    type Item = &'a V;

    fn next(&mut self) -> Option<&'a V> {
        let o: Option<(&K, &V)> = self.inner.next();
        if o.is_some() {
            let x: (&K, &V) = o.unwrap();
            return Some(x.1);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[doc(hidden)]
pub struct ValuesMut<'a, K: 'a, V: 'a>
{
    inner: IterMut<'a, K, V>
}

impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V>
    where K: ThinSentinel + Eq,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V>
    where K: ThinSentinel + Eq, {}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V>
    where K: ThinSentinel + Eq,
{
    type Item = &'a mut V;

    fn next(&mut self) -> Option<&'a mut V> {
        let o: Option<(&K, &mut V)> = self.inner.next();
        if o.is_some() {
            let x: (&K, &mut V) = o.unwrap();
            return Some(x.1);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K: ThinSentinel + Eq + Hash, V, H: BuildHasher> ThinMap<K, V, H> {
    /// Creates an empty `ThinMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not heap-allocate until it
    /// is first inserted into. On first insert, enough room is reserved for 8 pairs.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// use thincollections::thin_hasher::OneFieldHasherBuilder;
    ///
    /// let s = OneFieldHasherBuilder::new();
    /// let mut map = ThinMap::with_hasher(s);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    pub fn with_hasher(hash_builder: H) -> ThinMap<K, V, H> {
        ThinMap::with_capacity_and_hasher(0, hash_builder)
    }

    /// Creates an empty `ThinMap` with the specified capacity, using `hash_builder`
    /// to hash the keys.
    ///
    /// The map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the map will not heap-allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// use thincollections::thin_hasher::OneFieldHasherBuilder;
    ///
    /// let s = OneFieldHasherBuilder::new();
    /// let mut map: ThinMap<u64, i32, OneFieldHasherBuilder> = ThinMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> ThinMap<K, V, H> {
        if capacity == 0 {
            return ThinMap {
                table_size: 0,
                table: ptr::null_mut(),
                occupied: 0,
                sentinels: 0,
                occupied_sentinels: 0,
                hasher: hash_builder,
                _marker: marker::PhantomData,
            };
        }
        let size = (ceil_pow2(capacity as u64) << 1) as usize;
        let buffer = <ThinMap<K, V, H>>::allocate_table_for_size(size);
        ThinMap {
            table_size: size,
            occupied: 0,
            sentinels: 0,
            occupied_sentinels: 0,
            table: buffer,
            hasher: hash_builder,
            _marker: marker::PhantomData,
        }
    }
}

impl<K: ThinSentinel + Eq + Hash, V> ThinMap<K, V, OneFieldHasherBuilder> {
    /// Creates an empty `ThinMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not heap-allocate until it
    /// is first inserted into. On first insert, enough room is reserved for 8 pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// let mut map: ThinMap<u64, i32> = ThinMap::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        ThinMap {
            table_size: 0,
            occupied: 0,
            sentinels: 0,
            occupied_sentinels: 0,
            table: ptr::null_mut(),
            hasher: OneFieldHasherBuilder::new(),
            _marker: marker::PhantomData,
        }
    }

    /// Creates an empty `ThinMap` with the specified capacity, using OneFieldHasherBuilder
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the map will not heap-allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// let mut map: ThinMap<u64, i32> = ThinMap::with_capacity(10);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return ThinMap::new();
        }
        let size = (ceil_pow2(capacity as u64) << 1) as usize;
        let buffer = <ThinMap<K, V, OneFieldHasherBuilder>>::allocate_table_for_size(size);
        ThinMap {
            table_size: size,
            occupied: 0,
            sentinels: 0,
            occupied_sentinels: 0,
            table: buffer,
            hasher: OneFieldHasherBuilder::new(),
            _marker: marker::PhantomData,
        }
    }
}

#[derive(PartialEq)]
enum BucketState {
    FULL,
    EMPTY,
    REMOVED,
}

impl BucketState {
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        *self == BucketState::FULL
    }
}

impl<K: ThinSentinel + Eq + Hash, V, H: BuildHasher> ThinMap<K, V, H> {
    /// Returns a reference to the map's [`BuildHasher`].
    ///
    /// [`BuildHasher`]: ../../std/hash/trait.BuildHasher.html
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// use thincollections::thin_hasher::OneFieldHasherBuilder;
    ///
    /// let s = OneFieldHasherBuilder::new();
    /// let map: ThinMap<i32, u64> = ThinMap::with_hasher(s);
    /// let hasher: &OneFieldHasherBuilder = map.hasher();
    /// ```
    #[inline]
    pub fn hasher(&self) -> &H {
        &self.hasher
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the `ThinMap<K, V>` might be able to hold
    /// more, but is guaranteed to be able to hold at least this many.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// let map: ThinMap<i32, i32> = ThinMap::with_capacity(100);
    /// assert!(map.capacity() >= 100);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.max_occupied()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `ThinMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows [`usize`].
    ///
    /// [`usize`]: ../../std/primitive.usize.html
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    /// let mut map: ThinMap<u64, i32> = ThinMap::new();
    /// map.reserve(10);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        if additional == 0 { return; }
        if self.table_size == 0 {
            let new_size = (ceil_pow2(self.occupied as u64 + additional as u64) << 1) as usize;
            self.table = <ThinMap<K, V, H>>::allocate_table_for_size(new_size);
            self.table_size = new_size;
        } else if self.capacity() - self.occupied < additional {
            let new_size = (ceil_pow2(self.occupied as u64 + additional as u64) << 1) as usize;
            self.rehash_for_size(new_size);
        }
    }

    /// Shrinks the capacity of the map as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map: ThinMap<i32, i32> = ThinMap::with_capacity(100);
    /// map.insert(7, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to_fit();
    /// assert!(map.capacity() < 100);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        if self.capacity() > self.occupied {
            if self.is_empty() {
                unsafe {
                    let layout = Layout::from_size_align(mem::size_of::<(K, V)>() * (self.table_size + 2), mem::align_of::<(K, V)>()).unwrap();
                    alloc::dealloc(self.table.offset(-2) as *mut u8, layout);
                    self.table_size = 0;
                    self.sentinels = 0;
                    self.table = ptr::null_mut();
                }
            } else {
                let new_size = (ceil_pow2(self.occupied as u64) << 1) as usize;
                self.rehash_for_size(new_size);
            }
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut letters = ThinMap::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     let counter = letters.entry(ch).or_insert(0);
    ///     *counter += 1;
    /// }
    ///
    /// assert_eq!(letters[&'s'], 2);
    /// assert_eq!(letters[&'t'], 3);
    /// assert_eq!(letters[&'u'], 1);
    /// assert_eq!(letters.get(&'y'), None);
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        if self.table_size == 0 {
            self.allocate_table();
        }
        unsafe {
            if K::thin_sentinel_zero() == key {
                let ptr = self.table.offset(-2);
                if (*ptr).0 == K::thin_sentinel_zero() {
                    return Entry::Occupied(OccupiedEntry {
                        ptr: ptr.as_mut().unwrap(),
                        occupied: &mut self.occupied,
                        sentinels: &mut self.sentinels,
                        occupied_sentinels: &mut self.occupied_sentinels,
                    });
                }
                return Entry::Vacant(VacantEntry {
                    ptr: ptr.as_mut().unwrap(),
                    occupied: &mut self.occupied,
                    sentinels: &mut self.sentinels,
                    occupied_sentinels: &mut self.occupied_sentinels,
                    key: key,
                });
            }
            if K::thin_sentinel_one() == key {
                let ptr = self.table.offset(-1);
                if (*ptr).0 == K::thin_sentinel_one() {
                    return Entry::Occupied(OccupiedEntry {
                        ptr: ptr.as_mut().unwrap(),
                        occupied: &mut self.occupied,
                        sentinels: &mut self.sentinels,
                        occupied_sentinels: &mut self.occupied_sentinels,
                    });
                }
                return Entry::Vacant(VacantEntry {
                    ptr: ptr.as_mut().unwrap(),
                    occupied: &mut self.occupied,
                    sentinels: &mut self.sentinels,
                    occupied_sentinels: &mut self.occupied_sentinels,
                    key: key,
                });
            }
        }
        let (entry, bucket_state) = self.probe(&key);
        unsafe {
            if bucket_state.is_full() {
                return Entry::Occupied(OccupiedEntry {
                    ptr: entry.as_mut().unwrap(),
                    occupied: &mut self.occupied,
                    sentinels: &mut self.sentinels,
                    occupied_sentinels: &mut self.occupied_sentinels,
                });
            }
            return Entry::Vacant(VacantEntry {
                ptr: entry.as_mut().unwrap(),
                occupied: &mut self.occupied,
                sentinels: &mut self.sentinels,
                occupied_sentinels: &mut self.occupied_sentinels,
                key: key,
            });
        }
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut a = ThinMap::new();
    /// a.insert(1, 100);
    /// a.insert(2, 101);
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == 100 || v == 200);
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) -> Drain<K, V> {
        if self.is_empty() {
            return Drain {
                sentinel_zero_ptr: ptr::null_mut(),
                sentinel_one_ptr: ptr::null_mut(),
                cur: ptr::null_mut(),
                end: ptr::null_mut(),
                occupied: &mut self.occupied,
                sentinels: &mut self.sentinels,
                occupied_sentinels: &mut self.occupied_sentinels,
                _marker: marker::PhantomData,
            };
        }
        unsafe {
            let mut zero_ptr = self.table.offset(-2);
            if (*zero_ptr).0 != K::thin_sentinel_zero() {
                zero_ptr = ptr::null_mut();
            }
            let mut one_ptr = self.table.offset(-1);
            if (*one_ptr).0 != K::thin_sentinel_one() {
                one_ptr = ptr::null_mut();
            }
            Drain {
                sentinel_zero_ptr: zero_ptr,
                sentinel_one_ptr: one_ptr,
                cur: self.table,
                end: self.table.add(self.table_size),
                occupied: &mut self.occupied,
                sentinels: &mut self.sentinels,
                occupied_sentinels: &mut self.occupied_sentinels,
                _marker: marker::PhantomData,
            }
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(101, 1);
    /// map.insert(202, 2);
    /// map.insert(303, 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<K, V> {
        if self.is_empty() {
            return Iter {
                sentinel_zero_ptr: ptr::null_mut(),
                sentinel_one_ptr: ptr::null_mut(),
                todo: 0,
                cur: ptr::null_mut(),
                end: ptr::null_mut(),
                _marker: marker::PhantomData,
            };
        }
        unsafe {
            let mut zero_ptr = self.table.offset(-2);
            if (*zero_ptr).0 != K::thin_sentinel_zero() {
                zero_ptr = ptr::null_mut();
            }
            let mut one_ptr = self.table.offset(-1);
            if (*one_ptr).0 != K::thin_sentinel_one() {
                one_ptr = ptr::null_mut();
            }
            Iter {
                sentinel_zero_ptr: zero_ptr,
                sentinel_one_ptr: one_ptr,
                todo: self.len(),
                cur: self.table,
                end: self.table.add(self.table_size),
                _marker: marker::PhantomData,
            }
        }
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(100, 1);
    /// map.insert(-100, 2);
    /// map.insert(17, 3);
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values(&self) -> Values<K, V> {
        Values {
            inner: self.iter()
        }
    }

    /// An iterator visiting all values mutably in arbitrary order.
    /// The iterator element type is `&'a mut V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    ///
    /// map.insert(100, 1);
    /// map.insert(101, 2);
    /// map.insert(102, 3);
    ///
    /// for val in map.values_mut() {
    ///     *val = *val + 10;
    /// }
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        ValuesMut { inner: self.iter_mut() }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(700, 1);
    /// map.insert(800, 2);
    /// map.insert(900, 3);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            inner: self.iter()
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(0, 1);
    /// map.insert(1, 2);
    /// map.insert(4, 3);
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in &map {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        if self.is_empty() {
            return IterMut {
                sentinel_zero_ptr: ptr::null_mut(),
                sentinel_one_ptr: ptr::null_mut(),
                todo: 0,
                cur: ptr::null_mut(),
                end: ptr::null_mut(),
                _marker: marker::PhantomData,
            };
        }
        unsafe {
            let mut zero_ptr = self.table.offset(-2);
            if (*zero_ptr).0 != K::thin_sentinel_zero() {
                zero_ptr = ptr::null_mut();
            }
            let mut one_ptr = self.table.offset(-1);
            if (*one_ptr).0 != K::thin_sentinel_one() {
                one_ptr = ptr::null_mut();
            }
            IterMut {
                sentinel_zero_ptr: zero_ptr,
                sentinel_one_ptr: one_ptr,
                todo: self.len(),
                cur: self.table,
                end: self.table.add(self.table_size),
                _marker: marker::PhantomData,
            }
        }
    }

    /// Returns true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut a = ThinMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, 2);
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut a = ThinMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, 17);
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.occupied_sentinels as usize + self.occupied
    }

    #[inline]
    fn insert_sentinel(&mut self, offset: isize, full_key: K, value: V) -> Option<V> {
        unsafe {
            let sentinel_ptr: *mut (K, V) = self.table.offset(offset);
            if (*sentinel_ptr).0 == full_key {
                return Some(mem::replace(&mut ((*sentinel_ptr).1), value));
            }
            self.occupied_sentinels += 1;
            ptr::write(sentinel_ptr, (full_key, value));
            None
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// assert_eq!(map.insert(37, 123), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, 289);
    /// assert_eq!(map.insert(37, 333), Some(289));
    /// assert_eq!(map[&37], 333);
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.table_size == 0 {
            self.allocate_table();
        }
        if K::thin_sentinel_zero() == key {
            return self.insert_sentinel(-2, K::thin_sentinel_zero(), value);
        } else if K::thin_sentinel_one() == key {
            return self.insert_sentinel(-1, K::thin_sentinel_one(), value);
        }
        let state;
        {
            let (entry, bucket_state) = self.probe(&key);
            unsafe {
                if bucket_state.is_full() {
                    return Some(mem::replace(&mut (*entry).1, value));
                }
                state = bucket_state;
                ptr::write(entry, (key, value));
            }
        }
        if BucketState::REMOVED == state {
            self.sentinels -= 1;
        }
        self.occupied += 1;
        if self.occupied + self.sentinels > self.max_occupied() {
            self.rehash();
        }
        return None;
    }

    fn rehash(&mut self) {
        let max = self.max_occupied();
        let mut new_size = cmp::max(max, ceil_pow2(((self.occupied + 1) << 1) as u64) as usize);
        if self.sentinels > 0 && (max >> 1) + (max >> 2) < self.occupied {
            new_size = new_size << 1;
        }
        self.rehash_for_size(new_size);
    }

    fn rehash_for_size(&mut self, new_size: usize) {
        let old_table = self.table;
        let old_size = self.table_size;
        self.table = <ThinMap<K, V, H>>::allocate_table_for_size(new_size);
        self.table_size = new_size;
        self.sentinels = 0;
//        println!("old size {} new size {} old table {:?} new table {:?} entry size {}", old_size, new_size,
//                 old_table as usize, self.table as usize, mem::size_of::<(K, V)>());
//        println!("should be empty {:?}", &self);
        unsafe {
            let sentinel_ptr: *mut (K, V) = old_table.offset(-2);
            ptr::copy_nonoverlapping(sentinel_ptr, self.table.offset(-2), 2);
            let mut ptr: *mut (K, V) = old_table;
            let table_end = old_table.add(old_size);
            while ptr < table_end {
//                println!("considering {:?} {:?}", &(*ptr).0, &(*ptr).1);
                if K::thin_sentinel_zero() != (*ptr).0 && K::thin_sentinel_one() != (*ptr).0 {
                    let (entry, _bucket_state) = self.probe(&(*ptr).0);
                    ptr::copy_nonoverlapping(ptr, entry, 1);
//                    println!("copied {:?} {:?} {:?}", &(*ptr).0, &(*ptr).1, &self);
                }
                ptr = ptr.add(1);
            }
            let layout = Layout::from_size_align(mem::size_of::<(K, V)>() * (old_size + 2),
                                                 mem::align_of::<(K, V)>()).unwrap();
            alloc::dealloc(old_table.offset(-2) as *mut u8, layout);
        }
//        println!("should have everything {:?}", &self);
    }

    fn allocate_table(&mut self) {
        self.table = <ThinMap<K, V, H>>::allocate_table_for_size(16);
        self.table_size = 16;
    }

    fn allocate_table_for_size(size: usize) -> *mut (K, V) {
        unsafe {
            let layout = Layout::from_size_align(mem::size_of::<(K, V)>() * (size + 2), mem::align_of::<(K, V)>()).unwrap();
            let buffer = alloc::alloc(layout);
            let table = buffer as *mut (K, V);
            {
                let table_end = table.add(size + 2);
                let mut ptr = table;
                while ptr < table_end {
                    ptr::write(ptr as *mut K, K::thin_sentinel_zero());
                    ptr = ptr.add(1);
                }
            }
            overwrite_k(table, K::thin_sentinel_one());
            table.add(2)
        }
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

    #[inline]
    fn max_occupied(&self) -> usize {
        (self.table_size >> 1) + (self.table_size >> 3)
    }

    #[inline]
    fn probe(&self, key: &K) -> (*mut (K, V), BucketState) {
//        let hash = spread_one(self.hash(&key));
//        let index = self.mask(hash) as isize;
        let (hash, index) = self.hash_and_mask(key);
        unsafe {
            let mut ptr: *mut (K, V) = self.table.offset(index);
            if K::thin_sentinel_zero() == (*ptr).0 {
                return (ptr, BucketState::EMPTY);
            }
            if (*ptr).0 == *key {
                return (ptr, BucketState::FULL);
            }

            let mut removed_ptr: *mut (K, V) = if K::thin_sentinel_one() == (*ptr).0 { ptr } else { ptr::null_mut() };
            let table_end: usize = self.table.add(self.table_size) as usize;
            let mut end_ptr: usize = ((ptr as usize) & !63) + 64;
            if table_end < end_ptr { end_ptr = table_end; }
            ptr = ptr.add(1);
            while (ptr as usize) < end_ptr {
                if (*ptr).0 == *key {
                    return (ptr, BucketState::FULL);
                }
                if K::thin_sentinel_zero() == (*ptr).0 {
                    if removed_ptr.is_null() {
                        return (ptr, BucketState::EMPTY);
                    } else {
                        return (removed_ptr, BucketState::REMOVED);
                    }
                }
                if K::thin_sentinel_one() == (*ptr).0 && removed_ptr.is_null() {
                    removed_ptr = ptr;
                }
                ptr = ptr.add(1);
            }
            self.probe2(key, removed_ptr, hash)
        }
    }

    #[cold]
    fn probe2(&self, key: &K, mut removed_ptr: *mut (K, V), hash: u64) -> (*mut (K, V), BucketState) {
        let index = self.spread_two_and_mask(hash);
        unsafe {
            let mut ptr: *mut (K, V) = self.table.offset(index);
            let table_end: usize = self.table.add(self.table_size) as usize;
            let mut end_ptr: usize = ((ptr as usize) & !63) + 64;
            if table_end < end_ptr { end_ptr = table_end; }
            while (ptr as usize) < end_ptr {
                if (*ptr).0 == *key {
                    return (ptr, BucketState::FULL);
                }
                if K::thin_sentinel_zero() == (*ptr).0 {
                    if removed_ptr.is_null() {
                        return (ptr, BucketState::EMPTY);
                    } else {
                        return (removed_ptr, BucketState::REMOVED);
                    }
                }
                if K::thin_sentinel_one() == (*ptr).0 && removed_ptr.is_null() {
                    removed_ptr = ptr;
                }
                ptr = ptr.add(1);
            }
            self.probe3(key, removed_ptr, hash)
        }
    }

    //    #[cold]
    fn probe3(&self, key: &K, mut removed_ptr: *mut (K, V), hash: u64) -> (*mut (K, V), BucketState) {
        let mut next_index = spread_one(hash) as isize;
        let spread_two = spread_two(hash).rotate_right(32) | 1;

        loop {
            unsafe {
                next_index = self.mask((next_index as u64).wrapping_add(spread_two)) as isize;
                let ptr: *mut (K, V) = self.table.offset(next_index);
                if (*ptr).0 == *key {
                    return (ptr, BucketState::FULL);
                }
                if K::thin_sentinel_zero() == (*ptr).0 {
                    if removed_ptr.is_null() {
                        return (ptr, BucketState::EMPTY);
                    } else {
                        return (removed_ptr, BucketState::REMOVED);
                    }
                }
                if K::thin_sentinel_one() == (*ptr).0 && removed_ptr.is_null() {
                    removed_ptr = ptr;
                }
            }
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(1, 17);
    /// map.insert(7, 42);
    /// assert_eq!(map.get_key_value(&1), Some((&1, &17)));
    /// assert_eq!(map.get_key_value(&7), Some((&7, &42)));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    //note: std uses Borrow here
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }
        if K::thin_sentinel_zero() == *key {
            unsafe {
                let ptr = self.table.offset(-2);
                if (*ptr).0 == K::thin_sentinel_zero() {
                    return Some((&(*ptr).0, &(*ptr).1));
                }
                return None;
            }
        } else if K::thin_sentinel_one() == *key {
            unsafe {
                let ptr = self.table.offset(-1);
                if (*ptr).0 == K::thin_sentinel_one() {
                    return Some((&(*ptr).0, &(*ptr).1));
                }
                return None;
            }
        }
        let (entry, state) = self.probe(key);
        unsafe {
            if state.is_full() {
                return Some((&(*entry).0, &(*entry).1));
            }
        }
        None
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(1, 200);
    /// map.insert(7, 300);
    /// assert_eq!(map.get(&1), Some(&200));
    /// assert_eq!(map.get(&7), Some(&300));
    /// assert_eq!(map.get(&2), None);
    /// ```
    //note: std uses Borrow here
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.is_empty() {
            return None;
        }
        let got_it: bool;
        let ptr: *mut (K, V);
        if K::thin_sentinel_zero() == *key {
            unsafe {
                ptr = self.table.offset(-2);
                got_it = (*ptr).0 == K::thin_sentinel_zero();
            }
        } else if K::thin_sentinel_one() == *key {
            unsafe {
                ptr = self.table.offset(-1);
                got_it = (*ptr).0 == K::thin_sentinel_one();
            }
        } else {
            let (entry, state) = self.probe(key);
            ptr = entry;
            got_it = state.is_full();
        }
        unsafe {
            if got_it {
                return Some(&(*ptr).1);
            }
        }
        None
    }


    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(1, 200);
    /// map.insert(7, 300);
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = 42;
    /// }
    /// assert_eq!(map[&1], 42);
    /// ```
    //note: std uses Borrow here
    pub fn get_mut(&self, key: &K) -> Option<&mut V> {
        if self.is_empty() {
            return None;
        }
        if K::thin_sentinel_zero() == *key {
            unsafe {
                let ptr = self.table.offset(-2);
                if (*ptr).0 == K::thin_sentinel_zero() {
                    return Some(&mut (*ptr).1);
                }
                return None;
            }
        } else if K::thin_sentinel_one() == *key {
            unsafe {
                let ptr = self.table.offset(-1);
                if (*ptr).0 == K::thin_sentinel_one() {
                    return Some(&mut (*ptr).1);
                }
                return None;
            }
        }
        let (entry, state) = self.probe(key);
        unsafe {
            if state.is_full() {
                return Some(&mut (*entry).1);
            }
        }
        None
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(1, 200);
    /// map.insert(7, 42);
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&7), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key(&self, key: &K) -> bool {
        if self.is_empty() {
            return false;
        }
        if K::thin_sentinel_zero() == *key {
            unsafe {
                let ptr = self.table.offset(-2);
                return (*ptr).0 == K::thin_sentinel_zero();
            }
        } else if K::thin_sentinel_one() == *key {
            unsafe {
                let ptr = self.table.offset(-1);
                return (*ptr).0 == K::thin_sentinel_one();
            }
        }
        let (_entry, state) = self.probe(key);
        return state.is_full();
    }

    fn remove_sentinel(&mut self, offset: isize, full_key: K, empty_key: K) -> Option<V> {
        unsafe {
            let ptr: *mut (K, V) = self.table.offset(offset);
            if (*ptr).0 == full_key {
                overwrite_k(ptr, empty_key);
                self.occupied_sentinels -= 1;
                return Some(ptr::read(&(*ptr).1 as *const V));
            }
        }
        None
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(7, 123);
    /// assert_eq!(1, map.len());
    /// assert_eq!(map.remove(&7), Some(123));
    /// assert_eq!(map.remove(&7), None);
    /// assert!(map.is_empty());
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.is_empty() {
            return None;
        }
        if K::thin_sentinel_zero() == *key {
            return self.remove_sentinel(-2, K::thin_sentinel_zero(), K::thin_sentinel_one());
        } else if K::thin_sentinel_one() == *key {
            return self.remove_sentinel(-1, K::thin_sentinel_one(), K::thin_sentinel_zero());
        }
        let mut r: Option<V> = None;
        {
            let (entry, state) = self.probe(key);
            if state.is_full() {
                unsafe {
                    overwrite_k(entry, K::thin_sentinel_one());
                    r = Some(ptr::read(&(*entry).1 as *const V));
                }
            }
        }
        if r.is_some() {
            self.sentinels += 1;
            self.occupied -= 1;
        }
        r
    }


    fn remove_sentinel_entry(&mut self, offset: isize, full_key: K, empty_key: K) -> Option<(K, V)> {
        unsafe {
            let ptr: *mut (K, V) = self.table.offset(offset);
            if (*ptr).0 == full_key {
                overwrite_k(ptr, empty_key);
                self.occupied_sentinels -= 1;
                return Some((full_key, ptr::read(&(*ptr).1 as *const V)));
            }
        }
        None
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    ///
    /// map.insert(7, 123);
    /// assert_eq!(map.remove_entry(&7), Some((7, 123)));
    /// assert_eq!(map.remove(&7), None);
    /// ```
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
        if self.is_empty() {
            return None;
        }
        if K::thin_sentinel_zero() == *key {
            return self.remove_sentinel_entry(-2, K::thin_sentinel_zero(), K::thin_sentinel_one());
        } else if K::thin_sentinel_one() == *key {
            return self.remove_sentinel_entry(-1, K::thin_sentinel_one(), K::thin_sentinel_zero());
        }
        let mut r: Option<(K, V)> = None;
        {
            let (entry, state) = self.probe(key);
            if state.is_full() {
                unsafe {
                    r = Some(ptr::read(entry));
                    overwrite_k(entry, K::thin_sentinel_one());
                }
            }
        }
        if r.is_some() {
            self.sentinels += 1;
            self.occupied -= 1;
        }
        r
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` such that `f(&k,&mut v)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map: ThinMap<i32, i32> = (0..8).map(|x|(x, x*10)).collect();
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    pub fn retain<F>(&mut self, mut retain_fn: F)
        where F: FnMut(&K, &mut V) -> bool
    {
        if self.is_empty() {
            return;
        }
        unsafe {
            if self.occupied_sentinels > 0 {
                let mut ptr: *mut (K, V) = self.table.offset(-2);
                if (*ptr).0 == K::thin_sentinel_zero() {
                    if !retain_fn(&(*ptr).0, &mut (*ptr).1) {
                        ptr::drop_in_place(ptr);
                        overwrite_k(ptr, K::thin_sentinel_one());
                        self.occupied_sentinels -= 1;
                    }
                }
                ptr = ptr.add(1);
                if (*ptr).0 == K::thin_sentinel_one() {
                    if !retain_fn(&(*ptr).0, &mut (*ptr).1) {
                        ptr::drop_in_place(ptr);
                        overwrite_k(ptr, K::thin_sentinel_zero());
                        self.occupied_sentinels -= 1;
                    }
                }
            }
            if self.occupied > 0 {
                let mut ptr: *mut (K, V) = self.table;
                let table_end = self.table.add(self.table_size);
                while ptr < table_end {
                    if K::thin_sentinel_zero() != (*ptr).0 && K::thin_sentinel_one() != (*ptr).0 && !retain_fn(&(*ptr).0, &mut (*ptr).1) {
                        ptr::drop_in_place(ptr);
                        overwrite_k(ptr, K::thin_sentinel_one());
                        self.occupied -= 1;
                        self.sentinels += 1;
                    }
                    ptr = ptr.add(1);
                }
            }
        }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut a = ThinMap::new();
    /// a.insert(1, 20);
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        if self.table_size > 0 && !self.is_empty() {
            unsafe {
                if self.occupied_sentinels > 0 {
                    let mut ptr: *mut (K, V) = self.table.offset(-2);
                    if (*ptr).0 == K::thin_sentinel_zero() {
                        ptr::drop_in_place(ptr);
                        overwrite_k(ptr, K::thin_sentinel_one());
                    }
                    ptr = ptr.add(1);
                    if (*ptr).0 == K::thin_sentinel_one() {
                        ptr::drop_in_place(ptr);
                        overwrite_k(ptr, K::thin_sentinel_zero());
                    }
                }
                if self.occupied > 0 {
                    let mut ptr: *mut (K, V) = self.table;
                    let table_end = self.table.add(self.table_size);
                    while ptr < table_end {
                        if K::thin_sentinel_zero() != (*ptr).0 {
                            if K::thin_sentinel_one() != (*ptr).0 {
                                ptr::drop_in_place(ptr);
                            }
                            overwrite_k(ptr, K::thin_sentinel_zero());
                        }
                        ptr = ptr.add(1);
                    }
                }
            }
        }
        self.occupied = 0;
        self.occupied_sentinels = 0;
        self.sentinels = 0;
    }

    #[inline(always)]
    fn mask(&self, hash: u64) -> u64 {
        hash & ((self.table_size - 1) as u64)
    }

    fn spread_two_and_mask(&self, hash: u64) -> isize {
        self.mask(spread_two(hash)) as isize
    }
}

impl<K: ThinSentinel + Eq + Hash, V, H: BuildHasher> Drop for ThinMap<K, V, H> {
    fn drop(&mut self) {
        if self.table_size > 0 {
            unsafe {
                if mem::needs_drop::<(K, V)>() {
                    if self.occupied_sentinels > 0 {
                        let mut ptr: *mut (K, V) = self.table.offset(-2);
                        if (*ptr).0 == K::thin_sentinel_zero() {
                            ptr::drop_in_place(ptr);
                        }
                        ptr = ptr.add(1);
                        if (*ptr).0 == K::thin_sentinel_one() {
                            ptr::drop_in_place(ptr);
                        }
                    }
                    if self.occupied > 0 {
                        let mut ptr: *mut (K, V) = self.table;
                        let table_end = self.table.add(self.table_size);
                        while ptr < table_end {
                            if K::thin_sentinel_zero() != (*ptr).0 && K::thin_sentinel_one() != (*ptr).0 {
                                ptr::drop_in_place(ptr);
                            }
                            ptr = ptr.add(1);
                        }
                    }
                }
                let layout = Layout::from_size_align(mem::size_of::<(K, V)>() * (self.table_size + 2), mem::align_of::<(K, V)>()).unwrap();
                alloc::dealloc(self.table.offset(-2) as *mut u8, layout);
            }
        }
    }
}

impl<'a, K: 'a, V, S> Index<&'a K> for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `ThinMap`.
    #[inline]
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, S> Debug for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel + Debug,
          V: Debug,
          S: BuildHasher
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug_map = f.debug_map();
        if self.table_size > 0 {
            unsafe {
                let mut ptr: *mut (K, V) = self.table.offset(-2);
                if (*ptr).0 == K::thin_sentinel_zero() {
                    debug_map.entry(&(*ptr).0, &(*ptr).1);
                }
                ptr = ptr.add(1);
                if (*ptr).0 == K::thin_sentinel_one() {
                    debug_map.entry(&(*ptr).0, &(*ptr).1);
                }
            }
            unsafe {
                let mut ptr: *mut (K, V) = self.table;
                let table_end = self.table.add(self.table_size);
                while ptr < table_end {
                    if K::thin_sentinel_zero() != (*ptr).0 && K::thin_sentinel_one() != (*ptr).0 {
                        debug_map.entry(&(*ptr).0, &(*ptr).1);
                    }
                    ptr = ptr.add(1);
                }
            }
        }
        debug_map.finish()
    }
}

impl<K, V, S> ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel + Debug, V: Debug,
          S: BuildHasher
{
    pub fn debug(&self) {
        if self.table_size == 0 {
            println!("[]");
        }
        println!("occupied {}, sentinels {}, occupied_sentinels {}", self.occupied, self.sentinels, self.occupied_sentinels);
        unsafe {
            let mut ptr: *mut (K, V) = self.table.offset(-2);
            if (*ptr).0 == K::thin_sentinel_zero() {
                println!("[{:?},{:?}]", &(*ptr).0, &(*ptr).1);
            }
            ptr = ptr.add(1);
            if (*ptr).0 == K::thin_sentinel_one() {
                println!("[{:?},{:?}]", &(*ptr).0, &(*ptr).1);
            }
        }
        unsafe {
            let mut ptr: *mut (K, V) = self.table;
            let table_end = self.table.add(self.table_size);
            while ptr < table_end {
                if K::thin_sentinel_zero() != (*ptr).0 && K::thin_sentinel_one() != (*ptr).0 {
                    println!("[{:?},{:?}]", &(*ptr).0, &(*ptr).1);
                }
                ptr = ptr.add(1);
            }
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
    where K: Eq + Hash + ThinSentinel,
{
    pub fn remove_entry(self) -> (K, V) {
        let sen;
        let rem;
        if K::thin_sentinel_one() == (*self.ptr).0 {
            rem = K::thin_sentinel_zero();
            sen = true;
        } else {
            rem = K::thin_sentinel_one();
            sen = K::thin_sentinel_zero() == (*self.ptr).0;
        }
        unsafe {
            let r = ptr::read(self.ptr);
            overwrite_k(self.ptr, rem);
            if sen {
                *self.occupied_sentinels -= 1;
            } else {
                *self.occupied -= 1;
                *self.sentinels += 1;
            }
            r
        }
    }

    pub fn get(&self) -> &V {
        &(*self.ptr).1
    }

    pub fn insert(&mut self, mut value: V) -> V {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }

    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    pub fn into_mut(self) -> &'a mut V {
        &mut (*self.ptr).1
    }

    pub fn get_mut(&mut self) -> &mut V {
        &mut (*self.ptr).1
    }

    pub fn key(&self) -> &K {
        &(*self.ptr).0
    }
}

impl<'a, K, V> VacantEntry<'a, K, V>
    where K: Eq + Hash + ThinSentinel,
{
    pub fn insert(self, value: V) -> &'a mut V {
        if self.is_sentry() {
            *self.occupied_sentinels += 1;
        } else {
            if (*self.ptr).0 == K::thin_sentinel_one() {
                *self.sentinels -= 1;
            }
            *self.occupied += 1;
        }
        unsafe {
            ptr::write(self.ptr, (self.key, value));
        }
        &mut (*self.ptr).1
    }

    #[inline]
    fn is_sentry(&self) -> bool {
        self.key == K::thin_sentinel_one() || self.key == K::thin_sentinel_zero()
    }

    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<'a, K, V> Entry<'a, K, V>
    where K: Eq + Hash + ThinSentinel,
{
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
        where F: FnOnce(&mut V)
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }
}

impl<'a, K, V> Entry<'a, K, V>
    where K: Eq + Hash + ThinSentinel,
          V: Default
{
    pub fn or_default(self) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

#[doc(hidden)]
pub struct IntoIter<K: Eq + ThinSentinel, V> {
    sentinel_zero_ptr: *mut (K, V),
    sentinel_one_ptr: *mut (K, V),
    table: *mut (K, V),
    table_size: usize,
    left: usize,
    cur: *mut (K, V),
    end: *mut (K, V),
    _marker: marker::PhantomData<(K, V)>,
}

impl<K, V, S> IntoIterator for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_map::ThinMap;
    ///
    /// let mut map = ThinMap::new();
    /// map.insert(100, 1);
    /// map.insert(101, 2);
    /// map.insert(102, 3);
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(u8, i32)> = map.into_iter().collect();
    /// ```
    fn into_iter(self) -> IntoIter<K, V> {
        if self.table_size == 0 {
            return IntoIter {
                sentinel_zero_ptr: ptr::null_mut(),
                sentinel_one_ptr: ptr::null_mut(),
                table: ptr::null_mut(),
                table_size: 0,
                left: 0,
                cur: ptr::null_mut(),
                end: ptr::null_mut(),
                _marker: marker::PhantomData,
            };
        }
        unsafe {
            let mut zero_ptr = self.table.offset(-2);
            if (*zero_ptr).0 != K::thin_sentinel_zero() {
                zero_ptr = ptr::null_mut();
            }
            let mut one_ptr = self.table.offset(-1);
            if (*one_ptr).0 != K::thin_sentinel_one() {
                one_ptr = ptr::null_mut();
            }
            let iter = IntoIter {
                sentinel_zero_ptr: zero_ptr,
                sentinel_one_ptr: one_ptr,
                table: self.table,
                table_size: self.table_size,
                left: self.len(),
                cur: self.table,
                end: self.table.add(self.table_size),
                _marker: marker::PhantomData,
            };
            mem::forget(self);
            iter
        }
    }
}

//// todo: is this appropriate?
////unsafe impl<K: Sync, V: Sync> Sync for IntoIter<K, V> {}
////unsafe impl<K: Send, V: Send> Send for IntoIter<K, V> {}
//
impl<K: ThinSentinel + Eq, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        unsafe {
            if !self.sentinel_zero_ptr.is_null() {
                let r = Some(ptr::read(self.sentinel_zero_ptr));
                self.left -= 1;
                self.sentinel_zero_ptr = ptr::null_mut();
                return r;
            }
            if !self.sentinel_one_ptr.is_null() {
                let r = Some(ptr::read(self.sentinel_one_ptr));
                self.left -= 1;
                self.sentinel_one_ptr = ptr::null_mut();
                return r;
            }
        }
        unsafe {
            while self.cur < self.end {
                if K::thin_sentinel_zero() != (*self.cur).0 && K::thin_sentinel_one() != (*self.cur).0 {
                    let r = Some(ptr::read(self.cur));
                    self.left -= 1;
                    self.cur = self.cur.add(1);
                    return r;
                }
                self.cur = self.cur.add(1);
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl<K: ThinSentinel + Eq, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.left
    }
}

impl<K: ThinSentinel + Eq, V> FusedIterator for IntoIter<K, V> {}

impl<K: ThinSentinel + Eq, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        if !self.table.is_null() {
            unsafe {
                if mem::needs_drop::<(K, V)>() {
                    if !self.sentinel_zero_ptr.is_null() {
                        ptr::drop_in_place(self.sentinel_zero_ptr);
                    }
                    if !self.sentinel_one_ptr.is_null() {
                        ptr::drop_in_place(self.sentinel_one_ptr);
                    }
                    while self.cur < self.end {
                        if K::thin_sentinel_zero() != (*self.cur).0 && K::thin_sentinel_one() != (*self.cur).0 {
                            ptr::drop_in_place(self.cur);
                        }
                        self.cur = self.cur.add(1);
                    }
                }
                let layout = Layout::from_size_align(mem::size_of::<(K, V)>() * (self.table_size + 2), mem::align_of::<(K, V)>()).unwrap();
                alloc::dealloc(self.table.offset(-2) as *mut u8, layout);
            }
        }
    }
}

#[inline(always)]
fn overwrite_k<K, V>(ptr: *mut (K, V), k: K) {
    unsafe {
        ptr::write(ptr as *mut K, k);
    }
}

impl<K, V, S> Extend<(K, V)> for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    fn extend<T: IntoIterator<Item=(K, V)>>(&mut self, iter: T) {
        // Keys may be already present or show multiple times in the iterator.
        // Reserve the entire hint lower bound if the map is empty.
        // Otherwise reserve half the hint (rounded up), so the map
        // will only resize twice in the worst case.
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) >> 1
        };
        self.reserve(reserve);
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, S> FromIterator<(K, V)> for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher + Default
{
    fn from_iter<T: IntoIterator<Item=(K, V)>>(iter: T) -> ThinMap<K, V, S> {
        let mut map = ThinMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

impl<'a, K, V, S> Extend<(&'a K, &'a V)> for ThinMap<K, V, S>
    where K: Eq + Hash + Copy + ThinSentinel,
          V: Copy,
          S: BuildHasher
{
    fn extend<T: IntoIterator<Item=(&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K, V, S> Clone for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel + Clone,
          V: PartialEq + Clone,
          S: BuildHasher + Clone
{
    fn clone(&self) -> Self {
        if self.table_size == 0 {
            return ThinMap {
                table_size: 0,
                table: ptr::null_mut(),
                occupied_sentinels: 0,
                occupied: 0,
                sentinels: 0,
                hasher: self.hasher.clone(),
                _marker: marker::PhantomData,
            };
        }
        let mut r = ThinMap::with_capacity_and_hasher(self.len(), self.hasher.clone());
        for (k, v) in self.iter() {
            r.insert((*k).clone(), (*v).clone());
        }
        r
    }
}

impl<K, V, S> PartialEq for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          V: PartialEq,
          S: BuildHasher
{
    fn eq(&self, other: &ThinMap<K, V, S>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> Eq for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          V: Eq,
          S: BuildHasher
{}

impl<K, V, S> Default for ThinMap<K, V, S>
    where K: Eq + Hash + ThinSentinel,
          S: BuildHasher + Default
{
    /// Creates an empty `ThinMap<K, V, S>`, with the `Default` value for the hasher.
    fn default() -> ThinMap<K, V, S> {
        ThinMap::with_hasher(Default::default())
    }
}

#[cfg(test)]
mod test_map {
    extern crate rand;

    use super::ThinMap;
    use super::ThinSentinel;
    use super::OneFieldHasherBuilder;
    use super::Entry::{Occupied, Vacant};
    use std::cell::RefCell;
    use thin_map::test_map::rand::prelude::*;

    #[test]
    fn test_zero_capacities() {
        type HM = ThinMap<i32, i32>;

        let m = HM::new();
        assert_eq!(m.capacity(), 0);

        let m = HM::default();
        assert_eq!(m.capacity(), 0);

        let m = HM::with_hasher(OneFieldHasherBuilder::new());
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity(0);
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity_and_hasher(0, OneFieldHasherBuilder::new());
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.insert(1, 1);
        m.insert(2, 2);
        m.remove(&1);
        m.remove(&2);
        m.shrink_to_fit();
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.reserve(0);
        assert_eq!(m.capacity(), 0);
    }

    #[test]
    fn test_create_capacity_zero() {
        let mut m = ThinMap::with_capacity(0);

        assert!(m.insert(1, 1).is_none());

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = ThinMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&2).unwrap(), 4);
    }

    #[test]
    fn test_clone() {
        let mut m = ThinMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        let m2 = m.clone();
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    thread_local! { static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new()) }

    #[derive(Hash, PartialEq, Eq, Debug)]
    struct Droppable {
        k: usize,
    }

    impl Droppable {
        fn new(k: usize) -> Droppable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Droppable { k: k }
        }
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Droppable {
        fn clone(&self) -> Droppable {
            Droppable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        DROP_VECTOR.with(|slot| {
            *slot.borrow_mut() = vec![0; 200];
        });

        {
            let mut m = ThinMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..200 {
                let d1 = Droppable::new(i);
                m.insert(i, d1);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for i in 0..50 {
                let v = m.remove(&i);

                assert!(v.is_some());

                DROP_VECTOR.with(|v| {
                    assert_eq!(v.borrow()[i], 1);
                });
            }

            DROP_VECTOR.with(|v| {
                for i in 0..50 {
                    assert_eq!(v.borrow()[i], 0);
                }

                for i in 50..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_into_iter_drops() {
        DROP_VECTOR.with(|v| {
            *v.borrow_mut() = vec![0; 200];
        });

        let hm = {
            let mut hm = ThinMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..200 {
                let d1 = Droppable::new(i);
                hm.insert(i, d1);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            hm
        };

        // By the way, ensure that cloning doesn't screw up the dropping.
        drop(hm.clone());

        {
            let mut half = hm.into_iter().take(100);

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for _ in half.by_ref() {}

            DROP_VECTOR.with(|v| {
                let nv = (0..200)
                    .filter(|&i| v.borrow()[i] == 1)
                    .count();

                assert_eq!(nv, 100);
            });
        };

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_remove() {
        let mut m: ThinMap<i32, bool> = ThinMap::new();
        assert_eq!(m.remove(&0), None);
    }

    #[test]
    fn test_empty_entry() {
        let mut m: ThinMap<i32, bool> = ThinMap::new();
        match m.entry(0) {
            Occupied(_) => panic!(),
            Vacant(_) => {}
        }
        assert!(*m.entry(0).or_insert(true));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_empty_iter() {
        let mut m: ThinMap<i32, bool> = ThinMap::new();
        assert_eq!(m.drain().next(), None);
        assert_eq!(m.keys().next(), None);
        assert_eq!(m.values().next(), None);
        assert_eq!(m.values_mut().next(), None);
        assert_eq!(m.iter().next(), None);
        assert_eq!(m.iter_mut().next(), None);
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.into_iter().next(), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = ThinMap::new();

        // Try this a few times to make sure we never screw up the ThinMap's
        // internal state.
        for _ in 0..10 {
            assert!(m.is_empty());

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());

                for j in 1..i + 1 {
                    let r = m.get(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in i + 1..1001 {
                    let r = m.get(&j);
                    assert_eq!(r, None);
                }
            }

            for i in 1001..2001 {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in 1..1001 {
                assert!(m.remove(&i).is_some());

                for j in 1..i + 1 {
                    assert!(!m.contains_key(&j));
                }

                for j in i + 1..1001 {
                    assert!(m.contains_key(&j));
                }
            }

            for i in 1..1001 {
                assert!(!m.contains_key(&i));
            }

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in (1..1001).rev() {
                assert!(m.remove(&i).is_some());

                for j in i..1001 {
                    assert!(!m.contains_key(&j));
                }

                for j in 1..i {
                    assert!(m.contains_key(&j));
                }
            }
        }
    }

    #[test]
    fn test_find_mut() {
        let mut m = ThinMap::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = ThinMap::new();
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(!m.insert(1, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = ThinMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(5, 3).is_none());
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = ThinMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(m.insert(5, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert!(m.remove(&1).is_some());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = ThinMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(!m.is_empty());
        assert!(m.remove(&1).is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut m = ThinMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_remove_entry() {
        let mut m = ThinMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove_entry(&1), Some((1, 2)));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_iterate() {
        let mut m = ThinMap::with_capacity(4);
        for i in 0..32 {
            assert!(m.insert(i, i * 2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;

        for (k, v) in &m {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: ThinMap<_, _> = vec.into_iter().collect();
        let keys: Vec<_> = map.keys().cloned().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: ThinMap<_, _> = vec.into_iter().collect();
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_values_mut() {
        let vec = vec![(1, 1), (2, 2), (3, 3)];
        let mut map: ThinMap<_, _> = vec.into_iter().collect();
        for value in map.values_mut() {
            *value = (*value) * 2
        }
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&2));
        assert!(values.contains(&4));
        assert!(values.contains(&6));
    }

    #[test]
    fn test_find() {
        let mut m = ThinMap::new();
        assert!(m.get(&1).is_none());
        m.insert(1, 2);
        match m.get(&1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2),
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = ThinMap::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = ThinMap::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_show() {
        let mut map = ThinMap::new();
        let empty: ThinMap<i32, i32> = ThinMap::new();

        map.insert(1, 2);
        map.insert(3, 4);

        let map_str = format!("{:?}", map);

        assert!(map_str == "{1: 2, 3: 4}" ||
            map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_expand() {
        let mut m = ThinMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0;
        let old_cap = m.capacity();
        while old_cap == m.capacity() {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_behavior_resize_policy() {
        let mut m = ThinMap::new();

        assert_eq!(m.len(), 0);
        assert_eq!(m.capacity(), 0);
        assert!(m.is_empty());

        m.insert(0, 0);
        m.remove(&0);
        assert!(m.is_empty());
        let initial_cap = m.capacity();
        m.reserve(initial_cap);
        let cap = m.capacity();

        assert_eq!(cap, initial_cap);

        let mut i = 2;
        for _ in 0..cap * 3 / 4 {
            m.insert(i, i);
            i += 1;
        }
        // three quarters full

        assert_eq!(m.len(), i - 2);
        assert_eq!(m.capacity(), cap);

        for _ in 0..cap / 4 + 2 {
            m.insert(i, i);
            i += 1;
        }
        // full + 1

        let new_cap = m.capacity();
        assert_eq!(new_cap, cap * 2);

        for _ in 0..cap / 2 - 1 {
            i -= 1;
            m.remove(&i);
            assert_eq!(m.capacity(), new_cap);
        }
        // A little more than one quarter full.
        m.shrink_to_fit();
        assert_eq!(m.capacity(), cap);
        // again, a little more than half full
        for _ in 0..cap / 2 - 1 {
            i -= 1;
            m.remove(&i);
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), i - 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_reserve_shrink_to_fit() {
        let mut m = ThinMap::new();
        m.insert(0, 0);
        m.remove(&0);
        assert!(m.capacity() >= m.len());
        for i in 0..128 {
            m.insert(i, i);
        }
        m.reserve(256);

        let usable_cap = m.capacity();
        for i in 128..(128 + 256) {
            m.insert(i, i);
            assert_eq!(m.capacity(), usable_cap);
        }

        for i in 100..(128 + 256) {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), 100);
        assert!(!m.is_empty());
        assert!(m.capacity() >= m.len());

        for i in 0..100 {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();
        m.insert(0, 0);

        assert_eq!(m.len(), 1);
        assert!(m.capacity() >= m.len());
        assert_eq!(m.remove(&0), Some(0));
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: ThinMap<_, _> = xs.iter().cloned().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: ThinMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: ThinMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_mut_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: ThinMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_mut_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: ThinMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut map = ThinMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = ThinMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[&4];
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: ThinMap<_, _> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        assert_eq!(map.get(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);


        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        assert_eq!(map.get(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);


        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_entry_take_doesnt_corrupt() {
        #![allow(deprecated)] //rand
        // Test for #19292
        fn check(m: &ThinMap<i32, ()>) {
            for k in m.keys() {
                assert!(m.contains_key(k),
                        "{} is in keys() but not in the map?", k);
            }
        }

        let mut m = ThinMap::new();
        let mut rng = thread_rng();

        // Populate the map with some items.
        for _ in 0..50 {
            let x = rng.gen_range(-10, 10);
            m.insert(x, ());
        }

        for i in 0..1000 {
            let x = rng.gen_range(-10, 10);
            match m.entry(x) {
                Vacant(_) => {}
                Occupied(e) => {
                    println!("{}: remove {}", i, x);
                    e.remove();
                }
            }

            check(&m);
        }
    }

    #[test]
    fn test_extend_ref() {
        let mut a = ThinMap::new();
        a.insert(1, "one");
        let mut b = ThinMap::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[&1], "one");
        assert_eq!(a[&2], "two");
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn test_capacity_not_less_than_len() {
        let mut a = ThinMap::new();
        let mut item = 2;

        for _ in 0..116 {
            a.insert(item, 0);
            item += 1;
        }

        assert!(a.capacity() > a.len());

        let free = a.capacity() - a.len();
        for _ in 0..free {
            a.insert(item, 0);
            item += 1;
        }

        assert_eq!(a.len(), a.capacity());

        // Insert at capacity should cause allocation.
        a.insert(item, 0);
        assert!(a.capacity() > a.len());
    }

    #[test]
    fn test_occupied_entry_key() {
        let mut a = ThinMap::new();
        let key = 17;
        let value = 222;
        assert!(a.is_empty());
        a.insert(key.clone(), value.clone());
        assert_eq!(a.len(), 1);
        assert_eq!(a[&key], value);

        match a.entry(key.clone()) {
            Vacant(_) => panic!(),
            Occupied(e) => assert_eq!(key, *e.key()),
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[&key], value);
    }

    #[test]
    fn test_vacant_entry_key() {
        let mut a = ThinMap::new();
        let key = 17;
        let value = 222;

        assert!(a.is_empty());
        match a.entry(key.clone()) {
            Occupied(_) => panic!(),
            Vacant(e) => {
                assert_eq!(key, *e.key());
                e.insert(value.clone());
            }
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[&key], value);
    }

    #[test]
    fn test_retain() {
        let mut map: ThinMap<i32, i32> = (0..100).map(|x| (x, x * 10)).collect();

        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 50);
        assert_eq!(map[&2], 20);
        assert_eq!(map[&4], 40);
        assert_eq!(map[&6], 60);
    }
}
