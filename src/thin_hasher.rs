// Copyright 2018 Mohammad Rezaei.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

//! Implementations of `Hasher` that work well with `ThinMap`/`ThinSet`
use util::*;

use std::sync::atomic::*;
use std::hash::Hasher;
use std::hash::BuildHasher;

static SEED: AtomicUsize = AtomicUsize::new(0xcafebabe_usize);

fn next_seed() -> u64 {
    let x = SEED.load(Ordering::Acquire) as u64;
    let y = spread_three(x);
    SEED.compare_and_swap(x as usize, y as usize, Ordering::Release); // we don't care if it fails
    y
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneFieldHasherBuilder {
    seed: u64
}

/// The default hasher used by `ThinMap`/`ThinSet`. Resilient and fast.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneFieldHasher {
    hash: u64
}

impl Default for OneFieldHasherBuilder {
    fn default() -> Self {
        OneFieldHasherBuilder::new()
    }
}

impl OneFieldHasherBuilder {
    pub fn new() -> Self {
        OneFieldHasherBuilder { seed: next_seed() }
    }
}

impl BuildHasher for OneFieldHasherBuilder {
    type Hasher = OneFieldHasher;

    #[inline]
    fn build_hasher(&self) -> <Self as BuildHasher>::Hasher {
        OneFieldHasher::new(self.seed)
    }
}

impl OneFieldHasher {
    #[inline]
    pub fn new(seed: u64) -> Self {
        OneFieldHasher { hash: seed }
    }
}

impl Hasher for OneFieldHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        spread_one(self.hash)
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut x: u64 = 0;
        let mut iter = bytes.iter();
        while let Some(byte) = iter.next() {
            x ^= *byte as u64;
            x <<= 8;
        }
        self.hash ^= x;
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        self.hash ^= i as u64;
        self.hash ^= (i >> 64) as u64;
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i8(&mut self, i: i8) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i16(&mut self, i: i16) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i64(&mut self, i: i64) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128)
    }

    #[inline(always)]
    fn write_isize(&mut self, i: isize) {
        self.hash ^= i as u64
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiFieldHasherBuilder {
    seed: u64
}

/// A hasher that should work a little bit better for multi-field custom keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiFieldHasher {
    hash: u64
}

impl Default for MultiFieldHasherBuilder {
    #[inline]
    fn default() -> Self {
        MultiFieldHasherBuilder::new()
    }
}

impl MultiFieldHasherBuilder {
    #[inline]
    pub fn new() -> Self {
        MultiFieldHasherBuilder { seed: next_seed() }
    }
}

impl BuildHasher for MultiFieldHasherBuilder {
    type Hasher = MultiFieldHasher;

    #[inline]
    fn build_hasher(&self) -> <Self as BuildHasher>::Hasher {
        MultiFieldHasher::new(self.seed)
    }
}

impl MultiFieldHasher {
    #[inline]
    pub fn new(seed: u64) -> Self {
        MultiFieldHasher { hash: seed }
    }
}

impl Hasher for MultiFieldHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        spread_one(self.hash)
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut x: u64 = 0;
        let mut iter = bytes.iter();
        while let Some(byte) = iter.next() {
            x ^= *byte as u64;
            x <<= 8;
        }
        self.hash ^= x;
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.hash ^= i as u64;
        self.hash = self.hash.rotate_right(8);
    }

    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.hash ^= i as u64;
        self.hash = self.hash.rotate_right(16);
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.hash ^= i as u64;
        self.hash = self.hash.rotate_right(32);
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        self.hash ^= i as u64;
        self.hash ^= (i >> 64) as u64;
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i8(&mut self, i: i8) {
        self.write_u8(i as u8)
    }

    #[inline(always)]
    fn write_i16(&mut self, i: i16) {
        self.write_u16(i as u16)
    }

    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.write_u32(i as u32)
    }

    #[inline(always)]
    fn write_i64(&mut self, i: i64) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128)
    }

    #[inline(always)]
    fn write_isize(&mut self, i: isize) {
        self.hash ^= i as u64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrivialOneFieldHasherBuilder {
    seed: u64
}

/// A very fast hasher that has low resilience. Still appropriate for `ThinMap`/`ThinSet`
/// because of the non-linear, adaptive collision resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrivialOneFieldHasher {
    hash: u64
}

impl Default for TrivialOneFieldHasherBuilder {
    fn default() -> Self {
        TrivialOneFieldHasherBuilder::new()
    }
}

impl TrivialOneFieldHasherBuilder {
    pub fn new() -> Self {
        TrivialOneFieldHasherBuilder { seed: next_seed() }
    }
}

impl BuildHasher for TrivialOneFieldHasherBuilder {
    type Hasher = TrivialOneFieldHasher;

    #[inline]
    fn build_hasher(&self) -> <Self as BuildHasher>::Hasher {
        TrivialOneFieldHasher::new(self.seed)
    }
}

impl TrivialOneFieldHasher {
    #[inline]
    pub fn new(seed: u64) -> Self {
        TrivialOneFieldHasher { hash: seed }
    }
}

impl Hasher for TrivialOneFieldHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut x: u64 = 0;
        let mut iter = bytes.iter();
        while let Some(byte) = iter.next() {
            x ^= *byte as u64;
            x <<= 8;
        }
        self.hash ^= x;
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        self.hash ^= i as u64;
        self.hash ^= (i >> 64) as u64;
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i8(&mut self, i: i8) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i16(&mut self, i: i16) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i64(&mut self, i: i64) {
        self.hash ^= i as u64
    }

    #[inline(always)]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128)
    }

    #[inline(always)]
    fn write_isize(&mut self, i: isize) {
        self.hash ^= i as u64
    }
}
