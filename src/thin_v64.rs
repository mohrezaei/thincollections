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

//! # `V64` a general `Vec` replacement in a single 64 bit pointer.
//! Ideal for elements that are 32 bits or less and the expected vector size
//! is small (`total number of elements x element size < 7 bytes`)
use std::{
    alloc::{self, Layout},
    mem, ptr, marker,
};
use std::fmt::{self};
use std::cmp;
use std::ops::Index;
use std::iter::FromIterator;
use std::ops::Deref;
use std::slice;
use std::slice::SliceIndex;
use std::ptr::NonNull;
use std::ops::DerefMut;
use std::ops::IndexMut;
use std::borrow::Cow;
use std::iter::FusedIterator;
use std::ops::RangeBounds;
use std::ops::Bound::*;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::num::NonZeroU64;

/// A thin (64bit) vector. Guaranteed to be a 64 bit smart pointer.
///
/// Rust's `std::collections::Vec` (`std::Vec` for short) is a triple-fat (3 x usize) pointer to the heap.
/// When `std::Vec` is on the stack, it works well. However, when `std::Vec` is used inside other data
/// structures, such as `Vec<Vec<_>>`, the triple fatness starts to become a problem. For example,
/// when a `Vec<Vec<_>>` has to resize, it needs to move 3 times as much memory.
///
/// `V64` uses a single 64bit value as a smart pointer, making it attractive for building larger
/// data structures. Additionally, `V64` can store a small number of values on the stack if they
/// fit in 7 bytes. A single `i32`, three `u16`, or seven `u8` are examples of data that can be stored
/// with no heap allocation.
///
/// `V64` is also null optimized, which makes an `Option<V64<_>>` also 64 bits.
///
/// # Capacity and reallocation
///
/// The capacity of a vector is the amount of space allocated for any future
/// elements that will be added onto the vector. This is not to be confused with
/// the *length* of a vector, which specifies the number of actual elements
/// within the vector. If a vector's length exceeds its capacity, its capacity
/// will automatically be increased, but its elements will have to be
/// reallocated.
///
/// For example, a vector with capacity 10 and length 0 would be an empty vector
/// with space for 10 more elements. Pushing 10 or fewer elements onto the
/// vector will not change its capacity or cause reallocation to occur. However,
/// if the vector's length is increased to 11, it will have to reallocate, which
/// can be slow. For this reason, it is recommended to use [`V64::with_capacity`]
/// whenever possible to specify how big the vector is expected to get.
///
/// `V64` also uses its smart pointer for storage of small values. Data that can be
/// stored within seven bytes and the required alignment will not allocate any heap memory.
///
#[cfg(target_endian = "little")]
pub struct V64<T> {
    u: NonZeroU64,
    _marker: marker::PhantomData<T>,
}

const ZST_MASK: u64 = 0x8000_0000_0000_0000u64;

pub enum Control {
    Heap(*mut u8),
    Stack(usize),
}

impl<T> V64<T> {
    /// Constructs a new, empty `V64<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec: V64<i32> = V64::new();
    /// vec.push(17); // no heap allocation yet, as we can store "17" on the stack
    /// vec.push(42); // there is no more room for another i32, so now heap is allocated and both 17 and 42 are moved there.
    /// ```
    #[inline]
    pub fn new() -> V64<T> {
        unsafe {
            if mem::size_of::<T>() == 0 { return V64 { u: NonZeroU64::new_unchecked(ZST_MASK), _marker: marker::PhantomData }; }
            V64 { u: NonZeroU64::new_unchecked(8), _marker: marker::PhantomData }
        }
    }

    /// Constructs a new, empty `V64<T>` with the specified capacity.
    ///
    /// The vector will be able to hold exactly `capacity` elements without
    /// reallocating. If `capacity` is 0, the vector will not allocate.
    ///
    /// It is important to note that although the returned vector has the
    /// *capacity* specified, the vector will have a zero *length*. For an
    /// explanation of the difference between length and capacity, see
    /// *[Capacity and reallocation]*.
    ///
    /// [Capacity and reallocation]: #capacity-and-reallocation
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec = V64::with_capacity(10);
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     vec.push(i);
    /// }
    ///
    /// // ...but this may make the vector reallocate
    /// vec.push(11);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> V64<T> {
        if mem::size_of::<T>() == 0 { return V64::new(); }
        if capacity == 0 || mem::size_of::<T>() * capacity < 8 {
            return <V64<T>>::new();
        }
        let array = <V64<T>>::allocate_array(capacity);
        unsafe {
            V64 { u: NonZeroU64::new_unchecked(array as u64), _marker: marker::PhantomData }
        }
    }

    /// Returns the number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let vec: V64<i32> = V64::with_capacity(10);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 { return <usize>::max_value(); }
        match self.control() {
            Control::Heap(ptr) => {
                unsafe {
                    let len_ptr = ptr as *mut usize;
                    return *len_ptr.add(1);
                }
            }
            Control::Stack(_) => {
                let s = mem::size_of::<T>();
                return if s == 0 { 7 } else { 7 / s };
            }
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `V64<T>`. The collection may reserve more space to avoid
    /// frequent reallocations. After calling `reserve`, capacity will be
    /// greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec = V64::new();
    /// vec.push(1);
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        if mem::size_of::<T>() == 0 { return; }
        let len = self.len();
        self.reserve_exact(cmp::max(len * 2, len + additional));
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to
    /// be inserted in the given `V64<T>`. After calling `reserve_exact`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that when the stack storage capacity exceeds `self.len() + additional`,
    /// nothing is allocated on the heap and the capacity remains as the stack capacity.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec = V64::new();
    /// vec.push(1);
    /// vec.reserve_exact(10);
    /// assert_eq!(11, vec.capacity());
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        if mem::size_of::<T>() == 0 { return; }
        let remain = self.capacity() - self.len();
        if remain < additional {
            match self.control() {
                Control::Heap(ptr) => {
                    unsafe {
                        let len_ptr = ptr as *mut usize;
                        self.realloc_heap(len_ptr, *len_ptr + additional);
                    }
                }
                Control::Stack(len) => {
                    self.move_to_heap(len, len + additional);
                }
            }
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    ///
    /// It will drop down as close as possible to the length but the allocator
    /// may still inform the vector that there is space for a few more elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec = V64::with_capacity(10);
    /// vec.push(1);
    /// vec.push(2);
    /// vec.push(3);
    /// assert_eq!(vec.capacity(), 10);
    /// vec.shrink_to_fit();
    /// assert!(vec.capacity() < 10);
    /// assert!(vec.capacity() == 3);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        if mem::size_of::<T>() == 0 { return; }
        match self.control() {
            Control::Heap(ptr) => {
                //todo: implement shrinking back to stack mode.
                unsafe {
                    let len_ptr = ptr as *mut usize;
                    self.realloc_heap(len_ptr, *len_ptr);
                }
            }
            Control::Stack(_) => {}
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// The [`drain`] method can emulate `truncate`, but causes the excess
    /// elements to be returned instead of dropped.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// Truncating a five element vector to two elements:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3, 4, 5];
    /// vec.truncate(2);
    /// assert_eq!(2, vec.len());
    /// assert_eq!(1, vec[0]);
    /// assert_eq!(2, vec[1]);
    /// # }
    /// ```
    ///
    /// Truncation works even if everything is on the stack:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 3, 4, 5]; // can have up to 7 u8's on stack
    /// assert_eq!(0, vec.bytes_on_heap());
    /// assert_eq!(5, vec.len());
    /// assert_eq!([1,2,3,4,5], vec[..]);
    /// assert_eq!(1, vec[0]);
    /// assert_eq!(2, vec[1]);
    /// vec.truncate(2);
    /// assert_eq!(2, vec.len());
    /// assert_eq!(1, vec[0]);
    /// assert_eq!(2, vec[1]);
    /// # }
    /// ```
    ///
    /// No truncation occurs when `len` is greater than the vector's current
    /// length:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3];
    /// vec.truncate(8);
    /// assert_eq!(vec[..], [1, 2, 3]);
    /// # }
    /// ```
    ///
    /// Truncating when `len == 0` is equivalent to calling the [`clear`]
    /// method.
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3];
    /// vec.truncate(0);
    /// assert_eq!(vec[..], []);
    /// # }
    /// ```
    ///
    /// [`clear`]: #method.clear
    /// [`drain`]: #method.drain
    pub fn truncate(&mut self, len: usize) {
        if mem::size_of::<T>() == 0 {
            unsafe {
                self.set_len(len);
                return;
            }
        }
        match self.control() {
            Control::Heap(ptr) => {
                unsafe {
                    let len_ptr = ptr as *mut usize;
                    if *len_ptr > len {
                        if mem::needs_drop::<T>() {
                            let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                            let mut cur = ((len_ptr as *mut u8).add(header_bytes) as *mut T).add(len);
                            let end = cur.add(*len_ptr - len);
                            while cur < end {
                                *len_ptr -= 1;
                                ptr::drop_in_place(cur);
                                cur = cur.add(1);
                            }
                        } else {
                            *len_ptr = len;
                        }
                    }
                }
            }
            Control::Stack(stack_len) => {
                unsafe {
                    if stack_len > len {
                        let mut cur = self.stack_ptr().add(len);
                        let end = cur.add(stack_len - len);
                        let mut cur_len = stack_len;
                        while cur < end {
                            cur_len -= 1;
                            self.set_stack_len(cur_len);
                            ptr::drop_in_place(cur);
                            ptr::write(cur, mem::zeroed::<T>()); // we have to do this even if T doesn't require drop
                            cur = cur.add(1);
                        }
                    }
                }
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// use std::io::{self, Write};
    /// let buffer = v64![1, 2, 3, 5, 8];
    /// io::sink().write(buffer.as_slice()).unwrap();
    /// # }
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Returns the number of bytes this vector has allocated on the heap
    ///
    /// # Examples
    ///
    /// We have room for 1 x32 on the stack:
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec: V64<u32> = V64::new();
    /// assert_eq!(0, vec.bytes_on_heap());
    /// vec.push(0x12345678);
    /// assert_eq!(0, vec.bytes_on_heap());
    /// assert_eq!(1, vec.len());
    /// assert_eq!(0x12345678, vec[0]);
    /// vec.push(0x9ABCDEFF);
    /// assert!(vec.bytes_on_heap() > 0);
    /// assert_eq!(2, vec.len());
    /// assert_eq!(0x12345678, vec[0]);
    /// assert_eq!(0x9ABCDEFF, vec[1]);
    /// ```
    /// We have room for 7 x8 on the stack:
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut vec: V64<u8> = V64::new();
    /// assert_eq!(0, vec.bytes_on_heap());
    /// vec.push(0x12);
    /// vec.push(0x34);
    /// vec.push(0x56);
    /// vec.push(0x78);
    /// vec.push(0x9A);
    /// vec.push(0xBC);
    /// vec.push(0xDE);
    /// assert_eq!(0, vec.bytes_on_heap());
    /// assert_eq!(7, vec.len());
    /// assert_eq!(vec[..], [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE]);
    /// vec.push(0x42);
    /// assert!(vec.bytes_on_heap() > 0);
    /// assert_eq!(8, vec.len());
    /// assert_eq!(vec[..], [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0x42]);
    /// ```
    pub fn bytes_on_heap(&self) -> usize {
        if mem::size_of::<T>() == 0 { return 0; }
        match self.control() {
            Control::Heap(ptr) => {
                unsafe {
                    let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                    let cap_ptr: *mut usize = (ptr as *mut usize).add(1);
                    return header_bytes + *cap_ptr * mem::size_of::<T>();
                }
            }
            Control::Stack(_len) => { return 0; }
        }
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// use std::io::{self, Read};
    /// let mut buffer = v64![0; 3];
    /// io::repeat(0b101).read_exact(buffer.as_mut_slice()).unwrap();
    /// # }
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v = v64!["foo", "bar", "baz", "qux"];
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(v[..], ["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v[..], ["baz", "qux"]);
    /// # }
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v: V64<u8> = v64![0x12, 0x34, 0x56, 0x78];
    ///
    /// assert_eq!(v.swap_remove(1), 0x34);
    /// assert_eq!(v[..], [0x12, 0x78, 0x56]);
    ///
    /// assert_eq!(v.swap_remove(0), 0x12);
    /// assert_eq!(v[..], [0x56, 0x78]);
    /// # }
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len() - 1;
                self.set_len(len);
                return ptr::read(1 as *const T);
            }
        }
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check on hole succeeds there must be a last element (which
            // can be self[index] itself).
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    if index < *len_ptr {
                        *len_ptr -= 1;
                        let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                        <V64<T>>::replace(ptr.add(header_bytes) as *mut T, *len_ptr, index)
                    } else {
                        panic!("index out of bounds! len: {}, index {}", *len_ptr, index);
                    }
                }
                Control::Stack(len) => {
                    if index < len {
                        self.set_stack_len(len - 1);
                        <V64<T>>::replace(self.stack_ptr(),
                                          len - 1, index)
                    } else {
                        panic!("index out of bounds! len: {}, index {}", len, index);
                    }
                }
            }
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, [1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, [1, 4, 2, 3, 5]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 3];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, [1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, [1, 4, 2, 3, 5]);
    /// # }
    /// ```
    pub fn insert(&mut self, index: usize, val: T) {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len();
                assert!(index <= len);
                self.set_len(len + 1);
                return;
            }
        }
        match self.control() {
            Control::Heap(ptr) => {
                let len_ptr = ptr as *mut usize;
                unsafe { assert!(index <= *len_ptr); }
                self.possibly_grow_heap(ptr);
            }
            Control::Stack(len) => {
                assert!(index <= len);
                if mem::size_of::<T>() * (len + 1) < 8 {
                    self.stack_insert(index, val, len);
                    return;
                } else {
                    self.move_to_heap(len, 1);
                }
            }
        }
        self.heap_insert(index, val);
    }

    fn heap_insert(&mut self, index: usize, val: T) {
        unsafe {
            let len_ptr = self.u.get() as usize as *mut usize;
            let len = *len_ptr;
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().offset(index as isize);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy(p, p.offset(1), len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, val);
            }
            *len_ptr += 1;
        }
    }

    fn stack_insert(&mut self, index: usize, val: T, len: usize) {
        unsafe {
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().offset(index as isize);
                // Shift everything over to make space. (Duplicating the
                // `index`th element into two consecutive places.)
                ptr::copy(p, p.offset(1), len - index);
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, val);
            }
            self.set_stack_len(len + 1);
        }
    }

    #[inline]
    fn replace(ptr: *mut T, src: usize, dst: usize) -> T {
        unsafe {
            let hole: *mut T = ptr.add(dst);
            let last = ptr::read(ptr.add(src));
            ptr::replace(hole, last)
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v = v64![1, 2, 3];
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v, [1, 3]);
    /// # }
    /// ```
    ///
    /// Works on stack as well:
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v: V64<u8> = v64![1, 2, 3];
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v, [1, 3]);
    /// # }
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len();
                assert!(index < len);
                self.set_len(len - 1);
                return ptr::read(1 as *const T);
            }
        }
        let array: *mut T;
        let len: usize;
        {
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    unsafe {
                        len = *len_ptr;
                        *len_ptr -= 1;
                        let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                        array = ptr.add(header_bytes) as *mut T;
                    }
                }
                Control::Stack(stack_len) => {
                    len = stack_len;
                    array = self.stack_ptr();
                    self.set_stack_len(stack_len - 1);
                }
            }
        }
        assert!(index < len);
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = array.offset(index as isize);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.offset(1), ptr, len - index - 1);
            }
            ret
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3, 4];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, [2, 4]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 3, 4, 5, 6];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, [2, 4, 6]);
    /// # }
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        if mem::size_of::<T>() == 0 {
            let mut count = 0;
            unsafe {
                let t: T = ptr::read(1 as *const T);
                for _i in 0..self.len() {
                    if f(&t) { count += 1; }
                }
                self.set_len(count);
            }
            return;
        }
        let mut array: *mut T;
        let len: usize;
        let heap: bool;
        {
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    unsafe {
                        len = *len_ptr;
                        let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                        array = ptr.add(header_bytes) as *mut T;
                        heap = true;
                    }
                }
                Control::Stack(stack_len) => {
                    len = stack_len;
                    heap = false;
                    array = self.stack_ptr();
                }
            }
        }
        unsafe {
            let end = array.add(len);
            let mut cur = array;
            let mut removed: usize = len;
            while array < end {
                if f(&*array) {
                    if array != cur {
                        ptr::copy_nonoverlapping(array, cur, 1);
                    }
                    cur = cur.add(1);
                    removed -= 1;
                } else {
                    ptr::drop_in_place(array);
                }
                array = array.add(1);
            }
            if heap {
                let len_ptr = self.u.get() as usize as *mut usize;
                *len_ptr -= removed;
            } else {
                self.set_stack_len(len - removed);
            }
        }
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality
    /// relation.
    ///
    /// The `same_bucket` function is passed references to two elements from the vector, and
    /// returns `true` if the elements compare equal, or `false` if they do not. The elements are
    /// passed in opposite order from their order in the vector, so if `same_bucket(a, b)` returns
    /// `true`, `a` is removed.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64!["foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
    /// # }
    /// ```
    ///
    /// Also works on stack:
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 2, 3, 2];
    ///
    /// vec.dedup_by(|a, b| a == b);
    ///
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// # }
    /// ```
    pub fn dedup_by<F>(&mut self, mut same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
        if mem::size_of::<T>() == 0 {
            unsafe {
                if self.len() > 1 {
                    self.set_len(1);
                }
            }
            return;
        }
        unsafe {
            // Although we have a mutable reference to `self`, we cannot make
            // *arbitrary* changes. The `same_bucket` calls could panic, so we
            // must ensure that the vector is in a valid state at all time.
            //
            // The way that we handle this is by using swaps; we iterate
            // over all the elements, swapping as we go so that at the end
            // the elements we wish to keep are in the front, and those we
            // wish to reject are at the back. We can then truncate the
            // vector. This operation is still O(n).
            //
            // Example: We start in this state, where `r` represents "next
            // read" and `w` represents "next_write`.
            //
            //           r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //           w
            //
            // Comparing self[r] against self[w-1], this is not a duplicate, so
            // we swap self[r] and self[w] (no effect as r==w) and then increment both
            // r and w, leaving us with:
            //
            //               r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //               w
            //
            // Comparing self[r] against self[w-1], this value is a duplicate,
            // so we increment `r` but leave everything else unchanged:
            //
            //                   r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 1 | 2 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //               w
            //
            // Comparing self[r] against self[w-1], this is not a duplicate,
            // so swap self[r] and self[w] and advance r and w:
            //
            //                       r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 2 | 1 | 3 | 3 |
            //     +---+---+---+---+---+---+
            //                   w
            //
            // Not a duplicate, repeat:
            //
            //                           r
            //     +---+---+---+---+---+---+
            //     | 0 | 1 | 2 | 3 | 1 | 3 |
            //     +---+---+---+---+---+---+
            //                       w
            //
            // Duplicate, advance r. End of vec. Truncate to w.

            let ln = self.len();
            if ln <= 1 {
                return;
            }

            // Avoid bounds checks by using raw pointers.
            let p = self.as_mut_ptr();
            let mut r: usize = 1;
            let mut w: usize = 1;

            while r < ln {
                let p_r = p.offset(r as isize);
                let p_wm1 = p.offset((w - 1) as isize);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.offset(1);
                        mem::swap(&mut *p_r, &mut *p_w);
                    }
                    w += 1;
                }
                r += 1;
            }

            self.truncate(w);
        }
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same
    /// key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// # }
    /// ```
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F) where F: FnMut(&mut T) -> K, K: PartialEq {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let a = v64![1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        if mem::size_of::<T>() == 0 { return (self.u.get() & (ZST_MASK - 1)) as usize; }
        match self.control() {
            Control::Heap(ptr) => { unsafe { return *(ptr as *mut usize); } }
            Control::Stack(len) => { return len; }
        }
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_v64::V64;
    /// let mut v = V64::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn control(&self) -> Control {
        let c = (self.u.get() & 15) as u8;
        if c == 0 {
            return Control::Heap(self.u.get() as usize as *mut u8);
        }
        return Control::Stack((c & 7) as usize);
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// # }
    /// ```
    #[inline]
    pub fn push(&mut self, val: T) {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len();
                self.set_len(len + 1);
                return;
            }
        }
        match self.control() {
            Control::Heap(ptr) => {
                self.possibly_grow_heap(ptr);
            }
            Control::Stack(len) => {
                if mem::size_of::<T>() * (len + 1) < 8 {
                    self.stack_push(val, len);
                    return;
                } else {
                    self.move_to_heap(len, 1);
                }
            }
        }
        self.heap_push(val);
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// # }
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if mem::size_of::<T>() == 0 {
            let len = self.len();
            if len > 0 {
                unsafe {
                    self.set_len(len - 1);
                    return Some(ptr::read(1 as *const T));
                }
            }
            return None;
        }
        let array: *mut T;
        let len: usize;
        {
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    unsafe {
                        len = *len_ptr;
                        if len == 0 { return None; }
                        *len_ptr -= 1;
                        let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                        array = ptr.add(header_bytes) as *mut T;
                    }
                }
                Control::Stack(stack_len) => {
                    if stack_len == 0 { return None; }
                    len = stack_len;
                    array = self.stack_ptr();
                    self.set_stack_len(stack_len - 1);
                }
            }
        }
        unsafe {
            let ret;
            {
                let ptr = array.offset((len - 1) as isize);
                ret = ptr::read(ptr);
            }
            Some(ret)
        }
    }

    /// Moves all the elements of `other` into `Self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3];
    /// let mut vec2 = v64![4, 5, 6];
    /// vec.append(&mut vec2);
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 3];
    /// let mut vec2: V64<u8> = v64![4, 5, 6];
    /// vec.append(&mut vec2);
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// # }
    /// ```
    ///
    /// And between stack/heap:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1, 2, 3];
    /// let mut vec2: V64<u8> = v64![4, 5, 6, 7, 8];
    /// vec.append(&mut vec2);
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7, 8]);
    /// assert_eq!(vec2, []);
    /// # }
    /// ```
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len() + other.len();
                self.set_len(len);
                other.u = NonZeroU64::new_unchecked(ZST_MASK);
                return;
            }
        }
        unsafe {
            self.append_elements(other.as_slice() as _);
            match other.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    *len_ptr = 0;
                }
                Control::Stack(_stack_len) => {
                    other.u = NonZeroU64::new_unchecked(8);
                }
            }
        }
    }

    #[inline]
    unsafe fn append_elements(&mut self, other: *const [T]) {
        let count = (*other).len();
        self.reserve(count);
        let len = self.len();
        ptr::copy_nonoverlapping(other as *const T, self.get_unchecked_mut(len), count);
        self.set_len(len + count);
    }

    unsafe fn set_len(&mut self, len: usize) {
        if mem::size_of::<T>() == 0 {
            self.u = NonZeroU64::new_unchecked(len as u64 | ZST_MASK);
            return;
        }
        match self.control() {
            Control::Heap(ptr) => {
                let len_ptr = ptr as *mut usize;
                *len_ptr = len;
            }
            Control::Stack(_stack_len) => {
                self.set_stack_len(len);
            }
        }
    }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated `Self`. `self` contains elements `[0, at)`,
    /// and the returned `Self` contains elements `[at, len)`.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1,2,3];
    /// let vec2 = vec.split_off(1);
    /// assert_eq!(vec, [1]);
    /// assert_eq!(vec2, [2, 3]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec: V64<u8> = v64![1,2,3];
    /// let vec2 = vec.split_off(1);
    /// assert_eq!(vec, [1]);
    /// assert_eq!(vec2, [2, 3]);
    /// # }
    /// ```
    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        let len = self.len();
        assert!(at <= len, "`at` out of bounds");

        let other_len = len - at;
        let mut other = V64::with_capacity(other_len);

        // Unsafely `set_len` and copy items to `other`.
        unsafe {
            self.set_len(at);
            other.set_len(other_len);

            ptr::copy_nonoverlapping(self.as_ptr().offset(at as isize),
                                     other.as_mut_ptr(),
                                     other.len());
        }
        other
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v = v64![1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// # }
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0)
    }

    /// Converts the vector into [`Box<[T]>`][owned slice].
    ///
    /// This function causes a bunch of copying and should generally
    /// be avoided. V64<T> is far more versatile than Box<[T]>
    ///
    /// Note that this will drop any excess capacity.
    ///
    /// [owned slice]: ../../std/boxed/struct.Box.html
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let v = v64![1, 2, 3];
    ///
    /// let slice = v.into_boxed_slice();
    /// assert_eq!([1,2,3], slice[..]);
    /// # }
    /// ```
    ///
    /// Any excess capacity is removed:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # use thincollections::thin_v64::IntoV64;
    /// # fn main() {
    /// let mut vec = V64::with_capacity(10);
    /// vec.extend([1, 2, 3].iter().cloned());
    ///
    /// assert_eq!(vec.capacity(), 10);
    /// let slice: Box<[i32]> = vec.into_boxed_slice();
    /// assert_eq!(slice.into_v64().capacity(), 3);
    /// # }
    /// ```
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let slice = slice::from_raw_parts_mut(1 as *mut T, self.len());
                let output: Box<[T]> = Box::from_raw(slice);
                return output;
            }
        }
        unsafe {
            let array: *mut T;
            let len: usize;
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                    array = ptr.add(header_bytes) as *mut T;
                    len = *len_ptr;
                    *len_ptr = 0; // prevents dropping of contents
                }
                Control::Stack(stack_len) => {
                    array = self.stack_ptr();
                    len = stack_len;
                    self.set_stack_len(0); //prevents dropping of contents
                }
            }
            let layout = Layout::from_size_align(mem::size_of::<T>() * len, mem::align_of::<T>()).unwrap();
            let buffer = alloc::alloc(layout) as *mut T;
            ptr::copy_nonoverlapping(array, buffer, len);
            let slice = slice::from_raw_parts_mut(buffer, len);
            let output: Box<[T]> = Box::from_raw(slice);
            output
        }
    }

    /// Creates a draining iterator that removes the specified range in the vector
    /// and yields the removed items.
    ///
    /// Note 1: The element range is removed even if the iterator is only
    /// partially consumed or not consumed at all.
    ///
    /// Note 2: It is unspecified how many elements are removed from the vector
    /// if the `Drain` value is leaked.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v = v64![1, 2, 3];
    /// let u: V64<_> = v.drain(1..).collect();
    /// assert_eq!(v, &[1]);
    /// assert_eq!(u, &[2, 3]);
    ///
    /// // A full range clears the vector
    /// v.drain(..);
    /// assert_eq!(v, &[]);
    /// # }
    /// ```
    ///
    /// Works on stack too:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v: V64<u8> = v64![1, 2, 3];
    /// let u: V64<_> = v.drain(1..).collect();
    /// assert_eq!(v, &[1]);
    /// assert_eq!(u, &[2, 3]);
    /// assert_eq!(0, v.bytes_on_heap());
    ///
    /// // A full range clears the vector
    /// v.drain(..);
    /// assert_eq!(v, &[]);
    /// # }
    /// ```
    pub fn drain<R>(&mut self, range: R) -> Drain<T>
        where R: RangeBounds<usize>
    {
        // Memory safety
        //
        // When the Drain is first created, it shortens the length of
        // the source vector to make sure no uninitialized or moved-from elements
        // are accessible at all if the Drain's destructor never gets to run.
        //
        // Drain will ptr::read out the values to remove.
        // When finished, remaining tail of the vec is copied back to cover
        // the hole, and the vector length is restored to the new length.
        //
        let len = self.len();
        let start = match range.start_bound() {
            Included(&n) => n,
            Excluded(&n) => n + 1,
            Unbounded => 0,
        };
        let end = match range.end_bound() {
            Included(&n) => n + 1,
            Excluded(&n) => n,
            Unbounded => len,
        };
        assert!(start <= end);
        assert!(end <= len);

        unsafe {
            // set self.vec length's to start, to be safe in case Drain is leaked
            self.set_len(start);
            // Use the borrow in the IterMut to indicate borrowing behavior of the
            // whole Drain iterator (like &mut T).
            let range_slice = slice::from_raw_parts_mut(self.as_mut_ptr().offset(start as isize),
                                                        end - start);
            Drain {
                tail_start: end,
                tail_len: len - end,
                iter: range_slice.iter(),
                vec: NonNull::from(self),
            }
        }
    }

    /// Creates a splicing iterator that replaces the specified range in the vector
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// Note 1: The element range is removed even if the iterator is not
    /// consumed until the end.
    ///
    /// Note 2: It is unspecified how many elements are removed from the vector,
    /// if the `Splice` value is leaked.
    ///
    /// Note 3: The input iterator `replace_with` is only consumed
    /// when the `Splice` value is dropped.
    ///
    /// Note 4: This is optimal if:
    ///
    /// * The tail (elements in the vector after `range`) is empty,
    /// * or `replace_with` yields fewer elements than `range`’s length
    /// * or the lower bound of its `size_hint()` is exact.
    ///
    /// Otherwise, a temporary vector is allocated and the tail is moved twice.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v = v64![1, 2, 3];
    /// let new = [7, 8];
    /// let u: V64<_> = v.splice(..2, new.iter().cloned()).collect();
    /// assert_eq!(v, &[7, 8, 3]);
    /// assert_eq!(u, &[1, 2]);
    /// # }
    /// ```
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut v: V64<u8> = v64![1, 2, 3];
    /// let new = [7, 8];
    /// let u: V64<_> = v.splice(..2, new.iter().cloned()).collect();
    /// assert_eq!(v, &[7, 8, 3]);
    /// assert_eq!(u, &[1, 2]);
    /// # }
    /// ```
    #[inline]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<I::IntoIter>
        where R: RangeBounds<usize>, I: IntoIterator<Item=T>
    {
        Splice {
            drain: self.drain(range),
            replace_with: replace_with.into_iter(),
        }
    }

    /// Creates an iterator which uses a closure to determine if an element should be removed.
    ///
    /// If the closure returns true, then the element is removed and yielded.
    /// If the closure returns false, the element will remain in the vector and will not be yielded
    /// by the iterator.
    ///
    /// Using this method is equivalent to the following code:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// # let some_predicate = |x: &mut i32| { *x == 2 || *x == 3 || *x == 6 };
    /// # let mut vec = v64![1, 2, 3, 4, 5, 6];
    /// let mut i = 0;
    /// while i != vec.len() {
    ///     if some_predicate(&mut vec[i]) {
    ///         let val = vec.remove(i);
    ///         // your code here
    ///     } else {
    ///         i += 1;
    ///     }
    /// }
    ///
    /// # assert_eq!(vec, v64![1, 4, 5]);
    /// # }
    /// ```
    ///
    /// But `drain_filter` is easier to use. `drain_filter` is also more efficient,
    /// because it can backshift the elements of the array in bulk.
    ///
    /// Note that `drain_filter` also lets you mutate every element in the filter closure,
    /// regardless of whether you choose to keep or remove it.
    ///
    ///
    /// # Examples
    ///
    /// Splitting an array into evens and odds, reusing the original allocation:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut numbers = v64![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    ///
    /// let evens = numbers.drain_filter(|x| *x % 2 == 0).collect::<V64<_>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, v64![2, 4, 6, 8, 14]);
    /// assert_eq!(odds, v64![1, 3, 5, 9, 11, 13, 15]);
    /// # }
    /// ```
    pub fn drain_filter<F>(&mut self, filter: F) -> DrainFilter<T, F>
        where F: FnMut(&mut T) -> bool,
    {
        let old_len = self.len();

        // Guard against us getting leaked (leak amplification)
        unsafe { self.set_len(0); }

        DrainFilter {
            vec: self,
            idx: 0,
            del: 0,
            old_len,
            pred: filter,
        }
    }

    fn extend_desugared<I: Iterator<Item=T>>(&mut self, mut iterator: I) {
        // This is the case for a general iterator.
        //
        // This function should be the moral equivalent of:
        //
        //      for item in iterator {
        //          self.push(item);
        //      }
        if mem::size_of::<T>() == 0 {
            let mut count = 0;
            while let Some(_element) = iterator.next() { count += 1; }
            unsafe {
                let len = self.len();
                self.set_len(len + count);
            }
            return;
        }
        while let Some(element) = iterator.next() {
            let len = self.len();
            if len == self.capacity() {
                let (lower, _) = iterator.size_hint();
                self.reserve(lower.saturating_add(1));
            }
            unsafe {
                ptr::write(self.get_unchecked_mut(len), element);
                // NB can't overflow since we would have had to alloc the address space
                self.set_len(len + 1);
            }
        }
    }

    #[inline(always)]
    fn stack_ptr(&mut self) -> *mut T {
        unsafe {
            ((&mut self.u) as *mut _ as *mut u8).add(mem::align_of::<T>()) as *mut T
        }
    }
    #[inline(always)]
    fn inc_stack_len(&mut self, old: usize) {
        self.set_stack_len(old + 1);
    }

    #[inline(always)]
    fn set_stack_len(&mut self, len: usize) {
        assert!(len < 8);
        let new_len = len as u64;
        unsafe {
            self.u = NonZeroU64::new_unchecked((self.u.get() & 0xFFFF_FFFF_FFFF_FFF8) | new_len);
        }
    }

    #[cold]
    fn allocate_array(capacity: usize) -> *mut u8 {
        unsafe {
            // align_of is a power of 2. 2 * size_of::<usize> is a power of 2.
            // if align_of is smaller than 2*size_of::<usize>, we'll have no padding between the header and array
            // if align_of is bigger than 2*size_of::<usize>, we'll use the first align for the header
            let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
            let align = cmp::max(16, mem::align_of::<T>());
            let layout = Layout::from_size_align(mem::size_of::<T>() * capacity + header_bytes, align).unwrap();
            let buffer = alloc::alloc(layout);
            assert!(buffer as usize & 15 == 0); // check the allocator respects our assumptions
            ptr::write(buffer as *mut usize, 0); // current length
            ptr::write((buffer as *mut usize).add(1), capacity);
            buffer
        }
    }

    #[inline]
    fn possibly_grow_heap(&mut self, arr: *mut u8) {
        unsafe {
            let len_ptr = arr as *mut usize;
            let cap_ptr = len_ptr.add(1);
            if *len_ptr == *cap_ptr {
                self.realloc_heap(len_ptr, *len_ptr * 2);
            }
        }
    }

    #[inline(always)]
    fn realloc_heap(&mut self, len_ptr: *mut usize, new_capacity: usize) {
        unsafe {
            //realloc seems to bench slower...
            let head_ptr = <V64<T>>::allocate_array(new_capacity);
            let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
            let to_move = cmp::min(new_capacity, *(len_ptr.add(1)));
            ptr::copy_nonoverlapping(len_ptr as *mut u8, head_ptr, to_move * mem::size_of::<T>() + header_bytes);
            let new_cap_ptr = (head_ptr as *mut usize).add(1);
            *new_cap_ptr = new_capacity;
            self.u = NonZeroU64::new_unchecked(head_ptr as u64);
            let old = len_ptr as *mut u8;
            let align = cmp::max(16, mem::align_of::<T>());
            let layout = Layout::from_size_align(mem::size_of::<T>() * (*(len_ptr.add(1))) + header_bytes, align).unwrap();
            alloc::dealloc(old, layout);
        }
    }

    fn move_to_heap(&mut self, len: usize, heap_capacity_min: usize) {
        unsafe {
            let stack_capacity = 7 / mem::size_of::<T>();
            let heap_capacity = cmp::max(stack_capacity * 2, heap_capacity_min);
            let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
            let len_ptr = <V64<T>>::allocate_array(heap_capacity) as *mut usize;
            let arr = (len_ptr as *mut u8).add(header_bytes) as *mut T;
            ptr::copy_nonoverlapping(self.stack_ptr(), arr, stack_capacity);
            *len_ptr += len;
            self.u = NonZeroU64::new_unchecked(len_ptr as u64);
        }
    }

    #[inline]
    fn heap_push(&mut self, val: T) {
        unsafe {
            let len_ptr = self.u.get() as usize as *mut usize;
            let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
            let arr = (len_ptr as *mut u8).add(header_bytes) as *mut T;
            ptr::write(arr.add(*len_ptr), val);
            *len_ptr += 1;
        }
    }

    #[inline]
    fn stack_push(&mut self, val: T, len: usize) {
        unsafe {
            let start = self.stack_ptr().add(len);
            ptr::write_unaligned(start, val);
            self.inc_stack_len(len);
        }
    }
}

impl<T: Clone> V64<T> {
    pub fn from_elem(elem: T, count: usize) -> V64<T> {
        let mut v64 = V64::with_capacity(count);
        for _i in 0..count as isize {
            v64.push(elem.clone());
        }
        v64
    }

    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.reserve(slice.len());
        unsafe {
            let mut len = self.len();
            let mut dst = self.get_unchecked_mut(len) as *mut T;
            for t in slice.iter() {
                ptr::write(dst, t.clone());
                dst = dst.add(1);
                len += 1;
                self.set_len(len); // we set len here in the loop in case clone breaks.
            }
        }
    }
}

impl<T> Drop for V64<T> {
    fn drop(&mut self) {
        unsafe {
            if mem::size_of::<T>() == 0 {
                self.u = NonZeroU64::new_unchecked(ZST_MASK);
                return;
            }
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());

                    if mem::needs_drop::<T>() {
                        let mut cur = (len_ptr as *mut u8).add(header_bytes) as *mut T;
                        let end = cur.add(*len_ptr);
                        while cur < end {
                            ptr::drop_in_place(cur);
                            cur = cur.add(1);
                        }
                    }

                    let align = cmp::max(16, mem::align_of::<T>());
                    let layout = Layout::from_size_align(mem::size_of::<T>() * (*(len_ptr.add(1))) + header_bytes, align).unwrap();
                    alloc::dealloc(ptr, layout);
                }
                Control::Stack(len) => {
                    if mem::needs_drop::<T>() {
                        let mut cur = self.stack_ptr();
                        let end = cur.add(len);
                        while cur < end {
                            ptr::drop_in_place(cur);
                            cur = cur.add(1);
                        }
                    }
                }
            }
            self.u = NonZeroU64::new_unchecked(8);
        }
    }
}

impl<T> V64<T> {
    /// Transmutes the vector `V64<T>` into another type of vector `V64<X>`.
    /// Consumes the original vector.
    ///
    /// `mem::size_of::<X>` must equal `mem::size_of::<T>`
    ///
    /// `mem::align_of::<X>` must equal `mem::align_of::<T>`
    ///
    /// This is achieved with no copying at all, making it super fast.
    /// `X` and `T` are enforced to not have a `Drop` implementation.
    ///
    /// This is useful and safe for all primitive types that have the same width
    /// For example:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let vecu: V64<u8> = v64![1, 2, 3, 128, 0xFF];
    /// unsafe {
    ///     let veci: V64<i8> = vecu.transmute();
    ///     assert_eq!(1, veci[0]);
    ///     assert_eq!(-128i8, veci[3]);
    ///     assert_eq!(-1i8, veci[4]);
    /// }
    /// # }
    /// ```
    ///
    /// Please note that this works the same way as mem::transmute.
    /// It does not do any sort of numeric conversion. It reuses the same
    /// bits for the new type. So going from `f32` to `i32` is not going to
    /// create sensible numbers, just the same "weird" integers that `f32::to_bits`
    /// produces.
    ///
    /// It might also be useful for converting from primitives to single elements structs,
    /// including some of the special types like `NonZeroU32`, if you know you don't violate
    /// any of the struct's invariants (e.g. no zeros if you're converting to `NonZeroU32`).
    ///
    /// One thing this can help with is with a total ordered floating point wrapper struct that will
    /// then allow for sorting, max, binary search, etc.
    /// see https://stackoverflow.com/questions/28247990/how-to-do-a-binary-search-on-a-vec-of-floats
    ///
    /// Avoid using this for (multi-element) `repr(Rust)` structs. It might be somewhat useful for
    /// `repr(C)` structs if you know what you're doing.
    ///
    /// Under no circumstances does it make sense to transmute to enum wrappers such as Option, as
    /// the bit pattern for these is not well specified (even if they happen to have the same size).
    ///
    /// # Panics
    ///
    /// Panics if the the size or alignment of X and T are different.
    ///
    pub unsafe fn transmute<X>(self) -> V64<X> {
        assert!(mem::size_of::<X>() == mem::size_of::<T>());
        assert!(mem::align_of::<X>() == mem::align_of::<T>());
        assert!(!mem::needs_drop::<X>());
        assert!(!mem::needs_drop::<T>());
        let v: V64<X> = V64 { u: self.u, _marker: marker::PhantomData };
        mem::forget(self);
        v
    }
}

impl<T, I> Index<I> for V64<T>
    where
        I: SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<T, I> IndexMut<I> for V64<T>
    where
        I: SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<T> Deref for V64<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            if mem::size_of::<T>() == 0 { return slice::from_raw_parts(1 as *const T, self.len()); }
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                    let arr = (len_ptr as *mut u8).add(header_bytes) as *mut T;
                    slice::from_raw_parts(arr, *len_ptr)
                }
                Control::Stack(len) => {
                    let arr = ((&self.u) as *const _ as *const u8).add(mem::align_of::<T>()) as *const T;
                    slice::from_raw_parts(arr, len)
                }
            }
        }
    }
}

impl<T> DerefMut for V64<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            if mem::size_of::<T>() == 0 { return slice::from_raw_parts_mut(1 as *mut T, self.len()); }
            match self.control() {
                Control::Heap(ptr) => {
                    let len_ptr = ptr as *mut usize;
                    let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                    let arr = (len_ptr as *mut u8).add(header_bytes) as *mut T;
                    slice::from_raw_parts_mut(arr, *len_ptr)
                }
                Control::Stack(len) => {
                    let arr = self.stack_ptr();
                    slice::from_raw_parts_mut(arr, len)
                }
            }
        }
    }
}

macro_rules! __impl_slice_eq1 {
    ($Lhs: ty, $Rhs: ty) => {
        __impl_slice_eq1! { $Lhs, $Rhs, Sized }
    };
    ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
        impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
            #[inline]
            fn eq(&self, other: &$Rhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$Rhs) -> bool { self[..] != other[..] }
        }
    }
}

__impl_slice_eq1! { V64<A>, V64<B> }
__impl_slice_eq1! { V64<A>, &'b [B] }
__impl_slice_eq1! { V64<A>, &'b mut [B] }
//__impl_slice_eq1! { &'b [A], V64<B> }
//__impl_slice_eq1! { &'b mut [A], V64<B> }

macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            // NOTE: some less important impls are omitted to reduce code bloat
            __impl_slice_eq1! { V64<A>, [B; $N] }
            __impl_slice_eq1! { V64<A>, &'b [B; $N] }
            // __impl_slice_eq1! { Vec<A>, &'b mut [B; $N] }
            // __impl_slice_eq1! { Cow<'a, [A]>, [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b [B; $N], Clone }
            // __impl_slice_eq1! { Cow<'a, [A]>, &'b mut [B; $N], Clone }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

impl<T: fmt::Debug> fmt::Debug for V64<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the `into_iter` method on [`V64`][`V64`] (provided
/// by the [`IntoIterator`] trait).
///
/// [`V64`]: struct.V64.html
/// [`IntoIterator`]: ../../std/iter/trait.IntoIterator.html
pub struct IntoIter<T> {
    buf: *mut u8,
    _marker: marker::PhantomData<T>,
    ptr: *const T,
    end: *const T,
    is_heap: bool,
}

impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

impl<T> IntoIterator for V64<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the vector (from start to end). The vector cannot be used after calling
    /// this.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let v = v64!["a".to_string(), "b".to_string()];
    /// for s in v.into_iter() {
    ///     // s has type String, not &String
    ///     println!("{}", s);
    /// }
    /// # }
    /// ```
    ///
    /// Works on stack too:
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let v: V64<u8> = v64![1, 2];
    /// for s in v.into_iter() {
    ///     println!("{}", s);
    /// }
    /// # }
    /// ```
    #[inline]
    fn into_iter(mut self) -> IntoIter<T> {
        unsafe {
            if mem::size_of::<T>() == 0 {
                let ptr = self.as_mut_ptr();
                let end = (ptr as *const u8).add(self.len()) as *const T;
                mem::forget(self);
                return IntoIter {
                    buf: ptr::null_mut(),
                    _marker: marker::PhantomData,
                    ptr,
                    end,
                    is_heap: false,
                };
            }
            if self.u.get() & 8 == 8 {
                let len = (self.u.get() & 7) as usize;
                if len == 0 {
                    return IntoIter {
                        buf: ptr::null_mut(),
                        _marker: marker::PhantomData,
                        ptr: mem::align_of::<T>() as *mut T,
                        end: mem::align_of::<T>() as *mut T,
                        is_heap: false,
                    };
                }
                self.move_to_heap(len, 0);
            }
            let len_ptr = self.u.get() as usize as *mut u8 as *mut usize;
            let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
            let begin = (len_ptr as *mut u8).add(header_bytes) as *mut T;
            let end = begin.offset(*len_ptr as isize) as *const T;

            let buf = len_ptr as *mut u8;
            mem::forget(self);
            IntoIter {
                buf,
                _marker: marker::PhantomData,
                ptr: begin,
                end,
                is_heap: true,
            }
        }
    }
}

impl<'a, T> IntoIterator for &'a V64<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut V64<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> IntoIter<T> {
    /// Returns the remaining items of this iterator as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let vec = v64!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// let _ = into_iter.next().unwrap();
    /// assert_eq!(into_iter.as_slice(), &['b', 'c']);
    /// # }
    /// ```
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.len())
        }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let vec = v64!['a', 'b', 'c'];
    /// let mut into_iter = vec.into_iter();
    /// assert_eq!(into_iter.as_slice(), &['a', 'b', 'c']);
    /// into_iter.as_mut_slice()[2] = 'z';
    /// assert_eq!(into_iter.next().unwrap(), 'a');
    /// assert_eq!(into_iter.next().unwrap(), 'b');
    /// assert_eq!(into_iter.next().unwrap(), 'z');
    /// # }
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr as *mut T, self.len())
        }
    }
}

unsafe impl<T: Send> Send for IntoIter<T> {}

unsafe impl<T: Sync> Sync for IntoIter<T> {}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            if self.ptr as *const _ == self.end {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    // cast to u8, so we add 1, not zero
                    self.ptr = (self.ptr as *mut u8).add(1) as *mut T;

                    // Use a non-null pointer value
                    // (self.ptr might be null because of wrapping)
                    Some(ptr::read(1 as *mut T))
                } else {
                    let old = self.ptr;
                    self.ptr = self.ptr.offset(1);

                    Some(ptr::read(old))
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = if mem::size_of::<T>() == 0 {
            (self.end as usize).wrapping_sub(self.ptr as usize)
        } else {
            (self.end as usize).wrapping_sub(self.ptr as usize) / mem::size_of::<T>()
        };
        (exact, Some(exact))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            if self.end == self.ptr {
                None
            } else {
                if mem::size_of::<T>() == 0 {
                    self.end = (self.end as *const i8).sub(1) as *mut T;

                    // Use a non-null pointer value
                    // (self.end might be null because of wrapping)
                    Some(ptr::read(1 as *mut T))
                } else {
                    self.end = self.end.offset(-1);

                    Some(ptr::read(self.end))
                }
            }
        }
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> FusedIterator for IntoIter<T> {}

impl<T: Clone> Clone for IntoIter<T> {
    fn clone(&self) -> IntoIter<T> {
        self.as_slice().to_v64().into_iter()
    }
}

pub trait ToV64<T> {
    fn to_v64(&self) -> V64<T>;
}

pub trait IntoV64<T> {
    fn into_v64(self) -> V64<T>;
}

impl<T: Clone> ToV64<T> for [T] {
    fn to_v64(&self) -> V64<T> {
        let mut vector = V64::with_capacity(self.len());
        vector.extend_desugared(self.iter().cloned());
        vector
    }
}

impl<T> IntoV64<T> for Box<[T]> {
    fn into_v64(self) -> V64<T> {
        unsafe {
            let len = self.len();
            let ptr = self.as_ptr();
            let mut vec = V64::with_capacity(len);
            ptr::copy(ptr, vec.as_mut_ptr(), len);
            vec.set_len(len);
            mem::forget(self); // prevent it from dropping the contents
            vec
        }
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        if mem::size_of::<T>() == 0 { return; }
        // destroy the remaining elements
        for _x in self.by_ref() {}

        unsafe {
            if self.is_heap {
                let len_ptr = self.buf as *mut usize;
                let header_bytes = cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>());
                let align = cmp::max(16, mem::align_of::<T>());
                let layout = Layout::from_size_align(mem::size_of::<T>() * (*(len_ptr.add(1))) + header_bytes, align).unwrap();
                alloc::dealloc(self.buf, layout);
                self.is_heap = false;
            }
        }
    }
}

impl<T> Extend<T> for V64<T> {
    #[inline]
    fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
        self.extend_desugared(iter.into_iter())
    }
}

impl<'a, T: 'a + Copy> Extend<&'a T> for V64<T> {
    #[inline]
    fn extend<I: IntoIterator<Item=&'a T>>(&mut self, iter: I) {
        self.extend_desugared(iter.into_iter().cloned())
    }
}

impl<T> Default for V64<T> {
    /// Creates an empty `V64<T>`.
    fn default() -> V64<T> {
        V64::new()
    }
}

impl<'a, T: Clone> From<&'a [T]> for V64<T> {
    fn from(s: &'a [T]) -> V64<T> {
        s.to_v64()
    }
}

impl<'a, T: Clone> From<&'a mut [T]> for V64<T> {
    fn from(s: &'a mut [T]) -> V64<T> {
        s.to_v64()
    }
}

impl<T> From<Box<[T]>> for V64<T> {
    fn from(s: Box<[T]>) -> V64<T> {
        s.into_v64()
    }
}

impl<'a, T> From<Cow<'a, [T]>> for V64<T> where [T]: ToOwned<Owned=V64<T>> {
    fn from(s: Cow<'a, [T]>) -> V64<T> {
        s.into_owned()
    }
}

impl<'a> From<&'a str> for V64<u8> {
    fn from(s: &'a str) -> V64<u8> {
        From::from(s.as_bytes())
    }
}

impl<T> FromIterator<T> for V64<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> V64<T> {
        let into_iter = iter.into_iter();
        let (lower, _) = into_iter.size_hint();
        let mut v64 = V64::with_capacity(lower);
        v64.extend_desugared(into_iter);
        v64
    }
}

/// A draining iterator for `V64<T>`.
///
/// This `struct` is created by the [`drain`] method on [`V64`].
///
/// [`drain`]: struct.V64.html#method.drain
/// [`V64`]: struct.V64.html
pub struct Drain<'a, T: 'a> {
    /// Index of tail to preserve
    tail_start: usize,
    /// Length of tail
    tail_len: usize,
    /// Current remaining range to remove
    iter: slice::Iter<'a, T>,
    vec: NonNull<V64<T>>,
}

impl<'a, T: 'a + fmt::Debug> fmt::Debug for Drain<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.iter.as_slice())
            .finish()
    }
}

unsafe impl<'a, T: Sync> Sync for Drain<'a, T> {}

unsafe impl<'a, T: Send> Send for Drain<'a, T> {}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|elt| unsafe { ptr::read(elt as *const _) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(|elt| unsafe { ptr::read(elt as *const _) })
    }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // exhaust self first
        self.for_each(drop);

        if self.tail_len > 0 {
            unsafe {
                let source_vec = self.vec.as_mut();
                // memmove back untouched tail, update to new length
                let start = source_vec.len();
                let tail = self.tail_start;
                if tail != start {
                    let src = source_vec.as_ptr().offset(tail as isize);
                    let dst = source_vec.as_mut_ptr().offset(start as isize);
                    ptr::copy(src, dst, self.tail_len);
                }
                source_vec.set_len(start + self.tail_len);
            }
        }
    }
}


impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T> FusedIterator for Drain<'a, T> {}

/// A splicing iterator for `V64`.
///
/// This struct is created by the [`splice()`] method on [`V64`]. See its
/// documentation for more.
///
/// [`splice()`]: struct.V64.html#method.splice
/// [`V64`]: struct.V64.html
#[derive(Debug)]
pub struct Splice<'a, I: Iterator + 'a> {
    drain: Drain<'a, I::Item>,
    replace_with: I,
}

impl<'a, I: Iterator> Iterator for Splice<'a, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<'a, I: Iterator> DoubleEndedIterator for Splice<'a, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<'a, I: Iterator> ExactSizeIterator for Splice<'a, I> {}


impl<'a, I: Iterator> Drop for Splice<'a, I> {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
//        unsafe { self.drain.vec.as_ref().debug_i32(); }

        unsafe {
            if self.drain.tail_len == 0 {
                self.drain.vec.as_mut().extend(self.replace_with.by_ref());
                return;
            }

            // First fill the range left by drain().
            if !self.drain.fill(&mut self.replace_with) {
                return;
            }

            // There may be more elements. Use the lower bound as an estimate.
            let (lower_bound, _upper_bound) = self.replace_with.size_hint();
            if lower_bound > 0 {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return;
                }
            }

            // Collect any remaining elements.
            // This is a zero-length vector which does not allocate if `lower_bound` was exact.
            let mut collected = self.replace_with.by_ref().collect::<V64<I::Item>>().into_iter();
            // Now we have an exact count.
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                let filled = self.drain.fill(&mut collected);
                debug_assert!(filled);
                debug_assert_eq!(collected.len(), 0);
            }
        }
        // Let `Drain::drop` move the tail back if necessary and restore `vec.len`.
    }
}

/// Private helper methods for `Splice::drop`
impl<'a, T> Drain<'a, T> {
    /// The range from `self.vec.len` to `self.tail_start` contains elements
    /// that have been moved out.
    /// Fill that range as much as possible with new elements from the `replace_with` iterator.
    /// Return whether we filled the entire range. (`replace_with.next()` didn’t return `None`.)
    unsafe fn fill<I: Iterator<Item=T>>(&mut self, replace_with: &mut I) -> bool {
        let vec = self.vec.as_mut();
        let range_start = vec.len();
        let range_end = self.tail_start;
        let range_slice = slice::from_raw_parts_mut(
            vec.as_mut_ptr().offset(range_start as isize),
            range_end - range_start);

        let mut count = range_start;
        for place in range_slice {
            if let Some(new_item) = replace_with.next() {
                ptr::write(place, new_item);
                count += 1;
                vec.set_len(count);
            } else {
                return false;
            }
        }
        true
    }

    /// Make room for inserting more elements before the tail.
    unsafe fn move_tail(&mut self, extra_capacity: usize) {
        let vec = self.vec.as_mut();
        vec.reserve(extra_capacity);

        let new_tail_start = self.tail_start + extra_capacity;
        let src = vec.as_ptr().offset(self.tail_start as isize);
        let dst = vec.as_mut_ptr().offset(new_tail_start as isize);
        ptr::copy(src, dst, self.tail_len);
        self.tail_start = new_tail_start;
    }
}

pub trait CloneIntoV64<T> {
    fn clone_into_v64(&self, target: &mut V64<T>);
}

impl<T: Clone> CloneIntoV64<T> for [T] {
    fn clone_into_v64(&self, target: &mut V64<T>) {
        // drop anything in target that will not be overwritten
        target.truncate(self.len());
        let len = target.len();

        // reuse the contained values' allocations/resources.
        target.clone_from_slice(&self[..len]);

        // target.len <= self.len due to the truncate above, so the
        // slice here is always in-bounds.
        target.extend_from_slice(&self[len..]);
    }
}

impl<T: Clone> Clone for V64<T> {
    fn clone(&self) -> V64<T> {
        <[T]>::to_v64(&**self)
    }

    fn clone_from(&mut self, other: &V64<T>) {
        other.as_slice().clone_into_v64(self);
    }
}

/// An iterator produced by calling `drain_filter` on V64.
#[derive(Debug)]
pub struct DrainFilter<'a, T: 'a, F>
    where F: FnMut(&mut T) -> bool,
{
    vec: &'a mut V64<T>,
    idx: usize,
    del: usize,
    old_len: usize,
    pred: F,
}

impl<'a, T, F> Iterator for DrainFilter<'a, T, F>
    where F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            while self.idx != self.old_len {
                let i = self.idx;
                self.idx += 1;
                let v = slice::from_raw_parts_mut(self.vec.as_mut_ptr(), self.old_len);
                if (self.pred)(&mut v[i]) {
                    self.del += 1;
                    return Some(ptr::read(&v[i]));
                } else if self.del > 0 {
                    let del = self.del;
                    let src: *const T = &v[i];
                    let dst: *mut T = &mut v[i - del];
                    // This is safe because self.vec has length 0
                    // thus its elements will not have Drop::drop
                    // called on them in the event of a panic.
                    ptr::copy_nonoverlapping(src, dst, 1);
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.old_len - self.idx))
    }
}

impl<'a, T, F> Drop for DrainFilter<'a, T, F>
    where F: FnMut(&mut T) -> bool,
{
    fn drop(&mut self) {
        self.for_each(drop);
        unsafe {
            self.vec.set_len(self.old_len - self.del);
        }
    }
}

impl<T: PartialEq> V64<T> {
    /// Removes consecutive repeated elements in the vector.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = vec![1, 2, 2, 3, 2];
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// # }
    /// ```
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }

    /// Removes the first instance of `item` from the vector if the item exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_v64::V64;
    /// # fn main() {
    /// let mut vec = v64![1, 2, 3, 1];
    ///
    /// vec.remove_item(&1);
    ///
    /// assert_eq!(vec, v64![2, 3, 1]);
    /// # }
    /// ```
    pub fn remove_item(&mut self, item: &T) -> Option<T> {
        let pos = self.iter().position(|x| *x == *item)?;
        Some(self.remove(pos))
    }
}

impl<T> Borrow<[T]> for V64<T> {
    fn borrow(&self) -> &[T] {
        &self[..]
    }
}

impl<T> BorrowMut<[T]> for V64<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}