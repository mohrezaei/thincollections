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

//! # `ThinVec` a general `Vec` replacement in a single usize-sized pointer.
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
use std::num::NonZeroUsize;

/// A thin (usize) vector. Guaranteed to be a usize-sized smart pointer.
///
/// Rust's `std::collections::Vec` (`std::Vec` for short) is a triple-fat (3 x usize) pointer to the heap.
/// When `std::Vec` is on the stack, it works well. However, when `std::Vec` is used inside other data
/// structures, such as `Vec<Vec<_>>`, the triple fatness starts to become a problem. For example,
/// when a `Vec<Vec<_>>` has to resize, it needs to move 3 times as much memory.
///
/// `ThinVec` uses a single usize value as a smart pointer, making it attractive for building larger
/// data structures.
///
/// `ThinVec` is also null optimized, which makes an `Option<ThinVec<_>>` also usize-sized.
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
/// can be slow. For this reason, it is recommended to use [`ThinVec::with_capacity`]
/// whenever possible to specify how big the vector is expected to get.
///
/// # Usage Patterns
/// Aside from the simple single element methods, `push`, `pop`, `insert` and `remove`,
/// ThinVec, just as `std::Vec` has a large number of API's that can be a bit hard
/// to navigate, especially because it also dereferences to a slice, which has a massive
/// api count.
///
/// # Initialization Patterns
///
/// Fill a `ThinVec` with some clonable object:
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let v: ThinVec<i32> = thinvec![7; 42];
/// assert_eq!(42, v.len());
/// assert_eq!(7, v[0]);
/// assert_eq!(7, v[41]);
/// # }
/// ```
/// The macro just calls `ThinVec::from_elem(7, 42)`
/// which makes sure the vector is created with [`ThinVec::with_capacity`]
/// before filling it.
///
/// Fill a `ThinVec` with the result of some function:
///
/// ```
/// use thincollections::thin_vec::ThinVec;
/// let v : ThinVec<i32> = (0..42).map(|x| x+7).collect(); // x+7 is the example function here
/// assert_eq!(42, v.len());
/// assert_eq!(7, v[0]);
/// assert_eq!(48, v[41]);
/// ```
/// The above patterns is also useful for types that do funny stuff when cloned.
/// For example, when `ThinVec` is cloned, it only copies the values, not the capacity, so
/// trying to create a `ThinVec<ThinVec<_>>` with a given capacity for the inner vectors
/// should be done via a loop or as above, `collect`.
///
/// # Sorting
/// `ThinVec` can be sorted because a slice can be sorted and `ThinVec` derefs to slice.
/// To be sortable, the type inside the `ThinVec` must implement `cmp::Ord`.
/// Alternatively, the `sort_by` method can be used to specify a custom comparator.
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut v: ThinVec<i32> = thinvec![3, 1, 2, -1];
/// v.sort(); // the compiler turns this into: v.as_mut_slice().sort()
/// assert_eq!(-1, v[0]);
/// assert_eq!(3, v[3]);
/// # }
/// ```
///
/// Sorting floating point numbers is harder because they don't implement `cmp::Ord`.
/// This is because of the weird values in floats, most importantly `NaN`, which has
/// very odd semantics regarding equality (it's not equal to anything, not even itself).
/// We can use [`ThinVec::transmute`] to get around these issues.
///
/// ```
/// #[macro_use] extern crate thincollections;
/// extern crate ordered_float;
///
/// use thincollections::thin_vec::ThinVec;
/// use ordered_float::NotNan;
/// # fn main() {
/// let mut v: ThinVec<f32> = thinvec![3.0, 1.0, 2.0, -1.0];
/// unsafe {
///     let mut v_notnan: ThinVec<NotNan<f32>> = v.transmute();
///     v_notnan.sort();
///     let v: ThinVec<f32> = v_notnan.transmute();
///     assert_eq!(-1.0, v[0]);
///     assert_eq!(3.0, v[3]);
/// }
/// # }
/// ```
///
/// After sorting a `ThinVec`, you can then use `binary_search` to find stuff.
///
/// # Iteration
/// `ThinVec` "inherits" its iteration from slice. Iterating through `ThinVec` is far more
/// efficient compared to indexing operations.
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let v: ThinVec<i32> = thinvec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let mut sum = 0;
/// for i in v.iter() { // i is a &i32, not i32 here.
///     sum += i; // the compiler turns this into sum += *i;
/// }
/// assert_eq!(55, sum);
/// # }
/// ```
/// Because `iter()` produces a Rust `Iterator`, which has a large API, many
/// things can be done in a functional style without the explicit loop:
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let v: ThinVec<i32> = thinvec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// assert_eq!(55, v.iter().fold(0, |sum, x| sum + x)); // same as the loop above
/// assert_eq!(550, v.iter().map(|x| x * 10).fold(0, |sum, x| sum + x));
/// assert_eq!([2, 4, 6, 8, 10], v.iter().filter(|x| *x & 1 == 0).cloned().collect::<ThinVec<i32>>().as_slice());
/// # }
/// ```
///
/// We can also iterate and mutate (in place):
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut v: ThinVec<i32> = thinvec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// for i in v.iter_mut() { // i is a &mut i32
///     *i = *i + 10; // i points inside the vector!
/// }
/// assert_eq!(11, v[0]);
/// assert_eq!(20, v[9]);
/// # }
/// ```
///
/// Finally, we can move the contents of the vector out using `into_iter()`
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let v: ThinVec<i32> = thinvec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let mut sum = 0;
/// for i in v.into_iter() { // i is i32 here (not &i32).
///     sum += i;
/// }
/// // can't access v anymore here, it's been consumed.
/// assert_eq!(55, sum);
/// # }
/// ```
///
/// # Concatenation
///
/// Concatenating vectors to vectors requires understanding the ownership/copy sematics of Rust.
/// Vectors own their contents, so a simple concatenation using the [`ThinVec::append`]
/// will move the contents (leaving one of the vectors empty):
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut v: ThinVec<i32> = thinvec![10, 20, 30];
/// let mut v2: ThinVec<i32> = thinvec![40, 50, 60];
/// v.append(&mut v2); // we take a mutable reference, because we're about to empty out v2
/// assert_eq!(6, v.len());
/// assert_eq!(60, v[5]);
/// assert_eq!(0, v2.len());
/// # }
/// ```
///
/// Vectors can also be extended from clonable slices (and a vector derefs to a slice!).
/// We can use this to concatenate two vectors without emptying one out:
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut v: ThinVec<i32> = thinvec![10, 20, 30];
/// let v2: ThinVec<i32> = thinvec![40, 50, 60];
/// v.extend(&v2); // the compiler turns this into v2.as_slice();
/// assert_eq!(6, v.len());
/// assert_eq!(60, v[5]);
/// assert_eq!(3, v2.len());
/// # }
/// ```
///
/// We also use the same to concatenate a slice to a `ThinVec`:
///
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut v: ThinVec<i32> = thinvec![10, 20, 30];
/// let s: [i32; 3] = [40, 50, 60]; // s is an array
/// v.extend(&s); // &s is a slice, not an array
/// assert_eq!(6, v.len());
/// assert_eq!(60, v[5]);
/// assert_eq!(3, s.len());
/// # }
/// ```
///
/// # Splitting
///
/// You can split (and move the contents) of a vector into another vector:
/// ```
/// # #[macro_use] extern crate thincollections;
/// # use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut vec = thinvec![1,2,3];
/// let vec2 = vec.split_off(1);
/// assert_eq!(vec, [1]);
/// assert_eq!(vec2, [2, 3]);
/// # }
/// ```
///
/// You can also mutably borrow multiple parts of a vector using the `split_at_mut`
/// slice method:
/// ```
/// #[macro_use] extern crate thincollections;
/// use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let mut v = thinvec![1, 0, 3, 0, 5, 6];
/// // scoped to restrict the lifetime of the borrows
/// {
///     let (left, right) = v.split_at_mut(2);
///     assert!(left == [1, 0]);
///     assert!(right == [3, 0, 5, 6]);
///     left[1] = 2;
///     right[1] = 4;
/// }
/// assert!(v == [1, 2, 3, 4, 5, 6]);
/// # }
/// ```
///
///
pub struct ThinVec<T> {
    u: NonZeroUsize,
    _marker: marker::PhantomData<T>,
}

#[cfg(target_pointer_width = "64")]
const ZST_MASK: usize = 0x8000_0000_0000_0000;

#[cfg(target_pointer_width = "32")]
const ZST_MASK: usize = 0x8000_0000;

const DANGLE: usize = <usize>::max_value(); // it is impossible for alloc to return this value, as we require at least 2*usize space

impl<T> ThinVec<T> {
    /// Constructs a new, empty `ThinVec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_vec::ThinVec;
    /// let mut vec: ThinVec<i32> = ThinVec::new();
    /// vec.push(17); // initial allocation
    /// vec.push(42);
    /// ```
    #[inline]
    pub fn new() -> ThinVec<T> {
        unsafe {
            if mem::size_of::<T>() == 0 { return ThinVec { u: NonZeroUsize::new_unchecked(ZST_MASK), _marker: marker::PhantomData }; }
            ThinVec { u: NonZeroUsize::new_unchecked(DANGLE), _marker: marker::PhantomData }
        }
    }

    /// Constructs a new, empty `ThinVec<T>` with the specified capacity.
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
    /// use thincollections::thin_vec::ThinVec;
    /// let mut vec = ThinVec::with_capacity(10);
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
    pub fn with_capacity(capacity: usize) -> ThinVec<T> {
        if mem::size_of::<T>() == 0 { return ThinVec::new(); }
        if capacity == 0 {
            return <ThinVec<T>>::new();
        }
        let array = <ThinVec<T>>::allocate_array(capacity);
        unsafe {
            ThinVec { u: NonZeroUsize::new_unchecked(array as usize), _marker: marker::PhantomData }
        }
    }

    /// Returns the number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_vec::ThinVec;
    /// let vec: ThinVec<i32> = ThinVec::with_capacity(10);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 { return <usize>::max_value(); }
        if self.u.get() == DANGLE { return 0; }
        unsafe {
            let len_ptr = self.u.get() as *mut usize;
            return *len_ptr.add(1);
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `ThinVec<T>`. The collection may reserve more space to avoid
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
    /// use thincollections::thin_vec::ThinVec;
    /// let mut vec = ThinVec::new();
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
    /// be inserted in the given `ThinVec<T>`. After calling `reserve_exact`,
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
    /// use thincollections::thin_vec::ThinVec;
    /// let mut vec = ThinVec::new();
    /// vec.push(1);
    /// vec.reserve_exact(10);
    /// assert_eq!(11, vec.capacity());
    /// ```
    pub fn reserve_exact(&mut self, additional: usize) {
        if mem::size_of::<T>() == 0 { return; }
        if self.u.get() == DANGLE {
            unsafe {
                self.u = NonZeroUsize::new_unchecked(<ThinVec<T>>::allocate_array(additional) as usize);
                return;
            }
        }
        let len = self.len();
        let remain = self.capacity() - len;
        if remain < additional {
            let len_ptr = self.u.get() as *mut usize;
            unsafe {
                self.realloc_heap(len_ptr, *len_ptr + additional);
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
    /// use thincollections::thin_vec::ThinVec;
    /// let mut vec = ThinVec::with_capacity(10);
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
        if self.u.get() == DANGLE { return; }
        let len_ptr = self.u.get() as *mut usize;
        unsafe {
            self.realloc_heap(len_ptr, *len_ptr);
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3, 4, 5];
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3];
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3];
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
        if self.u.get() == DANGLE { return; }
        unsafe {
            let len_ptr = self.u.get() as *mut usize;
            if *len_ptr > len {
                if mem::needs_drop::<T>() {
                    let mut cur = ((len_ptr as *mut u8).add(<ThinVec<T>>::header_bytes()) as *mut T).add(len);
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

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// use std::io::{self, Write};
    /// let buffer = thinvec![1, 2, 3, 5, 8];
    /// io::sink().write(buffer.as_slice()).unwrap();
    /// # }
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// use std::io::{self, Read};
    /// let mut buffer = thinvec![0; 3];
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut v = thinvec!["foo", "bar", "baz", "qux"];
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(v[..], ["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v[..], ["baz", "qux"]);
    /// # }
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len() - 1;
                self.set_len(len);
                return ptr::read(NonNull::dangling().as_ptr());
            }
        }
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check on hole succeeds there must be a last element (which
            // can be self[index] itself).
            if index < self.len() {
                let len_ptr = self.u.get() as *mut usize;
                *len_ptr -= 1;
                let ptr = len_ptr as *mut u8;
                <ThinVec<T>>::replace(ptr.add(<ThinVec<T>>::header_bytes()) as *mut T, *len_ptr, index)
            } else {
                panic!("index out of bounds! len: {}, index {}", self.len(), index);
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, [1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, [1, 4, 2, 3, 5]);
    /// # }
    /// ```
    #[inline]
    pub fn insert(&mut self, index: usize, val: T) {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len();
                assert!(index <= len);
                self.set_len(len + 1);
                return;
            }
        }
        assert!(index <= self.len());
        self.possibly_grow_heap();
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut v = thinvec![1, 2, 3];
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
                return ptr::read(NonNull::dangling().as_ptr());
            }
        }
        let array: *mut T;
        let len: usize;
        let len_ptr = self.u.get() as *mut usize;
        let ptr = len_ptr as *mut u8;
        unsafe {
            len = *len_ptr;
            *len_ptr -= 1;
            array = ptr.add(<ThinVec<T>>::header_bytes()) as *mut T;
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3, 4];
    /// vec.retain(|&x| x%2 == 0);
    /// assert_eq!(vec, [2, 4]);
    /// # }
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        if mem::size_of::<T>() == 0 {
            let mut count = 0;
            unsafe {
                let t: T = ptr::read(NonNull::dangling().as_ptr());
                for _i in 0..self.len() {
                    if f(&t) { count += 1; }
                }
                self.set_len(count);
            }
            return;
        }
        let mut array: *mut T;
        let len: usize;
        {
            let len_ptr = self.u.get() as *mut usize;
            let ptr = len_ptr as *mut u8;
            unsafe {
                len = *len_ptr;
                array = ptr.add(<ThinVec<T>>::header_bytes()) as *mut T;
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
            let len_ptr = self.u.get() as usize as *mut usize;
            *len_ptr -= removed;
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec!["foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![10, 20, 21, 30, 20];
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let a = thinvec![1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// # }
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        if mem::size_of::<T>() == 0 { return (self.u.get() & (ZST_MASK - 1)) as usize; }
        unsafe { return if self.u.get() == DANGLE { 0 } else { *(self.u.get() as *mut usize) }; };
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_vec::ThinVec;
    /// let mut v = ThinVec::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2];
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
        self.possibly_grow_heap();
        unsafe {
            let len_ptr = self.u.get() as usize as *mut usize;
            let arr = (len_ptr as *mut u8).add(<ThinVec<T>>::header_bytes()) as *mut T;
            ptr::write(arr.add(*len_ptr), val);
            *len_ptr += 1;
        }
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3];
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
                    return Some(ptr::read(NonNull::dangling().as_ptr()));
                }
            }
            return None;
        }
        if self.len() == 0 { return None; }
        let array: *mut T;
        let len: usize;
        {
            let len_ptr = self.u.get() as *mut usize;
            unsafe {
                len = *len_ptr;
                if len == 0 { return None; }
                let ptr = len_ptr as *mut u8;
                *len_ptr -= 1;
                array = ptr.add(<ThinVec<T>>::header_bytes()) as *mut T;
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3];
    /// let mut vec2 = thinvec![4, 5, 6];
    /// vec.append(&mut vec2);
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6]);
    /// assert_eq!(vec2, []);
    /// # }
    /// ```
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let len = self.len() + other.len();
                self.set_len(len);
                other.u = NonZeroUsize::new_unchecked(ZST_MASK);
                return;
            }
        }
        unsafe {
            self.append_elements(other.as_slice() as _);
            other.set_len(0);
        }
    }

    #[inline]
    unsafe fn append_elements(&mut self, other: *const [T]) {
        let count = (*other).len();
        if count > 0 {
            self.reserve(count);
            let len = self.len();
            ptr::copy_nonoverlapping(other as *const T, self.get_unchecked_mut(len), count);
            self.set_len(len + count);
        }
    }

    unsafe fn set_len(&mut self, len: usize) {
        if mem::size_of::<T>() == 0 {
            self.u = NonZeroUsize::new_unchecked(len | ZST_MASK);
            return;
        }
        if self.u.get() == DANGLE {
            if len != 0 {
                panic!("can't set length of empyty ");
            } else {
                return;
            }
        }
        let len_ptr = self.u.get() as *mut usize;
        *len_ptr = len;
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1,2,3];
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
        let mut other = ThinVec::with_capacity(other_len);

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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut v = thinvec![1, 2, 3];
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
    /// be avoided. ThinVec<T> is far more versatile than Box<[T]>
    ///
    /// Note that this will drop any excess capacity.
    ///
    /// [owned slice]: ../../std/boxed/struct.Box.html
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let v = thinvec![1, 2, 3];
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # use thincollections::thin_vec::IntoThinVec;
    /// # fn main() {
    /// let mut vec = ThinVec::with_capacity(10);
    /// vec.extend([1, 2, 3].iter().cloned());
    ///
    /// assert_eq!(vec.capacity(), 10);
    /// let slice: Box<[i32]> = vec.into_boxed_slice();
    /// assert_eq!(slice.into_thinvec().capacity(), 3);
    /// # }
    /// ```
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        if mem::size_of::<T>() == 0 {
            unsafe {
                let slice = slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), self.len());
                let output: Box<[T]> = Box::from_raw(slice);
                return output;
            }
        }
        unsafe {
            let array: *mut T;
            let len: usize;
            let len_ptr = self.u.get() as *mut usize;
            array = (len_ptr as *mut u8).add(<ThinVec<T>>::header_bytes()) as *mut T;
            len = *len_ptr;
            *len_ptr = 0; // prevents dropping of contents
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut v = thinvec![1, 2, 3];
    /// let u: ThinVec<_> = v.drain(1..).collect();
    /// assert_eq!(v, &[1]);
    /// assert_eq!(u, &[2, 3]);
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut v = thinvec![1, 2, 3];
    /// let new = [7, 8];
    /// let u: ThinVec<_> = v.splice(..2, new.iter().cloned()).collect();
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// # let some_predicate = |x: &mut i32| { *x == 2 || *x == 3 || *x == 6 };
    /// # let mut vec = thinvec![1, 2, 3, 4, 5, 6];
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
    /// # assert_eq!(vec, thinvec![1, 4, 5]);
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut numbers = thinvec![1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15];
    ///
    /// let evens = numbers.drain_filter(|x| *x % 2 == 0).collect::<ThinVec<_>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens, thinvec![2, 4, 6, 8, 14]);
    /// assert_eq!(odds, thinvec![1, 3, 5, 9, 11, 13, 15]);
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
    fn allocate_array(capacity: usize) -> *mut u8 {
        unsafe {
            // align_of is a power of 2. 2 * size_of::<usize> is a power of 2.
            // if align_of is smaller than 2*size_of::<usize>, we'll have no padding between the header and array
            // if align_of is bigger than 2*size_of::<usize>, we'll use the first align for the header
            let align = cmp::max(mem::align_of::<usize>(), mem::align_of::<T>());
            let layout = Layout::from_size_align(mem::size_of::<T>() * capacity + <ThinVec<T>>::header_bytes(), align).unwrap();
            let buffer = alloc::alloc(layout);
            ptr::write(buffer as *mut usize, 0); // current length
            ptr::write((buffer as *mut usize).add(1), capacity);
            buffer
        }
    }

    #[inline]
    fn possibly_grow_heap(&mut self) {
        unsafe {
            if self.u.get() == DANGLE {
                self.u = NonZeroUsize::new_unchecked(<ThinVec<T>>::allocate_array(8) as usize);
                return;
            }
            let len_ptr = self.u.get() as *mut usize;
            let cap_ptr = len_ptr.add(1);
            if *len_ptr == *cap_ptr {
                self.realloc_heap(len_ptr, *len_ptr * 2);
            }
        }
    }

    #[inline(always)]
    fn header_bytes() -> usize {
        cmp::max(mem::size_of::<usize>() * 2, mem::align_of::<T>())
    }

    #[inline(always)]
    fn realloc_heap(&mut self, len_ptr: *mut usize, new_capacity: usize) {
        unsafe {
            //realloc seems to bench slower...
            let head_ptr = <ThinVec<T>>::allocate_array(new_capacity);
            let header_bytes = <ThinVec<T>>::header_bytes();
            let to_move = cmp::min(new_capacity, *(len_ptr.add(1)));
            ptr::copy_nonoverlapping(len_ptr as *mut u8, head_ptr, to_move * mem::size_of::<T>() + header_bytes);
            let new_cap_ptr = (head_ptr as *mut usize).add(1);
            *new_cap_ptr = new_capacity;
            self.u = NonZeroUsize::new_unchecked(head_ptr as usize);
            let old = len_ptr as *mut u8;
            let align = cmp::max(mem::align_of::<usize>(), mem::align_of::<T>());
            let layout = Layout::from_size_align(mem::size_of::<T>() * (*(len_ptr.add(1))) + header_bytes, align).unwrap();
            alloc::dealloc(old, layout);
        }
    }
}

impl<T: Clone> ThinVec<T> {
    pub fn from_elem(elem: T, count: usize) -> ThinVec<T> {
        let mut thinvec = ThinVec::with_capacity(count);
        for _i in 0..count as isize {
            thinvec.push(elem.clone());
        }
        thinvec
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

impl<T> Drop for ThinVec<T> {
    fn drop(&mut self) {
        unsafe {
            if mem::size_of::<T>() == 0 {
                self.u = NonZeroUsize::new_unchecked(ZST_MASK);
                return;
            }
            if self.u.get() == DANGLE { return; }
            let len_ptr = self.u.get() as *mut usize;
            let header_bytes = <ThinVec<T>>::header_bytes();

            if mem::needs_drop::<T>() {
                let mut cur = (len_ptr as *mut u8).add(header_bytes) as *mut T;
                let end = cur.add(*len_ptr);
                while cur < end {
                    ptr::drop_in_place(cur);
                    cur = cur.add(1);
                }
            }

            let align = cmp::max(mem::align_of::<usize>(), mem::align_of::<T>());
            let layout = Layout::from_size_align(mem::size_of::<T>() * (*(len_ptr.add(1))) + header_bytes, align).unwrap();
            alloc::dealloc(self.u.get() as *mut u8, layout);
            self.u = NonZeroUsize::new_unchecked(DANGLE);
        }
    }
}

impl<T> ThinVec<T> {
    /// Transmutes the vector `ThinVec<T>` into another type of vector `ThinVec<X>`.
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let vecu: ThinVec<u8> = thinvec![1, 2, 3, 128, 0xFF];
    /// unsafe {
    ///     let veci: ThinVec<i8> = vecu.transmute();
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
    /// repr(transparent) is a good annotation to use on such types.
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_vec::ThinVec;
    /// use std::cell::Cell;
    /// # fn main() {
    /// let vecf: ThinVec<f64> = thinvec![1.0, 2.0, 3.0];
    /// unsafe {
    ///     let vecc: ThinVec<Cell<f64>> = vecf.transmute();
    ///     let vecf: ThinVec<f64> = vecc.transmute();
    /// }
    /// # }
    /// ```
    ///
    /// One thing this can help with is with a total ordered floating point wrapper struct that will
    /// then allow for sorting, max, binary search, etc.
    /// see [Sorting] for an example
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
    /// [Sorting]: #sorting
    ///
    pub unsafe fn transmute<X>(self) -> ThinVec<X> {
        assert!(mem::size_of::<X>() == mem::size_of::<T>());
        assert!(mem::align_of::<X>() == mem::align_of::<T>());
        assert!(!mem::needs_drop::<X>());
        assert!(!mem::needs_drop::<T>());
        let v: ThinVec<X> = ThinVec { u: self.u, _marker: marker::PhantomData };
        mem::forget(self);
        v
    }
}

impl<T, I> Index<I> for ThinVec<T>
    where
        I: SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<T, I> IndexMut<I> for ThinVec<T>
    where
        I: SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<T> Deref for ThinVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            if mem::size_of::<T>() == 0 { return slice::from_raw_parts(NonNull::dangling().as_ptr(), self.len()); }
            let len: usize;
            let arr: *mut T;
            if self.u.get() == DANGLE {
                len = 0;
                arr = NonNull::dangling().as_ptr();
            } else {
                let len_ptr = self.u.get() as *mut usize;
                len = *len_ptr;
                arr = (len_ptr as *mut u8).add(<ThinVec<T>>::header_bytes()) as *mut T;
            }
            slice::from_raw_parts(arr, len)
        }
    }
}

impl<T> DerefMut for ThinVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            if mem::size_of::<T>() == 0 { return slice::from_raw_parts_mut(NonNull::dangling().as_ptr(), self.len()); }
            let len: usize;
            let arr: *mut T;
            if self.u.get() == DANGLE {
                len = 0;
                arr = NonNull::dangling().as_ptr();
            } else {
                let len_ptr = self.u.get() as *mut usize;
                len = *len_ptr;
                arr = (len_ptr as *mut u8).add(<ThinVec<T>>::header_bytes()) as *mut T;
            }
            slice::from_raw_parts_mut(arr, len)
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

__impl_slice_eq1! { ThinVec<A>, ThinVec<B> }
__impl_slice_eq1! { ThinVec<A>, &'b [B] }
__impl_slice_eq1! { ThinVec<A>, &'b mut [B] }
//__impl_slice_eq1! { &'b [A], ThinVec<B> }
//__impl_slice_eq1! { &'b mut [A], ThinVec<B> }

macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            // NOTE: some less important impls are omitted to reduce code bloat
            __impl_slice_eq1! { ThinVec<A>, [B; $N] }
            __impl_slice_eq1! { ThinVec<A>, &'b [B; $N] }
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

impl<T: fmt::Debug> fmt::Debug for ThinVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// An iterator that moves out of a vector.
///
/// This `struct` is created by the `into_iter` method on [`ThinVec`][`ThinVec`] (provided
/// by the [`IntoIterator`] trait).
///
/// [`ThinVec`]: struct.ThinVec.html
/// [`IntoIterator`]: ../../std/iter/trait.IntoIterator.html
pub struct IntoIter<T> {
    buf: *mut u8,
    _marker: marker::PhantomData<T>,
    ptr: *const T,
    end: *const T,
}

impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.as_slice())
            .finish()
    }
}

impl<T> IntoIterator for ThinVec<T> {
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let v = thinvec!["a".to_string(), "b".to_string()];
    /// for s in v.into_iter() {
    ///     // s has type String, not &String
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
                };
            }
            if self.u.get() == DANGLE {
                return IntoIter {
                    buf: ptr::null_mut(),
                    _marker: marker::PhantomData,
                    ptr: ptr::null_mut(),
                    end: ptr::null_mut(),
                };
            }
            let len_ptr = self.u.get() as usize as *mut u8 as *mut usize;
            let begin = (len_ptr as *mut u8).add(<ThinVec<T>>::header_bytes()) as *mut T;
            let end = begin.offset(*len_ptr as isize) as *const T;

            let buf = len_ptr as *mut u8;
            mem::forget(self);
            IntoIter {
                buf,
                _marker: marker::PhantomData,
                ptr: begin,
                end,
            }
        }
    }
}

impl<'a, T> IntoIterator for &'a ThinVec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ThinVec<T> {
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let vec = thinvec!['a', 'b', 'c'];
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let vec = thinvec!['a', 'b', 'c'];
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
                    Some(ptr::read(NonNull::dangling().as_ptr()))
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
                    Some(ptr::read(NonNull::dangling().as_ptr()))
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
        self.as_slice().to_thinvec().into_iter()
    }
}

pub trait ToThinVec<T> {
    fn to_thinvec(&self) -> ThinVec<T>;
}

pub trait IntoThinVec<T> {
    fn into_thinvec(self) -> ThinVec<T>;
}

impl<T: Clone> ToThinVec<T> for [T] {
    fn to_thinvec(&self) -> ThinVec<T> {
        let mut vector = ThinVec::with_capacity(self.len());
        vector.extend_desugared(self.iter().cloned());
        vector
    }
}

impl<T> IntoThinVec<T> for Box<[T]> {
    fn into_thinvec(self) -> ThinVec<T> {
        unsafe {
            let len = self.len();
            let ptr = self.as_ptr();
            let mut vec = ThinVec::with_capacity(len);
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
            if !self.buf.is_null() {
                let len_ptr = self.buf as *mut usize;
                let align = cmp::max(mem::align_of::<usize>(), mem::align_of::<T>());
                let layout = Layout::from_size_align(mem::size_of::<T>() * (*(len_ptr.add(1))) + <ThinVec<T>>::header_bytes(), align).unwrap();
                alloc::dealloc(self.buf, layout);
                self.buf = ptr::null_mut();
            }
        }
    }
}

impl<T> Extend<T> for ThinVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
        self.extend_desugared(iter.into_iter())
    }
}

impl<'a, T: 'a + Copy> Extend<&'a T> for ThinVec<T> {
    #[inline]
    fn extend<I: IntoIterator<Item=&'a T>>(&mut self, iter: I) {
        self.extend_desugared(iter.into_iter().cloned())
    }
}

impl<T> Default for ThinVec<T> {
    /// Creates an empty `ThinVec<T>`.
    fn default() -> ThinVec<T> {
        ThinVec::new()
    }
}

impl<'a, T: Clone> From<&'a [T]> for ThinVec<T> {
    fn from(s: &'a [T]) -> ThinVec<T> {
        s.to_thinvec()
    }
}

impl<'a, T: Clone> From<&'a mut [T]> for ThinVec<T> {
    fn from(s: &'a mut [T]) -> ThinVec<T> {
        s.to_thinvec()
    }
}

impl<T> From<Box<[T]>> for ThinVec<T> {
    fn from(s: Box<[T]>) -> ThinVec<T> {
        s.into_thinvec()
    }
}

impl<'a, T> From<Cow<'a, [T]>> for ThinVec<T> where [T]: ToOwned<Owned=ThinVec<T>> {
    fn from(s: Cow<'a, [T]>) -> ThinVec<T> {
        s.into_owned()
    }
}

impl<'a> From<&'a str> for ThinVec<u8> {
    fn from(s: &'a str) -> ThinVec<u8> {
        From::from(s.as_bytes())
    }
}

impl<T> FromIterator<T> for ThinVec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> ThinVec<T> {
        let into_iter = iter.into_iter();
        let (lower, _) = into_iter.size_hint();
        let mut thinvec = ThinVec::with_capacity(lower);
        thinvec.extend_desugared(into_iter);
        thinvec
    }
}

/// A draining iterator for `ThinVec<T>`.
///
/// This `struct` is created by the [`drain`] method on [`ThinVec`].
///
/// [`drain`]: struct.ThinVec.html#method.drain
/// [`ThinVec`]: struct.ThinVec.html
pub struct Drain<'a, T: 'a> {
    /// Index of tail to preserve
    tail_start: usize,
    /// Length of tail
    tail_len: usize,
    /// Current remaining range to remove
    iter: slice::Iter<'a, T>,
    vec: NonNull<ThinVec<T>>,
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

/// A splicing iterator for `ThinVec`.
///
/// This struct is created by the [`splice()`] method on [`ThinVec`]. See its
/// documentation for more.
///
/// [`splice()`]: struct.ThinVec.html#method.splice
/// [`ThinVec`]: struct.ThinVec.html
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
            let mut collected = self.replace_with.by_ref().collect::<ThinVec<I::Item>>().into_iter();
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

pub trait CloneIntoThinVec<T> {
    fn clone_into_thinvec(&self, target: &mut ThinVec<T>);
}

impl<T: Clone> CloneIntoThinVec<T> for [T] {
    fn clone_into_thinvec(&self, target: &mut ThinVec<T>) {
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

impl<T: Clone> Clone for ThinVec<T> {
    fn clone(&self) -> ThinVec<T> {
        <[T]>::to_thinvec(&**self)
    }

    fn clone_from(&mut self, other: &ThinVec<T>) {
        other.as_slice().clone_into_thinvec(self);
    }
}

/// An iterator produced by calling `drain_filter` on ThinVec.
#[derive(Debug)]
pub struct DrainFilter<'a, T: 'a, F>
    where F: FnMut(&mut T) -> bool,
{
    vec: &'a mut ThinVec<T>,
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

impl<T: PartialEq> ThinVec<T> {
    /// Removes consecutive repeated elements in the vector.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate thincollections;
    /// # use thincollections::thin_vec::ThinVec;
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
    /// # use thincollections::thin_vec::ThinVec;
    /// # fn main() {
    /// let mut vec = thinvec![1, 2, 3, 1];
    ///
    /// vec.remove_item(&1);
    ///
    /// assert_eq!(vec, thinvec![2, 3, 1]);
    /// # }
    /// ```
    pub fn remove_item(&mut self, item: &T) -> Option<T> {
        let pos = self.iter().position(|x| *x == *item)?;
        Some(self.remove(pos))
    }
}

impl<T> Borrow<[T]> for ThinVec<T> {
    fn borrow(&self) -> &[T] {
        &self[..]
    }
}

impl<T> BorrowMut<[T]> for ThinVec<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}