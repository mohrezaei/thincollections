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

//! # `ThinSet`: a fast, low memory set implementation for small elements.
//! `ThinSet` uses `ThinMap` underneath, so it inherits all the properties of
//! `ThinMap`.

use thin_sentinel::*;
use thin_hasher::*;
use thin_map::*;

use std::hash::BuildHasher;
use std::hash::Hash;
use std::fmt::{self};
use std::iter::FromIterator;
use std::iter::FusedIterator;
use std::ops::BitOr;
use std::ops::BitAnd;
use std::ops::BitXor;
use std::ops::Sub;
use std::iter::Chain;

/// A hash set implemented as a `ThinMap` where the value is `()`.
///
/// As with the [`ThinMap`] type, a `ThinSet` requires that the elements
/// implement the [`Eq`] and [`Hash`] and ['ThinSentinel'] traits.
/// [`Eq`] and [`Hash`] can frequently be achieved by
/// using `#[derive(PartialEq, Eq, Hash)]`. If you implement these yourself,
/// it is important that the following property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must be equal.
///
///
/// It is a logic error for an item to be modified in such a way that the
/// item's hash, as determined by the [`Hash`] trait, or its equality, as
/// determined by the [`Eq`] trait, changes while it is in the set. This is
/// normally only possible through [`Cell`], [`RefCell`], global state, I/O, or
/// unsafe code.
///
/// # Examples
///
/// ```
/// use thincollections::thin_set::ThinSet;
/// // Type inference lets us omit an explicit type signature (which
/// // would be `ThinSet<String>` in this example).
/// let mut nums = ThinSet::new();
///
/// // Add some nums.
/// nums.insert(17);
/// nums.insert(42);
/// nums.insert(225);
/// nums.insert(-5);
///
/// // Check for a specific one.
/// if !nums.contains(&44) {
///     println!("We have {} nums, but 44 ain't one.",
///              nums.len());
/// }
///
/// // Remove a num.
/// nums.remove(&225);
///
/// // Iterate over everything.
/// for n in &nums {
///     println!("{}", n);
/// }
/// ```
///
/// The easiest way to use `ThinSet` with a custom type is to derive
/// [`Eq`] and [`Hash`] and then implement [`ThinSentinel`]. We must also derive [`PartialEq`], this will in the
/// future be implied by [`Eq`].
///
/// ```
/// use thincollections::thin_set::ThinSet;
/// use thincollections::thin_sentinel::ThinSentinel;
/// 
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Color {
///     r: u8, g: u8, b: u8
/// }
/// 
/// impl ThinSentinel for Color {
///     fn thin_sentinel_zero() -> Self {
///         Color {r: 0, g: 0, b: 0}
///     }
///
///     fn thin_sentinel_one() -> Self {
///         Color {r : 0, g: 0, b: 1}
///     }
/// }
///
/// let mut colors = ThinSet::new();
///
/// colors.insert(Color { r: 255, g: 255, b: 255 });
/// colors.insert(Color { r: 255, g: 255, b: 0 });
/// colors.insert(Color { r: 255, g: 0, b: 255 });
/// colors.insert(Color { r: 0, g: 255, b: 255 });
///
/// // Use derived implementation to print the colors.
/// for x in &colors {
///     println!("{:?}", x);
/// }
/// ```
///
/// A `ThinSet` with fixed list of elements can be initialized from an array:
///
/// ```
/// use thincollections::thin_set::ThinSet;
///
/// fn main() {
///     let nums: ThinSet<i32> =
///         [ 2, 4, 6, 8 ].iter().cloned().collect();
///     // use the values stored in the set
/// }
/// ```
///
/// [`Cell`]: ../../std/cell/struct.Cell.html
/// [`Eq`]: ../../std/cmp/trait.Eq.html
/// [`Hash`]: ../../std/hash/trait.Hash.html
/// [`ThinMap`]: ../thin_map/struct.ThinMap.html
/// [`ThinSentinel`]: ../thin_sentinel/trait.ThinSentinel.html
/// [`PartialEq`]: ../../std/cmp/trait.PartialEq.html
/// [`RefCell`]: ../../std/cell/struct.RefCell.html
#[derive(Clone)]
pub struct ThinSet<T: ThinSentinel + Eq + Hash, S: BuildHasher = OneFieldHasherBuilder> {
    map: ThinMap<T, (), S>,
}

impl<T: Hash + Eq + ThinSentinel> ThinSet<T, OneFieldHasherBuilder> {
    /// Creates an empty `ThinSet`.
    ///
    /// The hash set is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let set: ThinSet<i32> = ThinSet::new();
    /// ```
    #[inline]
    pub fn new() -> ThinSet<T, OneFieldHasherBuilder> {
        ThinSet { map: ThinMap::new() }
    }

    /// Creates an empty `ThinSet` with the specified capacity.
    ///
    /// The hash set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash set will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let set: ThinSet<i32> = ThinSet::with_capacity(10);
    /// assert!(set.capacity() >= 10);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> ThinSet<T, OneFieldHasherBuilder> {
        ThinSet { map: ThinMap::with_capacity(capacity) }
    }
}

impl<T, S> ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    /// Creates a new empty hash set which will use the given hasher to hash
    /// keys.
    ///
    /// The hash set is also created with the default initial capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// use thincollections::thin_hasher::OneFieldHasherBuilder;
    ///
    /// let s = OneFieldHasherBuilder::new();
    /// let mut set = ThinSet::with_hasher(s);
    /// set.insert(2);
    /// ```
    #[inline]
    pub fn with_hasher(hasher: S) -> ThinSet<T, S> {
        ThinSet { map: ThinMap::with_hasher(hasher) }
    }

    /// Creates an empty `ThinSet` with with the specified capacity, using
    /// `hasher` to hash the keys.
    ///
    /// The hash set will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash set will not allocate.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow `ThinSet`s to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// use thincollections::thin_hasher::OneFieldHasherBuilder;
    ///
    /// let s = OneFieldHasherBuilder::new();
    /// let mut set = ThinSet::with_capacity_and_hasher(10, s);
    /// set.insert(1);
    /// ```
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> ThinSet<T, S> {
        ThinSet { map: ThinMap::with_capacity_and_hasher(capacity, hasher) }
    }

    /// Returns a reference to the set's [`BuildHasher`].
    ///
    /// [`BuildHasher`]: ../../std/hash/trait.BuildHasher.html
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// use thincollections::thin_hasher::OneFieldHasherBuilder;
    ///
    /// let hasher = OneFieldHasherBuilder::new();
    /// let set: ThinSet<i32> = ThinSet::with_hasher(hasher);
    /// let hasher: &OneFieldHasherBuilder = set.hasher();
    /// ```
    pub fn hasher(&self) -> &S {
        self.map.hasher()
    }

    /// Returns the number of elements the set can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let set: ThinSet<i32> = ThinSet::with_capacity(100);
    /// assert!(set.capacity() >= 100);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `ThinSet`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let mut set: ThinSet<i32> = ThinSet::new();
    /// set.reserve(10);
    /// assert!(set.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional)
    }

    /// Shrinks the capacity of the set as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut set = ThinSet::with_capacity(100);
    /// set.insert(1);
    /// set.insert(2);
    /// assert!(set.capacity() >= 100);
    /// set.shrink_to_fit();
    /// assert!(set.capacity() < 100);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit()
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is `&'a T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let mut set = ThinSet::new();
    /// set.insert(7);
    /// set.insert(22);
    ///
    /// // Will print in an arbitrary order.
    /// for x in set.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter { iter: self.map.keys() }
    }

    /// Visits the values representing the difference,
    /// i.e. the values that are in `self` but not in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let a: ThinSet<i32> = [1, 2, 3].iter().cloned().collect();
    /// let b: ThinSet<i32> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{}", x); // Print 1
    /// }
    ///
    /// let diff: ThinSet<i32> = a.difference(&b).cloned().collect();
    /// assert_eq!(diff, [1].iter().cloned().collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: ThinSet<_> = b.difference(&a).cloned().collect();
    /// let diff: ThinSet<_> = b.difference(&a).cloned().collect();
    /// assert_eq!(diff, [4].iter().cloned().collect());
    /// ```
    pub fn difference<'a>(&'a self, other: &'a ThinSet<T, S>) -> Difference<'a, T, S> {
        Difference {
            iter: self.iter(),
            other,
        }
    }

    /// Visits the values representing the symmetric difference,
    /// i.e. the values that are in `self` or in `other` but not in both.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let a: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: ThinSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 1, 4 in arbitrary order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let diff1: ThinSet<_> = a.symmetric_difference(&b).cloned().collect();
    /// let diff2: ThinSet<_> = b.symmetric_difference(&a).cloned().collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 4].iter().cloned().collect());
    /// ```
    pub fn symmetric_difference<'a>(&'a self,
                                    other: &'a ThinSet<T, S>)
                                    -> SymmetricDifference<'a, T, S> {
        SymmetricDifference { iter: self.difference(other).chain(other.difference(self)) }
    }

    /// Visits the values representing the intersection,
    /// i.e. the values that are both in `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let a: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: ThinSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 2, 3 in arbitrary order.
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let intersection: ThinSet<_> = a.intersection(&b).cloned().collect();
    /// assert_eq!(intersection, [2, 3].iter().cloned().collect());
    /// ```
    pub fn intersection<'a>(&'a self, other: &'a ThinSet<T, S>) -> Intersection<'a, T, S> {
        Intersection {
            iter: self.iter(),
            other,
        }
    }

    /// Visits the values representing the union,
    /// i.e. all the values in `self` or `other`, without duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let a: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let b: ThinSet<_> = [4, 2, 3, 4].iter().cloned().collect();
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order.
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// let union: ThinSet<_> = a.union(&b).cloned().collect();
    /// assert_eq!(union, [1, 2, 3, 4].iter().cloned().collect());
    /// ```
    pub fn union<'a>(&'a self, other: &'a ThinSet<T, S>) -> Union<'a, T, S> {
        Union { iter: self.iter().chain(other.difference(self)) }
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut v = ThinSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut v = ThinSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clears the set, returning all elements in an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut set: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert!(!set.is_empty());
    ///
    /// // print 1, 2, 3 in an arbitrary order
    /// for i in set.drain() {
    ///     println!("{}", i);
    /// }
    ///
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) -> Drain<T> {
        Drain { iter: self.map.drain() }
    }

    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut v = ThinSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear()
    }

    /// Returns `true` if the set contains a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let set: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    ///
    pub fn contains(&self, value: &T) -> bool
    {
        self.map.contains_key(value)
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let set: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    ///
    pub fn get(&self, value: &T) -> Option<&T>
    {
        let option = self.map.get_key_value(value);
        if option.is_none() { return None; }
        Some(option.unwrap().0)
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let a: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let mut b = ThinSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    pub fn is_disjoint(&self, other: &ThinSet<T, S>) -> bool {
        self.iter().all(|v| !other.contains(v))
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e. `other` contains at least all the values in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let sup: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// let mut set = ThinSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    pub fn is_subset(&self, other: &ThinSet<T, S>) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if the set is a superset of another,
    /// i.e. `self` contains at least all the values in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let sub: ThinSet<_> = [1, 2].iter().cloned().collect();
    /// let mut set = ThinSet::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    #[inline]
    pub fn is_superset(&self, other: &ThinSet<T, S>) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut set = ThinSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut set = ThinSet::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    #[inline]
    pub fn remove(&mut self, value: &T) -> bool
    {
        self.map.remove(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let mut set: ThinSet<_> = [1, 2, 3].iter().cloned().collect();
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    ///
    pub fn take(&mut self, value: &T) -> Option<T>
    {
        let option = self.map.remove_entry(value);
        if option.is_none() { return None; }
        Some(option.unwrap().0)
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let xs = [1,2,3,4,5,6];
    /// let mut set: ThinSet<i32> = xs.iter().cloned().collect();
    /// set.retain(|&k| k % 2 == 0);
    /// assert_eq!(set.len(), 3);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
        where F: FnMut(&T) -> bool
    {
        self.map.retain(|k, _| f(k));
    }
}

impl<T, S> PartialEq for ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    fn eq(&self, other: &ThinSet<T, S>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|key| other.contains(key))
    }
}

impl<T, S> Eq for ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{}

impl<T, S> fmt::Debug for ThinSet<T, S>
    where T: Eq + Hash + fmt::Debug + ThinSentinel,
          S: BuildHasher
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, S> FromIterator<T> for ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher + Default
{
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> ThinSet<T, S> {
        let mut set = ThinSet::with_hasher(Default::default());
        set.extend(iter);
        set
    }
}

impl<T, S> Extend<T> for ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
        self.map.extend(iter.into_iter().map(|k| (k, ())));
    }
}

impl<'a, T, S> Extend<&'a T> for ThinSet<T, S>
    where T: 'a + Eq + Hash + Copy + ThinSentinel,
          S: BuildHasher
{
    fn extend<I: IntoIterator<Item=&'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

impl<T, S> Default for ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher + Default
{
    /// Creates an empty `ThinSet<T, S>` with the `Default` value for the hasher.
    fn default() -> ThinSet<T, S> {
        ThinSet { map: ThinMap::default() }
    }
}

impl<'a, 'b, T, S> BitOr<&'b ThinSet<T, S>> for &'a ThinSet<T, S>
    where T: Eq + Hash + Clone + ThinSentinel,
          S: BuildHasher + Default
{
    type Output = ThinSet<T, S>;

    /// Returns the union of `self` and `rhs` as a new `ThinSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let a: ThinSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: ThinSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set = &a | &b;
    ///
    /// let mut i = 0;
    /// let expected = [1, 2, 3, 4, 5];
    /// for x in &set {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn bitor(self, rhs: &ThinSet<T, S>) -> ThinSet<T, S> {
        self.union(rhs).cloned().collect()
    }
}

impl<'a, 'b, T, S> BitAnd<&'b ThinSet<T, S>> for &'a ThinSet<T, S>
    where T: Eq + Hash + Clone + ThinSentinel,
          S: BuildHasher + Default
{
    type Output = ThinSet<T, S>;

    /// Returns the intersection of `self` and `rhs` as a new `ThinSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let a: ThinSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: ThinSet<_> = vec![2, 3, 4].into_iter().collect();
    ///
    /// let set = &a & &b;
    ///
    /// let mut i = 0;
    /// let expected = [2, 3];
    /// for x in &set {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn bitand(self, rhs: &ThinSet<T, S>) -> ThinSet<T, S> {
        self.intersection(rhs).cloned().collect()
    }
}

impl<'a, 'b, T, S> BitXor<&'b ThinSet<T, S>> for &'a ThinSet<T, S>
    where T: Eq + Hash + Clone + ThinSentinel,
          S: BuildHasher + Default
{
    type Output = ThinSet<T, S>;

    /// Returns the symmetric difference of `self` and `rhs` as a new `ThinSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let a: ThinSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: ThinSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set = &a ^ &b;
    ///
    /// let mut i = 0;
    /// let expected = [1, 2, 4, 5];
    /// for x in &set {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn bitxor(self, rhs: &ThinSet<T, S>) -> ThinSet<T, S> {
        self.symmetric_difference(rhs).cloned().collect()
    }
}

impl<'a, 'b, T, S> Sub<&'b ThinSet<T, S>> for &'a ThinSet<T, S>
    where T: Eq + Hash + Clone + ThinSentinel,
          S: BuildHasher + Default
{
    type Output = ThinSet<T, S>;

    /// Returns the difference of `self` and `rhs` as a new `ThinSet<T, S>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    ///
    /// let a: ThinSet<_> = vec![1, 2, 3].into_iter().collect();
    /// let b: ThinSet<_> = vec![3, 4, 5].into_iter().collect();
    ///
    /// let set = &a - &b;
    ///
    /// let mut i = 0;
    /// let expected = [1, 2];
    /// for x in &set {
    ///     assert!(expected.contains(x));
    ///     i += 1;
    /// }
    /// assert_eq!(i, expected.len());
    /// ```
    fn sub(self, rhs: &ThinSet<T, S>) -> ThinSet<T, S> {
        self.difference(rhs).cloned().collect()
    }
}

/// An iterator over the items of a `ThinSet`.
///
/// This `struct` is created by the [`iter`] method on [`ThinSet`].
/// See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`iter`]: struct.ThinSet.html#method.iter
#[derive(Clone)]
pub struct Iter<'a, K: 'a> {
    iter: Keys<'a, K, ()>,
}

/// An owning iterator over the items of a `ThinSet`.
///
/// This `struct` is created by the [`into_iter`] method on [`ThinSet`][`ThinSet`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`into_iter`]: struct.ThinSet.html#method.into_iter
pub struct IntoIter<K: Eq + ThinSentinel> {
    iter: super::thin_map::IntoIter<K, ()>,
}

/// A draining iterator over the items of a `ThinSet`.
///
/// This `struct` is created by the [`drain`] method on [`ThinSet`].
/// See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`drain`]: struct.ThinSet.html#method.drain
pub struct Drain<'a, K: 'a + ThinSentinel + Eq + Hash> {
    iter: super::thin_map::Drain<'a, K, ()>,
}

/// A lazy iterator producing elements in the intersection of `ThinSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`ThinSet`].
/// See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`intersection`]: struct.ThinSet.html#method.intersection
#[derive(Clone)]
pub struct Intersection<'a, T: 'a + ThinSentinel + Eq + Hash, S: 'a + BuildHasher> {
    // iterator of the first set
    iter: Iter<'a, T>,
    // the second set
    other: &'a ThinSet<T, S>,
}

/// A lazy iterator producing elements in the difference of `ThinSet`s.
///
/// This `struct` is created by the [`difference`] method on [`ThinSet`].
/// See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`difference`]: struct.ThinSet.html#method.difference
#[derive(Clone)]
pub struct Difference<'a, T: 'a + ThinSentinel + Eq + Hash, S: 'a + BuildHasher> {
    // iterator of the first set
    iter: Iter<'a, T>,
    // the second set
    other: &'a ThinSet<T, S>,
}

/// A lazy iterator producing elements in the symmetric difference of `ThinSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on
/// [`ThinSet`]. See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`symmetric_difference`]: struct.ThinSet.html#method.symmetric_difference
#[derive(Clone)]
pub struct SymmetricDifference<'a, T: 'a + ThinSentinel + Eq + Hash, S: 'a + BuildHasher> {
    iter: Chain<Difference<'a, T, S>, Difference<'a, T, S>>,
}

/// A lazy iterator producing elements in the union of `ThinSet`s.
///
/// This `struct` is created by the [`union`] method on [`ThinSet`].
/// See its documentation for more.
///
/// [`ThinSet`]: struct.ThinSet.html
/// [`union`]: struct.ThinSet.html#method.union
#[derive(Clone)]
pub struct Union<'a, T: 'a + ThinSentinel + Eq + Hash, S: 'a + BuildHasher + BuildHasher> {
    iter: Chain<Iter<'a, T>, Difference<'a, T, S>>,
}

impl<'a, T, S> IntoIterator for &'a ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<T, S> IntoIterator for ThinSet<T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out
    /// of the set in arbitrary order. The set cannot be used after calling
    /// this.
    ///
    /// # Examples
    ///
    /// ```
    /// use thincollections::thin_set::ThinSet;
    /// let mut set = ThinSet::new();
    /// set.insert(1_000_000);
    /// set.insert(200_000);
    ///
    /// // Not possible to collect to a Vec<u32> with a regular `.iter()`.
    /// let v: Vec<u32> = set.into_iter().collect();
    ///
    /// // Will print in an arbitrary order.
    /// for x in &v {
    ///     println!("{}", x);
    /// }
    /// ```
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { iter: self.map.into_iter() }
    }
}

//impl<'a, K: ThinSentinel + Eq + Hash> Clone for Iter<'a, K> {
//    fn clone(&self) -> Iter<'a, K> {
//        Iter { iter: self.iter.clone() }
//    }
//}

impl<'a, K: 'a + ThinSentinel + Eq> Iterator for Iter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: 'a + ThinSentinel + Eq> ExactSizeIterator for Iter<'a, K> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: 'a + ThinSentinel + Eq> FusedIterator for Iter<'a, K> {}

//impl<'a, K: 'a + fmt::Debug + ThinSentinel + Eq> fmt::Debug for Iter<'a, K> {
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.debug_list().entries(self.clone()).finish()
//    }
//}

impl<K: ThinSentinel + Eq> Iterator for IntoIter<K> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        self.iter.next().map(|(k, _)| k)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K: ThinSentinel + Eq> ExactSizeIterator for IntoIter<K> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: ThinSentinel + Eq> FusedIterator for IntoIter<K> {}
//
//impl<K: fmt::Debug + ThinSentinel + Eq + Hash> fmt::Debug for IntoIter<K> {
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        let entries_iter = self.iter
//            .iter()
//            .map(|(k, _)| k);
//        f.debug_list().entries(entries_iter).finish()
//    }
//}

impl<'a, K: ThinSentinel + Eq + Hash> Iterator for Drain<'a, K> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        self.iter.next().map(|(k, _)| k)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: ThinSentinel + Eq + Hash> ExactSizeIterator for Drain<'a, K> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: ThinSentinel + Eq + Hash> FusedIterator for Drain<'a, K> {}

//impl<'a, K: fmt::Debug + ThinSentinel + Eq + Hash> fmt::Debug for Drain<'a, K> {
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        let entries_iter = self.iter
//            .iter()
//            .map(|(k, _)| k);
//        f.debug_list().entries(entries_iter).finish()
//    }
//}
//
//impl<'a, T: ThinSentinel + Eq + Hash, S: BuildHasher> Clone for Intersection<'a, T, S> {
//    fn clone(&self) -> Intersection<'a, T, S> {
//        Intersection { iter: self.iter.clone(), ..*self }
//    }
//}

impl<'a, T, S> Iterator for Intersection<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            let elt = self.iter.next()?;
            if self.other.contains(elt) {
                return Some(elt);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

//impl<'a, T, S> fmt::Debug for Intersection<'a, T, S>
//    where T: fmt::Debug + Eq + Hash + ThinSentinel,
//          S: BuildHasher
//{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.debug_list().entries(self.clone()).finish()
//    }
//}
//
impl<'a, T, S> FusedIterator for Intersection<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{}

//impl<'a, T: ThinSentinel + Eq + Hash, S: BuildHasher> Clone for Difference<'a, T, S> {
//    fn clone(&self) -> Difference<'a, T, S> {
//        Difference { iter: self.iter.clone(), ..*self }
//    }
//}
//
impl<'a, T, S> Iterator for Difference<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            let elt = self.iter.next()?;
            if !self.other.contains(elt) {
                return Some(elt);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

impl<'a, T, S> FusedIterator for Difference<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{}

//impl<'a, T, S> fmt::Debug for Difference<'a, T, S>
//    where T: fmt::Debug + Eq + Hash + ThinSentinel,
//          S: BuildHasher
//{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.debug_list().entries(self.clone()).finish()
//    }
//}
//
//impl<'a, T: ThinSentinel + Eq + Hash, S: BuildHasher> Clone for SymmetricDifference<'a, T, S> {
//    fn clone(&self) -> SymmetricDifference<'a, T, S> {
//        SymmetricDifference { iter: self.iter.clone() }
//    }
//}

impl<'a, T, S> Iterator for SymmetricDifference<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T, S> FusedIterator for SymmetricDifference<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{}

//impl<'a, T, S> fmt::Debug for SymmetricDifference<'a, T, S>
//    where T: fmt::Debug + Eq + Hash + ThinSentinel,
//          S: BuildHasher
//{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.debug_list().entries(self.clone()).finish()
//    }
//}
//
impl<'a, T, S> FusedIterator for Union<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{}

//impl<'a, T, S> fmt::Debug for Union<'a, T, S>
//    where T: fmt::Debug + Eq + Hash + ThinSentinel,
//          S: BuildHasher
//{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        f.debug_list().entries(self.clone()).finish()
//    }
//}
//
impl<'a, T, S> Iterator for Union<'a, T, S>
    where T: Eq + Hash + ThinSentinel,
          S: BuildHasher
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[cfg(test)]
mod test_set {
    use super::ThinSet;
    use super::super::thin_hasher::OneFieldHasherBuilder;

    #[test]
    fn test_zero_capacities() {
        type HS = ThinSet<i32>;

        let s = HS::new();
        assert_eq!(s.capacity(), 0);

        let s = HS::default();
        assert_eq!(s.capacity(), 0);

        let s = HS::with_hasher(OneFieldHasherBuilder::new());
        assert_eq!(s.capacity(), 0);

        let s = HS::with_capacity(0);
        assert_eq!(s.capacity(), 0);

        let s = HS::with_capacity_and_hasher(0, OneFieldHasherBuilder::new());
        assert_eq!(s.capacity(), 0);

        let mut s = HS::new();
        s.insert(1);
        s.insert(2);
        s.remove(&1);
        s.remove(&2);
        s.shrink_to_fit();
        assert_eq!(s.capacity(), 0);

        let mut s = HS::new();
        s.reserve(0);
        assert_eq!(s.capacity(), 0);
    }

    #[test]
    fn test_disjoint() {
        let mut xs = ThinSet::new();
        let mut ys = ThinSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(ys.insert(-11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = ThinSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = ThinSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_iterate() {
        let mut a = ThinSet::new();
        for i in 0..32 {
            assert!(a.insert(i));
        }
        let mut observed: u32 = 0;
        for k in &a {
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_intersection() {
        let mut a = ThinSet::new();
        let mut b = ThinSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));
        assert!(a.insert(-5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(-9));
        assert!(b.insert(-42));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for x in a.intersection(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = ThinSet::new();
        let mut b = ThinSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for x in a.difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = ThinSet::new();
        let mut b = ThinSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(-2));
        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [-2, 1, 5, 11, 14, 22];
        for x in a.symmetric_difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = ThinSet::new();
        let mut b = ThinSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(-2));
        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24];
        for x in a.union(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_from_iter() {
        let xs = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: ThinSet<_> = xs.iter().cloned().collect();

        for x in &xs {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_move_iter() {
        let hs = {
            let mut hs = ThinSet::new();

            hs.insert('a');
            hs.insert('b');

            hs
        };

        let v = hs.into_iter().collect::<Vec<char>>();
        assert!(v == ['a', 'b'] || v == ['b', 'a']);
    }

    #[test]
    fn test_eq() {
        // These constants once happened to expose a bug in insert().
        // I'm keeping them around to prevent a regression.
        let mut s1 = ThinSet::new();

        s1.insert(1);
        s1.insert(2);
        s1.insert(3);

        let mut s2 = ThinSet::new();

        s2.insert(1);
        s2.insert(2);

        assert!(s1 != s2);

        s2.insert(3);

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_show() {
        let mut set = ThinSet::new();
        let empty = ThinSet::<i32>::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{:?}", set);

        assert!(set_str == "{1, 2}" || set_str == "{2, 1}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_trivial_drain() {
        let mut s = ThinSet::<i32>::new();
        for _ in s.drain() {}
        assert!(s.is_empty());
        drop(s);

        let mut s = ThinSet::<i32>::new();
        drop(s.drain());
        assert!(s.is_empty());
    }

    #[test]
    fn test_drain() {
        let mut s: ThinSet<_> = (1..100).collect();

        // try this a bunch of times to make sure we don't screw up internal state.
        for _ in 0..20 {
            assert_eq!(s.len(), 99);

            {
                let mut last_i = 0;
                let mut d = s.drain();
                for (i, x) in d.by_ref().take(50).enumerate() {
                    last_i = i;
                    assert!(x != 0);
                }
                assert_eq!(last_i, 49);
            }

            for _ in &s {
                panic!("s should be empty!");
            }

            // reset to try again.
            s.extend(1..100);
        }
    }

    #[test]
    fn test_extend_ref() {
        let mut a = ThinSet::new();
        a.insert(1);

        a.extend(&[2, 3, 4]);

        assert_eq!(a.len(), 4);
        assert!(a.contains(&1));
        assert!(a.contains(&2));
        assert!(a.contains(&3));
        assert!(a.contains(&4));

        let mut b = ThinSet::new();
        b.insert(5);
        b.insert(6);

        a.extend(&b);

        assert_eq!(a.len(), 6);
        assert!(a.contains(&1));
        assert!(a.contains(&2));
        assert!(a.contains(&3));
        assert!(a.contains(&4));
        assert!(a.contains(&5));
        assert!(a.contains(&6));
    }

    #[test]
    fn test_retain() {
        let xs = [1, 2, 3, 4, 5, 6];
        let mut set: ThinSet<i32> = xs.iter().cloned().collect();
        set.retain(|&k| k % 2 == 0);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&2));
        assert!(set.contains(&4));
        assert!(set.contains(&6));
    }
}
