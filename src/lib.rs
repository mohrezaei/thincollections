// Copyright 2018 Mohammad Rezaei.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

//! # Thin Collections for Rust
//! Alternative implementations for vector, map and set that are faster/smaller for some use cases.
//!
//! `ThinVec` is a general vector replacement that only uses a single `usize`.
//! `std::collections::Vec` uses 3. This makes `ThinVec` a much better choice when it's used
//! inside another data structure, such as a vector of vectors or a map of vectors, etc.
//!
//! `ThinMap` is a specialized map replacement for small key values. It uses less memory than `HashMap`
//! if `mem::size_of::<(K, V)>() < 18`. It's also 2x to 5x faster (see the benchmarks). It's perfect
//! for all the primitives, or your own keys, but for custom keys, you must implement the `ThinSentinel`
//! trait.
//!
//! `ThinSet` uses `ThinMap` underneath, so it's great for elements up to 18 bytes.
//!
//! `V64` is a specialized vector replacement that uses a single 64bit value to represent itself.
//! It can store up to 7 bytes in that, and then uses heap memory. It's ideal for small vectors,
//! especially if those vectors are used inside other data structures.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! thincollections = "0.5.0"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
//! #[macro_use] extern crate thincollections;
//! # fn main() {
//! # }
//! ```
//!

pub mod thin_sentinel;
pub mod thin_map;
pub mod thin_set;
pub mod thin_v64;
pub mod thin_vec;
pub mod thin_hasher;
#[doc(hidden)]
pub mod cla_map;
#[doc(hidden)]
pub mod util;


/// Creates a [`V64`] containing the arguments.
///
/// `v64!` allows `V64`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`V64`] containing a given list of elements:
///
/// ```
/// # #[macro_use] extern crate thincollections;
/// # use thincollections::thin_v64::V64;
/// # fn main() {
/// let v: V64<i32> = v64![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// # }
/// ```
///
/// - Create a [`V64`] from a given element and size:
///
/// ```
/// # #[macro_use] extern crate thincollections;
/// # use thincollections::thin_v64::V64;
/// # fn main() {
/// let v: V64<u64> = v64![1; 3];
/// assert_eq!(3, v.len());
/// // assert_eq!(v, V64::from_buf([1, 1, 1]));
/// # }
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `v64![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
#[macro_export]
macro_rules! v64 {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::thin_v64::V64::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ v64!(@one $x))*;
        let mut vec = $crate::thin_v64::V64::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
}

/// Creates a [`ThinVec`] containing the arguments.
///
/// `thinvec!` allows `ThinVec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`ThinVec`] containing a given list of elements:
///
/// ```
/// # #[macro_use] extern crate thincollections;
/// # use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let v: ThinVec<i32> = thinvec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// # }
/// ```
///
/// - Create a [`ThinVec`] from a given element and size:
///
/// ```
/// # #[macro_use] extern crate thincollections;
/// # use thincollections::thin_vec::ThinVec;
/// # fn main() {
/// let v: ThinVec<u64> = thinvec![1; 3];
/// assert_eq!(3, v.len());
/// // assert_eq!(v, ThinVec::from_buf([1, 1, 1]));
/// # }
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `thinvec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
#[macro_export]
macro_rules! thinvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::thin_vec::ThinVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        let count = 0usize $(+ thinvec!(@one $x))*;
        let mut vec = $crate::thin_vec::ThinVec::with_capacity(count);
        $(vec.push($x);)*
        vec
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
