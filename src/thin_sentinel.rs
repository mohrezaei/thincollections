// Copyright 2018 Mohammad Rezaei.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

//! A special trait required for `ThinMap` and `ThinSet`

/// `ThinMap` and `ThinSet` require two sepcial values to denote either an empty
/// or a removed element. This does NOT preclude these elements from being stored
/// in the map/set.
///
/// `ThinSentinel` is already implemented for all the primitives, so if you're just
/// using those, there is nothing to do.
///
/// It's generally difficult to implement `ThinSentinel` if the element requires a
/// `Drop` implementation. `ThinMap`/`ThinSet` have not been tested with such keys/elements.
/// The real requirement for this is that `Drop` should be no-op for the sentinel values.
///
/// Here is an example of a custom implementation:
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
/// colors.insert(Color { r: 0, g: 0, b: 0 }); // no trouble storing a sentinel!
///
/// // Use derived implementation to print the colors.
/// for x in &colors {
///     println!("{:?}", x);
/// }
/// ```
///
///
pub trait ThinSentinel {
    #[inline(always)]
    fn thin_sentinel_zero() -> Self;
    #[inline(always)]
    fn thin_sentinel_one() -> Self;
}

macro_rules! impl_sentinel_for_primitive {
    ($T:ty) => (
        impl ThinSentinel for $T {
            #[inline(always)]
            fn thin_sentinel_zero() -> Self {
                0 as $T
            }
            #[inline(always)]
            fn thin_sentinel_one() -> Self {
                1 as $T
            }
        }
    )
}

use std::any::TypeId;
//
//pub trait ThinSentinel {
//    const ZERO: Self;
//    const ONE: Self;
//}
//
//macro_rules! impl_sentinel_for_primitive {
//    ($T:ty) => (
//        impl ThinSentinel for $T {
//            const ZERO: $T = 0 as $T;
//            const ONE: $T = 1 as $T;
//        }
//    )
//}

impl_sentinel_for_primitive!(u8);
impl_sentinel_for_primitive!(u16);
impl_sentinel_for_primitive!(u32);
impl_sentinel_for_primitive!(u64);
impl_sentinel_for_primitive!(u128);

impl_sentinel_for_primitive!(i8);
impl_sentinel_for_primitive!(i16);
impl_sentinel_for_primitive!(i32);
impl_sentinel_for_primitive!(i64);
impl_sentinel_for_primitive!(i128);

impl_sentinel_for_primitive!(f32);
impl_sentinel_for_primitive!(f64);

impl_sentinel_for_primitive!(usize);
impl_sentinel_for_primitive!(isize);

impl_sentinel_for_primitive!(char);

/* this is a really bad idea:
impl<V> ThinSentinel for Box<V> where V: ThinSentinel {
    fn thin_sentinel_zero() -> Self {
        Box::new(V::thin_sentinel_zero()) //allocates!!!!
    }

    fn thin_sentinel_one() -> Self {
        Box::new(V::thin_sentinel_one())
    }
}
*/

impl<T: ThinSentinel, U: ThinSentinel> ThinSentinel for (T, U) {
    fn thin_sentinel_zero() -> Self {
        (T::thin_sentinel_zero(), U::thin_sentinel_zero())
    }

    fn thin_sentinel_one() -> Self {
        (T::thin_sentinel_zero(), U::thin_sentinel_one())
    }
}

impl<T: ThinSentinel, U: ThinSentinel, V: ThinSentinel> ThinSentinel for (T, U, V) {
    fn thin_sentinel_zero() -> Self {
        (T::thin_sentinel_zero(), U::thin_sentinel_zero(), V::thin_sentinel_zero())
    }

    fn thin_sentinel_one() -> Self {
        (T::thin_sentinel_zero(), U::thin_sentinel_zero(), V::thin_sentinel_one())
    }
}

//impl<T: ThinSentinel, U: ThinSentinel, V: ThinSentinel, W: ThinSentinel> ThinSentinel for (T, U, V, W) {
//    const ZERO: Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO);
//    const ONE:  Self = (T::ZERO,U::ZERO,V::ZERO,W::ONE);
//}
//
//impl<T: ThinSentinel, U: ThinSentinel, V: ThinSentinel, W: ThinSentinel, X: ThinSentinel> ThinSentinel for (T, U, V, W, X) {
//    const ZERO: Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO,X::ZERO);
//    const ONE:  Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO,X::ONE);
//}
//
//impl<T: ThinSentinel, U: ThinSentinel, V: ThinSentinel, W: ThinSentinel, X: ThinSentinel, Y: ThinSentinel> ThinSentinel for (T, U, V, W, X, Y) {
//    const ZERO: Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO,X::ZERO,Y::ZERO);
//    const ONE:  Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO,X::ZERO,Y::ONE);
//}
//
//impl<T: ThinSentinel, U: ThinSentinel, V: ThinSentinel, W: ThinSentinel, X: ThinSentinel, Y: ThinSentinel, Z: ThinSentinel> ThinSentinel for (T, U, V, W, X, Y, Z) {
//    const ZERO: Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO,X::ZERO,Y::ZERO,Z::ZERO);
//    const ONE:  Self = (T::ZERO,U::ZERO,V::ZERO,W::ZERO,X::ZERO,Y::ZERO,Z::ONE);
//}

pub enum ThinSentinelEnum<T> {
    ZERO,
    ONE,
    VALUE(T),
}

impl<T> ThinSentinel for ThinSentinelEnum<T> {
    fn thin_sentinel_zero() -> Self {
        ThinSentinelEnum::ZERO
    }

    fn thin_sentinel_one() -> Self {
        ThinSentinelEnum::ONE
    }
}

struct TypeIdZero {}

struct TypeIdOne {}

impl ThinSentinel for TypeId {
    fn thin_sentinel_zero() -> Self {
        TypeId::of::<TypeIdZero>()
    }

    fn thin_sentinel_one() -> Self {
        TypeId::of::<TypeIdOne>()
    }
}
