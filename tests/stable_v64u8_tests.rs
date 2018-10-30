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

#[macro_use]
extern crate thincollections;

use thincollections::thin_v64::V64;
use thincollections::thin_v64::ToV64;

#[test]
fn test_reserve() {
    let mut v: V64<u8> = V64::new();
    assert_eq!(v.capacity(), 7);

    v.reserve(2);
    assert!(v.capacity() >= 2);

    for i in 0..16 {
        v.push(i);
    }

    assert!(v.capacity() >= 16);
    v.reserve(16);
    assert!(v.capacity() >= 32);

    v.push(16);

    v.reserve(16);
    assert!(v.capacity() >= 33)
}

#[test]
fn test_extend() {
    let mut v: V64<u8> = V64::new();
    let mut w: V64<u8> = V64::new();

    v.extend(w.clone());
    assert_eq!(v, &[]);

    v.extend(0..3);
    for i in 0..3 {
        w.push(i)
    }

    assert_eq!(v, w);

    v.extend(3..10);
    for i in 3..10 {
        w.push(i)
    }

    assert_eq!(v, w);

    v.extend(w.clone()); // specializes to `append`
    assert!(v.iter().eq(w.iter().chain(w.iter())));
}

#[test]
fn test_extend_ref() {
    let mut v: V64<u8> = v64![1, 2];
    v.extend(&[3, 4, 5]);

    assert_eq!(v.len(), 5);
    assert_eq!(v, [1, 2, 3, 4, 5]);

    let w = v64![6, 7];
    v.extend(&w);

    assert_eq!(v.len(), 7);
    assert_eq!(v, [1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_slice_from_mut() {
    let mut values: V64<u8> = v64![1, 2, 3, 4, 5];
    {
        let slice = &mut values[2..];
        assert!(slice == [3, 4, 5]);
        for p in slice {
            *p += 2;
        }
    }

    assert!(values == [1, 2, 5, 6, 7]);
}

#[test]
fn test_slice_to_mut() {
    let mut values: V64<u8> = v64![1, 2, 3, 4, 5];
    {
        let slice = &mut values[..2];
        assert!(slice == [1, 2]);
        for p in slice {
            *p += 1;
        }
    }

    assert!(values == [2, 3, 3, 4, 5]);
}

#[test]
fn test_split_at_mut() {
    let mut values: V64<u8> = v64![1, 2, 3, 4, 5];
    {
        let (left, right) = values.split_at_mut(2);
        {
            let left: &[_] = left;
            assert!(&left[..left.len()] == &[1, 2]);
        }
        for p in left {
            *p += 1;
        }

        {
            let right: &[_] = right;
            assert!(&right[..right.len()] == &[3, 4, 5]);
        }
        for p in right {
            *p += 2;
        }
    }

    assert_eq!(values, [2, 3, 5, 6, 7]);
}

#[test]
fn test_clone() {
    let v: V64<u8> = v64![];
    let w: V64<u8> = v64![1, 2, 3];

    assert_eq!(v, v.clone());

    let z = w.clone();
    assert_eq!(w, z);
    // they should be disjoint in memory.
    assert!(w.as_ptr() != z.as_ptr())
}

#[test]
fn test_retain() {
    let mut v64: V64<u8> = v64![1, 2, 3, 4];
    v64.retain(|&x| x % 2 == 0);
    assert_eq!(v64, [2, 4]);
}

#[test]
fn test_dedup() {
    fn case(a: V64<u8>, b: V64<u8>) {
        let mut v = a;
        v.dedup();
        assert_eq!(v, b);
    }
    case(v64![], v64![]);
    case(v64![1], v64![1]);
    case(v64![1, 1], v64![1]);
    case(v64![1, 2, 3], v64![1, 2, 3]);
    case(v64![1, 1, 2, 3], v64![1, 2, 3]);
    case(v64![1, 2, 2, 3], v64![1, 2, 3]);
    case(v64![1, 2, 3, 3], v64![1, 2, 3]);
    case(v64![1, 1, 2, 2, 2, 3, 3], v64![1, 2, 3]);
}

#[test]
fn test_dedup_by_key() {
    fn case(a: V64<u8>, b: V64<u8>) {
        let mut v = a;
        v.dedup_by_key(|i| *i / 10);
        assert_eq!(v, b);
    }
    case(v64![], v64![]);
    case(v64![10], v64![10]);
    case(v64![10, 11], v64![10]);
    case(v64![10, 20, 30], v64![10, 20, 30]);
    case(v64![10, 11, 20, 30], v64![10, 20, 30]);
    case(v64![10, 20, 21, 30], v64![10, 20, 30]);
    case(v64![10, 20, 30, 31], v64![10, 20, 30]);
    case(v64![10, 11, 20, 21, 22, 30, 31], v64![10, 20, 30]);
}

#[test]
fn test_partition() {
    assert_eq!(v64![].into_iter().partition(|x: &i32| *x < 3),
               (v64![], v64![]));
    assert_eq!(v64![1u8, 2u8, 3u8].into_iter().partition(|x| *x < 4),
               (v64![1u8, 2u8, 3u8], v64![]));
    assert_eq!(v64![1u8, 2u8, 3u8].into_iter().partition(|x| *x < 2),
               (v64![1u8], v64![2u8, 3u8]));
    assert_eq!(v64![1i8, 2i8, 3i8].into_iter().partition(|x| *x < 0),
               (v64![], v64![1i8, 2i8, 3i8]));
}

#[test]
fn test_zip_unzip() {
    let z1: V64<(u8, u8)> = v64![(1, 4), (2, 5)];

    let (left, right): (V64<_>, V64<_>) = z1.iter().cloned().unzip();

    assert_eq!((1, 4), (left[0], right[0]));
    assert_eq!((2, 5), (left[1], right[1]));
}

#[test]
fn test_v64_truncate_drop() {
    static mut DROPS: u32 = 0;
    struct Elem(u8);
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let mut v = v64![Elem(1), Elem(2), Elem(3), Elem(4), Elem(5)];
    assert_eq!(unsafe { DROPS }, 0);
    v.truncate(3);
    assert_eq!(unsafe { DROPS }, 2);
    v.truncate(0);
    assert_eq!(unsafe { DROPS }, 5);
}

#[test]
fn test_index() {
    let v64: V64<u8> = v64![1, 2, 3];
    assert!(v64[1] == 2);
}

#[test]
fn test_move_items() {
    let v64: V64<u8> = v64![1, 2, 3];
    let mut v642 = v64![];
    for i in v64 {
        v642.push(i);
    }
    assert_eq!(v642, [1, 2, 3]);
}

#[test]
fn test_move_items_reverse() {
    let v64: V64<u8> = v64![1, 2, 3];
    let mut v642 = v64![];
    for i in v64.into_iter().rev() {
        v642.push(i);
    }
    assert_eq!(v642, [3, 2, 1]);
}

#[test]
fn test_drain_items() {
    let mut v64: V64<u8> = v64![1, 2, 3];
    let mut v642: V64<u8> = v64![];
    for i in v64.drain(..) {
        v642.push(i);
    }
    assert_eq!(v64, []);
    assert_eq!(v642, [1, 2, 3]);
}

#[test]
fn test_drain_items_reverse() {
    let mut v64: V64<u8> = v64![1, 2, 3];
    let mut v642: V64<u8> = v64![];
    for i in v64.drain(..).rev() {
        v642.push(i);
    }
    assert_eq!(v64, []);
    assert_eq!(v642, [3, 2, 1]);
}

#[test]
#[should_panic]
fn test_drain_out_of_bounds() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    v.drain(5..6);
}

#[test]
fn test_drain_range() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    for _ in v.drain(4..) {}
    assert_eq!(v, &[1, 2, 3, 4]);

    let mut v: V64<_> = (1..6).map(|x| x.to_string()).collect();
    for _ in v.drain(1..4) {}
    assert_eq!(v, &[1.to_string(), 5.to_string()]);

    let mut v: V64<_> = (1..6).map(|x| x.to_string()).collect();
    for _ in v.drain(1..4).rev() {}
    assert_eq!(v, &[1.to_string(), 5.to_string()]);

    let mut v: V64<_> = v64![(); 5];
    for _ in v.drain(1..4).rev() {}
    assert_eq!(v, &[(), ()]);
}

//#[test]
//fn test_drain_max_v64_size() {
//    let mut v = V64::<()>::with_capacity(<usize>::max_value());
//    unsafe { v.set_len(<usize>::max_value()); }
//    for _ in v.drain(<usize>::max_value() - 1..) {
//    }
//    assert_eq!(v.len(), <usize>::max_value() - 1);
//
//    let mut v = V64::<()>::with_capacity(<usize>::max_value());
//    unsafe { v.set_len(<usize>::max_value()); }
//    for _ in v.drain(<usize>::max_value() - 1..=<usize>::max_value() - 1) {
//    }
//    assert_eq!(v.len(),<usize>::max_value() - 1);
//}

#[test]
#[should_panic]
fn test_drain_inclusive_out_of_bounds() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    v.drain(5..=5);
}

#[test]
fn test_splice() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    let a: [u8; 3] = [10, 11, 12];
    let _t1: V64<_> = v.splice(2..4, a.iter().cloned()).collect();
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    let _t2: V64<_> = v.splice(1..3, Some(20)).collect();
    assert_eq!(v, &[1, 20, 11, 12, 5]);
}

#[test]
fn test_splice_inclusive_range() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    let a: [u8; 3] = [10, 11, 12];
    let t1: V64<_> = v.splice(2..=3, a.iter().cloned()).collect();
    assert_eq!(v, &[1, 2, 10, 11, 12, 5]);
    assert_eq!(t1, &[3, 4]);
    let t2: V64<_> = v.splice(1..=2, Some(20)).collect();
    assert_eq!(v, &[1, 20, 11, 12, 5]);
    assert_eq!(t2, &[2, 10]);
}

#[test]
#[should_panic]
fn test_splice_out_of_bounds() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    let a: [u8; 3] = [10, 11, 12];
    v.splice(5..6, a.iter().cloned());
}

#[test]
#[should_panic]
fn test_splice_inclusive_out_of_bounds() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    let a: [u8; 3] = [10, 11, 12];
    v.splice(5..=5, a.iter().cloned());
}

#[test]
fn test_splice_unbounded() {
    let mut v64: V64<u8> = v64![1, 2, 3, 4, 5];
    let t: V64<_> = v64.splice(.., None).collect();
    assert_eq!(v64, &[]);
    assert_eq!(t, &[1, 2, 3, 4, 5]);
}

#[test]
fn test_splice_forget() {
    let mut v: V64<u8> = v64![1, 2, 3, 4, 5];
    let a: [u8; 3] = [10, 11, 12];
    ::std::mem::forget(v.splice(2..4, a.iter().cloned()));
    assert_eq!(v, &[1, 2]);
}

#[test]
fn test_into_boxed_slice() {
    let xs: V64<u8> = v64![1, 2, 3];
    let ys = xs.into_boxed_slice();
    assert_eq!(&*ys, [1, 2, 3]);
}

#[test]
fn test_append() {
    let mut v64: V64<u8> = v64![1, 2, 3];
    let mut v642: V64<u8> = v64![4, 5, 6];
    v64.append(&mut v642);
    assert_eq!(v64, [1, 2, 3, 4, 5, 6]);
    assert_eq!(v642, []);
}

#[test]
fn test_split_off() {
    let mut v64: V64<u8> = v64![1, 2, 3, 4, 5, 6];
    let v642 = v64.split_off(4);
    assert_eq!(v64, [1, 2, 3, 4]);
    assert_eq!(v642, [5, 6]);
}

#[test]
fn test_into_iter_as_slice() {
    let v64: V64<u8> = v64![1, 2, 3];
    let mut into_iter = v64.into_iter();
    assert_eq!(into_iter.as_slice(), &[1, 2, 3]);
    let _ = into_iter.next().unwrap();
    assert_eq!(into_iter.as_slice(), &[2, 3]);
    let _ = into_iter.next().unwrap();
    let _ = into_iter.next().unwrap();
    assert_eq!(into_iter.as_slice(), &[]);
}

#[test]
fn test_into_iter_as_mut_slice() {
    let v64: V64<u8> = v64![1, 2, 3];
    let mut into_iter = v64.into_iter();
    assert_eq!(into_iter.as_slice(), &[1, 2, 3]);
    into_iter.as_mut_slice()[0] = 17;
    into_iter.as_mut_slice()[1] = 18;
    assert_eq!(into_iter.next().unwrap(), 17);
    assert_eq!(into_iter.as_slice(), &[18, 3]);
}

#[test]
fn test_into_iter_debug() {
    let v64: V64<u8> = v64![1, 2, 3];
    let into_iter = v64.into_iter();
    let debug = format!("{:?}", into_iter);
    assert_eq!(debug, "IntoIter([1, 2, 3])");
}

#[test]
fn test_into_iter_count() {
    assert_eq!(v64![1, 2, 3].into_iter().count(), 3);
}

#[test]
fn test_into_iter() {
    let mut it = v64![1u8, 2u8, 3u8].into_iter();
    assert_eq!(Some(1u8), it.next());
    assert_eq!(Some(2u8), it.next());
    assert_eq!(Some(3u8), it.next());
    assert_eq!(None, it.next());
    assert_eq!(None, it.next());

    let it = v64![1u8, 2u8, 3u8].into_iter();
    assert_eq!(it.as_slice(), &[1, 2, 3]);

    let it = v64![1u8, 2u8, 3u8].into_iter();
    assert_eq!(it.as_slice().to_v64(), v64![1u8, 2u8, 3u8]);
}

#[test]
fn test_into_iter_clone() {
    fn iter_equal<I: Iterator<Item=u8>>(it: I, slice: &[u8]) {
        let v: V64<u8> = it.collect();
        assert_eq!(&v[..], slice);
    }
    let mut it = v64![1u8, 2u8, 3u8].into_iter();
    iter_equal(it.clone(), &[1, 2, 3]);
    assert_eq!(it.next(), Some(1));
    let mut it = it.rev();
    iter_equal(it.clone(), &[3, 2]);
    assert_eq!(it.next(), Some(3));
    iter_equal(it.clone(), &[2]);
    assert_eq!(it.next(), Some(2));
    iter_equal(it.clone(), &[]);
    assert_eq!(it.next(), None);
}

//#[test]
//fn from_into_inner() {
//    let v64: V64<u8> = v64![1, 2, 3];
//    let ptr = v64.as_ptr();
//    let v64 = v64.into_iter().collect::<V64<_>>();
//    assert_eq!(v64, [1, 2, 3]);
//    assert_eq!(v64.as_ptr(), ptr);
//
//    let ptr = &v64[1] as *const _;
//    let mut it = v64.into_iter();
//    it.next().unwrap();
//    let v64 = it.collect::<V64<_>>();
//    assert_eq!(v64, [2, 3]);
//    assert!(ptr != v64.as_ptr());
//}

#[test]
fn drain_filter_empty() {
    let mut v64: V64<u8> = v64![];

    {
        let mut iter = v64.drain_filter(|_| true);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }
    assert_eq!(v64.len(), 0);
    assert_eq!(v64, v64![]);
}

#[test]
fn drain_filter_false() {
    let mut v64: V64<u8> = v64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let initial_len = v64.len();
    let mut count = 0;
    {
        let mut iter = v64.drain_filter(|_| false);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        for _ in iter.by_ref() {
            count += 1;
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, 0);
    assert_eq!(v64.len(), initial_len);
    assert_eq!(v64, v64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn drain_filter_true() {
    let mut v64: V64<u8> = v64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let initial_len = v64.len();
    let mut count = 0;
    {
        let mut iter = v64.drain_filter(|_| true);
        assert_eq!(iter.size_hint(), (0, Some(initial_len)));
        while let Some(_) = iter.next() {
            count += 1;
            assert_eq!(iter.size_hint(), (0, Some(initial_len - count)));
        }
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    assert_eq!(count, initial_len);
    assert_eq!(v64.len(), 0);
    assert_eq!(v64, v64![]);
}

#[test]
fn drain_filter_complex() {
    {   //                [+xxx++++++xxxxx++++x+x++]
        let mut v64: V64<u8> = v64![1,
                           2, 4, 6,
                           7, 9, 11, 13, 15, 17,
                           18, 20, 22, 24, 26,
                           27, 29, 31, 33,
                           34,
                           35,
                           36,
                           37, 39];

        let removed = v64.drain_filter(|x| *x % 2 == 0).collect::<V64<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, v64![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(v64.len(), 14);
        assert_eq!(v64, v64![1, 7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39]);
    }

    {   //                [xxx++++++xxxxx++++x+x++]
        let mut v64: V64<u8> = v64![2, 4, 6,
                           7, 9, 11, 13, 15, 17,
                           18, 20, 22, 24, 26,
                           27, 29, 31, 33,
                           34,
                           35,
                           36,
                           37, 39];

        let removed = v64.drain_filter(|x| *x % 2 == 0).collect::<V64<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, v64![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(v64.len(), 13);
        assert_eq!(v64, v64![7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35, 37, 39]);
    }

    {   //                [xxx++++++xxxxx++++x+x]
        let mut v64: V64<u8> = v64![2, 4, 6,
                           7, 9, 11, 13, 15, 17,
                           18, 20, 22, 24, 26,
                           27, 29, 31, 33,
                           34,
                           35,
                           36];

        let removed = v64.drain_filter(|x| *x % 2 == 0).collect::<V64<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, v64![2, 4, 6, 18, 20, 22, 24, 26, 34, 36]);

        assert_eq!(v64.len(), 11);
        assert_eq!(v64, v64![7, 9, 11, 13, 15, 17, 27, 29, 31, 33, 35]);
    }

    {   //                [xxxxxxxxxx+++++++++++]
        let mut v64: V64<u8> = v64![2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                           1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

        let removed = v64.drain_filter(|x| *x % 2 == 0).collect::<V64<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, v64![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(v64.len(), 10);
        assert_eq!(v64, v64![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }

    {   //                [+++++++++++xxxxxxxxxx]
        let mut v64: V64<u8> = v64![1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
                           2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

        let removed = v64.drain_filter(|x| *x % 2 == 0).collect::<V64<_>>();
        assert_eq!(removed.len(), 10);
        assert_eq!(removed, v64![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

        assert_eq!(v64.len(), 10);
        assert_eq!(v64, v64![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]);
    }
}

#[test]
fn test_reserve_exact() {
    // This is all the same as test_reserve

    let mut v: V64<u8> = V64::new();
    assert_eq!(v.capacity(), 7);

    v.reserve_exact(2);
    assert!(v.capacity() >= 2);

    for i in 0..16 {
        v.push(i);
    }

    assert!(v.capacity() >= 16);
    v.reserve_exact(16);
    assert!(v.capacity() >= 32);

    v.push(16);

    v.reserve_exact(16);
    assert!(v.capacity() >= 33)
}

#[test]
fn test_append_empty() {
    let mut a: V64<u8> = v64![1];
    let mut b = v64![];
    a.append(&mut b);
    assert_eq!(1, a.len());
}