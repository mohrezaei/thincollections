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

extern crate thincollections;

use thincollections::thin_hasher::*;
use thincollections::thin_map::ThinMap;
use thincollections::thin_sentinel::ThinSentinel;
use std::ptr;

#[derive(PartialEq, Eq, Hash)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl ThinSentinel for Color {
    fn thin_sentinel_zero() -> Self {
        Color { r: 0, g: 0, b: 0 }
    }

    fn thin_sentinel_one() -> Self {
        Color { r: 0, g: 0, b: 1 }
    }
}

#[test]
fn custom_key()
{
    let mut thin_map = ThinMap::new();
    thin_map.insert(Color {r: 0, g: 0, b: 0}, 17);
    thin_map.insert(Color {r: 0, g: 0, b: 1}, 42);
    thin_map.insert(Color {r: 1, g: 1, b: 1}, 1);

    assert_eq!(17, *thin_map.get(&Color {r: 0, g: 0, b: 0}).unwrap());
    assert_eq!(42, *thin_map.get(&Color {r: 0, g: 0, b: 1}).unwrap());
    assert_eq!(1, *thin_map.get(&Color {r: 1, g: 1, b: 1}).unwrap());
}

#[test]
fn test_simple_insert() {
    let thin_map = map_1_m();
    assert_eq!(1_000_000, thin_map.len());
//    println!("{:?}", &thin_map);
}

fn map_1_m() -> ThinMap<i32, u32, OneFieldHasherBuilder> {
    let mut thin_map = ThinMap::new();
    thin_map.insert(10i32, 100u32);
//    thin_map.debug();
    let mut c: i32 = 0;
    while c < 1_000_000 {
        thin_map.insert(c, c.wrapping_mul(10) as u32);
        c += 1;
    }
    thin_map
}

#[test]
fn test_simple_get() {
    let mut thin_map = ThinMap::new();
    assert_eq!(None, thin_map.get(&0));
    assert_eq!(None, thin_map.get(&1));
    assert_eq!(None, thin_map.get(&100));
    thin_map.insert(0i32, 100u32);
//    thin_map.debug();
    let mut c = 1;
    while c < 1_000_000 {
        thin_map.insert(c, c.wrapping_mul(10) as u32);
        c += 1;
    }
    assert_eq!(1_000_000, thin_map.len());
    assert_eq!(100u32, *thin_map.get(&0i32).unwrap());
    c = 1;
    while c < 1_000_000 {
        assert_eq!(c.wrapping_mul(10) as u32, *thin_map.get(&c).unwrap());
        c += 1;
    }
    while c < 2_000_000 {
        assert_eq!(None, thin_map.get(&c));
        c += 1;
    }
}

#[test]
fn test_remove()
{
    let mut thin_map = map_1_m();
    assert_eq!(1_000_000, thin_map.len());
    let mut c: i32 = 0;
    while c < 1_000_000 {
        assert_eq!(Some(c.wrapping_mul(10) as u32), thin_map.remove(&c), "For key {}", c);
        c += 1;
    }
    assert_eq!(0, thin_map.len());
    c = 0;
    while c < 1_000_000 {
        assert_eq!(None, thin_map.remove(&c));
        c += 1;
    }
}

#[test]
fn test_contains_key()
{
    let empty_map: ThinMap<i32, u32, OneFieldHasherBuilder> = ThinMap::new();
    assert_eq!(false, empty_map.contains_key(&0));
    assert_eq!(false, empty_map.contains_key(&1));
    let thin_map = map_1_m();
    assert_eq!(1_000_000, thin_map.len());
    let mut c: i32 = 0;
    while c < 1_000_000 {
        assert_eq!(true, thin_map.contains_key(&c), "For key {}", c);
        c += 1;
    }
    while c < 2_000_000 {
        assert_eq!(false, thin_map.contains_key(&c));
        c += 1;
    }
}

#[test]
fn test_drain() {
    let mut map: ThinMap<i32, u32, OneFieldHasherBuilder> = ThinMap::new();
    {
        for (_k, _v) in map.drain().take(1) {
            assert!(false, "should not get here");
        }
    }
    map.insert(0, 100);
    {
        let mut count = 0;
        for (k, v) in map.drain().take(1) {
            count += 1;
            assert_eq!((0, 100), (k, v));
        }
        assert_eq!(1, count);
    }
    assert_eq!(0, map.len());
    map.insert(1, 100);
    {
        let mut count = 0;
        for (k, v) in map.drain().take(1) {
            count += 1;
            assert_eq!((1, 100), (k, v));
        }
        assert_eq!(1, count);
    }
    assert_eq!(0, map.len());
    map.insert(2, 100);
    {
        let mut count = 0;
        for (k, v) in map.drain().take(1) {
            count += 1;
            assert_eq!((2, 100), (k, v));
        }
        assert_eq!(1, count);
    }
    assert_eq!(0, map.len());
}

#[test]
fn test_iter() {
    let mut map: ThinMap<i32, u32, OneFieldHasherBuilder> = ThinMap::new();
    {
        let iter = map.iter();
        for _kv in iter {
            assert!(false, "should not get here");
        }
    }
    map.insert(0, 100);
    {
        let iter = map.iter();
        for kv in iter {
            assert_eq!((0, 100), (*kv.0, *kv.1));
        }
    }
    map.clear();
    assert_eq!(0, map.len());
    map.insert(1, 100);
    {
        let iter = map.iter();
        for kv in iter {
            assert_eq!((1, 100), (*kv.0, *kv.1));
        }
    }
    map.clear();
    assert_eq!(0, map.len());
    map.insert(100, 200);
    map.insert(1, 100);
    {
        let iter = map.iter();
        let mut count = 0;
        for kv in iter {
            count += 1;
            if *kv.0 == 100 {
                assert_eq!(200, *kv.1);
            } else if *kv.0 == 1 {
                assert_eq!(100, *kv.1);
            } else {
                assert!(false, "Should not get here");
            }
        }
        assert_eq!(2, count);
    }
    let map = map_1_m();
    {
        let iter = map.iter();
        let mut count = 0;
        for kv in iter {
            count += 1;
            assert_eq!(*kv.1 as u64, *kv.0 as u64 * 10)
        }
        assert_eq!(1_000_000, count);
    }
}

#[test]
fn test_into_iter() {
    let map = map_1_m();
    let vec: Vec<(i32, u32)> = map.into_iter().collect();
    assert_eq!(1_000_000, vec.len());
}

struct Pu32u16 {
    _a: u32,
    _b: u16,
}

#[test]
fn play() {
    unsafe {
        let mut x: u64 = 0;
        let xptr = &mut x as *mut u64 as *mut u8;
        ptr::write(xptr, 1);
        println!("{:x?}", x);
        ptr::write(xptr.add(1), 1);
        println!("{:x?}", x);
        x = 0;
        ptr::write(xptr as *mut u32, 0x12345678);
        println!("{:x?}", x);
        x = 0xFFFFFFFF;
        ptr::write(xptr as *mut Pu32u16, Pu32u16 { _a: 0x12345678, _b: 0x9ABC });
        println!("{:x?}", x);
    }
}

//fn junk()
//{
//    let mut h: HashMap<i32, u32> = HashMap::new();
//    h.drain();
//    let x: String;
//    let mut s: HashSet<u32> = HashSet::new();
//    let a: ThinSet<i32> = [1, 2, 3].iter().cloned().collect();
//    let b: ThinSet<i32> = [4, 2, 3, 4].iter().cloned().collect();
//    let v = vec![7, 8, 9];
//    let option: Option<(&i32, &u32)> = h.get_key_value(&56);
//    let x:Option<&mut u32> = h.get_mut(&33);
//    let x:Option<(i32, u32)> = h.remove_entry(&34);
//    h.retain(| &kref, &mut vref | kref > (vref as i32));
//}