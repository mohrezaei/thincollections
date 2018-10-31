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

use thincollections::cla_map::ClaMap;
use thincollections::thin_hasher::*;

#[test]
fn test_a_few_inserts_get() {
    let mut cla_map = ClaMap::new();
    cla_map.insert(0i32, 100u32);
//    cla_map.debug();
    let mut c = 1;
    while c < 10 {
        cla_map.insert(c, c.wrapping_mul(10) as u32);
        c += 1;
    }
}

#[test]
fn test_simple_insert() {
    let thin_map = map_1_m();
    assert_eq!(1_000_000, thin_map.len());
//    println!("{:?}", &thin_map);
}

fn map_1_m() -> ClaMap<i32, u32, OneFieldHasherBuilder> {
    let mut cla_map = ClaMap::new();
    cla_map.insert(10i32, 100u32);
//    println!("{:?}", &cla_map);
    let mut c: i32 = 0;
    while c < 1_000_000 {
        cla_map.insert(c, c.wrapping_mul(10) as u32);
//    println!("{:?}", &cla_map);
        c += 1;
    }
    cla_map
}
