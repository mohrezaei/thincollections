// Copyright 2018 Mohammad Rezaei.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
#![feature(test)]

extern crate test;
#[macro_use]
extern crate thincollections;
extern crate rand;

use thincollections::thin_v64::V64;
use thincollections::thin_vec::ThinVec;

use test::Bencher;
use test::stats::Summary;
use test::black_box;
use std::rc::Rc;

#[bench]
pub fn benchv_v64_1M_insert(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: V64<u64> = V64::with_capacity(16);
        for i in 0..1_000_000 {
            vec.push(i);
        }
        black_box(vec.len());
    });
}

#[bench]
pub fn benchv_vec_1M_insert(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: Vec<u64> = Vec::with_capacity(16);
        for i in 0..1_000_000 {
            vec.push(i);
        }
        black_box(vec.len());
    });
}

#[bench]
pub fn benchv_thinvec_1M_insert(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: ThinVec<u64> = ThinVec::with_capacity(16);
        for i in 0..1_000_000 {
            vec.push(i);
        }
        black_box(vec.len());
    });
}

#[bench]
pub fn benchv_thinthin(b: &mut Bencher) {
    b.iter(|| {
        let mut vv: ThinVec<ThinVec<u32>> = ThinVec::with_capacity(16);
        for i in 0..1_000_000 {
            let mut v: ThinVec<u32> = ThinVec::new();
            if i % 2 == 0 {
                v.push(1);
            }
            if i % 3 == 0 {
                v.push(1);
            }
            vv.push(v);
        }
        black_box(vv.len());
    });
}

#[bench]
pub fn benchv_thinv64(b: &mut Bencher) {
    b.iter(|| {
        let mut vv: ThinVec<V64<u32>> = ThinVec::with_capacity(16);
        for i in 0..1_000_000 {
            let mut v: V64<u32> = V64::new();
            if i % 2 == 0 {
                v.push(1);
            }
            if i % 3 == 0 {
                v.push(1);
            }
            vv.push(v);
        }
        black_box(vv.len());
    });
}

#[bench]
pub fn benchv_vecvec(b: &mut Bencher) {
    b.iter(|| {
        let mut vv: Vec<Vec<u32>> = Vec::with_capacity(16);
        for i in 0..1_000_000 {
            let mut v: Vec<u32> = Vec::new();
            if i % 2 == 0 {
                v.push(1);
            }
            if i % 3 == 0 {
                v.push(1);
            }
            vv.push(v);
        }
        black_box(vv.len());
    });
}

fn powerset<T>(s: &[T]) -> Vec<Vec<T>> where T: Clone {
    (0..2usize.pow(s.len() as u32)).map(|i| {
        s.iter().enumerate().filter(|&(t, _)| (i >> t) % 2 == 1)
            .map(|(_, element)| element.clone())
            .collect()
    }).collect()
}

fn powerset_thin<T>(s: &[T]) -> ThinVec<ThinVec<T>> where T: Clone {
    (0..2usize.pow(s.len() as u32)).map(|i| {
        s.iter().enumerate().filter(|&(t, _)| (i >> t) % 2 == 1)
            .map(|(_, element)| element.clone())
            .collect()
    }).collect()
}

#[bench]
pub fn benchp_vec(b: &mut Bencher) {
    b.iter(|| {
        let v: Vec<i32> = vec![1; 20];
        let x = powerset(&v);
        black_box(x.len());
    })
}

#[bench]
pub fn benchp_thin(b: &mut Bencher) {
    b.iter(|| {
        let v: ThinVec<i32> = thinvec![1; 20];
        let x = powerset_thin(&v);
        black_box(x.len());
    })
}

#[bench]
pub fn benchprc_vec(b: &mut Bencher) {
    b.iter(|| {
        let v: Vec<_> = vec![Rc::new("b"); 20];
        let x = powerset(&v);
        black_box(x.len());
    })
}

#[bench]
pub fn benchprc_thin(b: &mut Bencher) {
    b.iter(|| {
        let v: ThinVec<_> = thinvec![Rc::new("b"); 20];
        let x = powerset_thin(&v);
        black_box(x.len());
    })
}


#[bench]
pub fn benchx(b: &mut Bencher) {
    b.iter(|| {
        let v: ThinVec<_> = thinvec![Rc::new("b"); 20];
        let x = powerset_thin(&v);
        black_box(x.len());
    })
}

