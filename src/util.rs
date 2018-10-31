// Copyright 2018 Mohammad Rezaei.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#[inline]
pub fn ceil_pow2(x: u64) -> u64 {
    if x == 0 { return 1; }
    let highest_one_bit = 1u64 << (64 - x.leading_zeros() - 1);
    if x > highest_one_bit { return highest_one_bit << 1; }
    highest_one_bit
}

#[inline]
pub fn spread_two(code: u64) -> u64 {
    let mut r = code;
    r ^= r >> 23;
    r = r.wrapping_mul(-6261870919139520145i64 as u64);
    r ^= r >> 39;
    r = r.wrapping_mul(2747051607443084853u64);
    r ^= r >> 37;
    r
}

#[inline]
pub fn spread_one(code: u64) -> u64 {
    let mut r = code;
    r ^= r >> 28;
    r = r.wrapping_mul(-4254747342703917655i64 as u64);
    r ^= r >> 43;
    r = r.wrapping_mul(-908430792394475837i64 as u64);
    r ^= r >> 23;
    r
}

#[inline]
pub fn spread_three(code: u64) -> u64 {
    let mut r = code;
    r ^= r >> 26;
    r = r.wrapping_mul(8238576523158062045u64);
    r ^= r >> 35;
    r = r.wrapping_mul(-6410243847380211633i64 as u64);
    r ^= r >> 34;
    r
}
