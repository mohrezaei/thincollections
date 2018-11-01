thincollections
===============

[![Latest version](https://img.shields.io/crates/v/thincollections.svg)](https://crates.io/crates/thincollections)
[![Documentation](https://docs.rs/thincollections/badge.svg)](https://docs.rs/thincollections)
![Minimum rustc version](https://img.shields.io/badge/rustc-1.28+-yellow.svg)

Alternative implementations for vector, map and set that are faster/smaller for some use cases.
`ThinMap` can be 2x to 5x faster than `std::collections::HashMap`. See the
[benchmarks](https://github.com/mohrezaei/thincollections/blob/master/benchmarks/map-results.md).

- [Documentation](https://docs.rs/thincollections)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
thincollections = "0.5"
```

and this to your crate root:

```rust
#[macro_use]
extern crate thincollections;
```

## Rust Version Support

The minimum supported Rust version is 1.28 due to use of allocator api and NonZero*.