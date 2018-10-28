- full benchmarks
    - find capacity points from 10 up to 5M.
        - bench cap +- 10%, (cap(N) + cap(N-1)) / 2
    - both (u32, u32) and (u64, u64)
    - some benchmarks from std and others
    - https://probablydance.com/2017/02/26/i-wrote-the-fastest-hashtable/
    - https://accidentallyquadratic.tumblr.com/post/153545455987/rust-hash-iteration-reinsertion
    - https://tessil.github.io/2016/08/29/benchmark-hopscotch-map.html
- Override methods count, last, and nth in thinvec::IntoIter.
    - can only optimize if drop is not required!
- benchmark results
- guard against large allocation and alloc failure
- method optimization and code reduction
- consider supporting no-std?
- fix links in the docs
- thinvec/v64: more efficient extend for slices, other vectors and such
- v64: implement shrink back to stack
- v64: into_iter without heap allocation
- more tests
    - around sentinels
    - custom keys
    - keys with drop???? Messes badly with ThinSentinel
    - maps with removed stuff (iteration, etc)
    - panic tests for vector index operations
- implement clone/eq for hashers so they can be tested
- post upload to github:
    - travis
    - coverage via travis
        - with tarpualin or kcov?
- a generic pointer sentinel?
    - alloc 1 byte with 64-byte alignment?
- done: careful with pointer assignments: "Note that *self = foo counts as a use because it will attempt to drop the value previously at *self."
    regex: ```\*[a-z\.0-9]+[).01]*.=[^=]```
- ThinMap has a 62.5% loadfactor.
    so each (K,V) takes up about (size_of((K,V))) * 100/(62.5 + 62.5/2) * 2 space.
- HashMap has a 90% loadfactor.
    so each (K,V) takes up about (size_of((K,V)) + 8) * 100/(90 + 45) * 2 space.
- ThinMap vs HashMap equal space point:
    SKV * 0.01067 == (SKV + 8) * 0.0074
    SKV * 0.00327 == 8 * 0.0074
    SKV = 18
- for size_of((K,V)) <= 18 ThinMap is always better. For example, (u64, i64) is better as thinmap.
- for size_of(T) <= 18, ThinSet is always better. For example, u128 is better as thinset.
- V128:
    - use first 64 bits like V64. Would need 32 byte alignment.
    - use second 64bits for more data. 8 more bytes. 15 u8 max, which fits in 4 bits.
    - use second 64 bits in heap mode to keep track of size, capacity up to 32bits (4G elements)
- map: ZERO is all bits zero for performance?
    - no! too dangerous
- map: consider a 75% load factor (needs benching)
    - no! too big a zig-zag
