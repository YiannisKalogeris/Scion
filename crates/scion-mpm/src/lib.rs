//! Native kernels for Scion's material point method.
//!
//! Deliberately free of any Python dependency, so the kernels can be unit
//! tested with `cargo test` alone and benchmarked without an interpreter in
//! the loop. The PyO3 bindings live in the sibling `scion-mpm-py` crate.

pub mod scatter;

pub use scatter::Scatter;
