//! # The REnforce crate.
//!
//! A crate built for reinforcment learning.

#![deny(missing_docs, trivial_casts, unstable_features, unused_extern_crates)]
#![warn(missing_debug_implementations, unused_import_braces, unused_qualifications)]

extern crate rand;
extern crate num;
extern crate rulinalg;

pub mod environment;
pub mod prelude;
pub mod trainer;
pub mod model;
pub mod agent;
pub mod util;
pub mod stat;