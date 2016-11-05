//! # The REnforce crate.
//!
//! A crate built for reinforcment learning.

#![deny(missing_docs, trivial_casts, unstable_features)]
#![warn(missing_debug_implementations, unused_import_braces, unused_qualifications)]

extern crate rand;

pub mod environment;
pub mod trainer;
pub mod agent;
pub mod util;