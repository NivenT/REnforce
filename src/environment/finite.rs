//! Finite Module

use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

/// Finite
///
/// Represents a Space with finitely many elements, {0, 1, 2, ..., size-1}
#[derive(Debug, Clone, Copy)]
pub struct Finite {
	/// The number of elements in the space
	size: usize,
}

impl Space for Finite {
	type Element = usize;

	fn sample(&self) -> usize {
		let mut rng = thread_rng();
		rng.gen_range(0, self.size)
	}
}

impl FiniteSpace for Finite {
	fn enumerate(&self) -> Vec<usize> {
		(0..self.size).collect()
	}
}

impl Finite {
	/// Returns a new Finite with the given number of elements
	pub fn new(size: usize) -> Finite {
		Finite {
			size: size
		}
	}
}