use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

/// Finite
///
/// Represents a Space with finitely many elements, {0, 1, 2, ..., size-1}
#[derive(Debug, Clone, Copy)]
pub struct Finite {
	/// The number of elements in the space
	size: u32,
}

impl Space for Finite {
	type Element = u32;

	fn sample(&self) -> u32 {
		let mut rng = thread_rng();
		rng.gen_range(0, self.size)
	}
}

impl FiniteSpace for Finite {
	fn enumerate(&self) -> Vec<u32> {
		(0..self.size).collect()
	}

	fn index(&self, elm: u32) -> isize {
		if elm < self.size {elm as isize} else {-1}
	}
}

impl Finite {
	/// Returns a new Finite with the given number of elements
	pub fn new(size: u32) -> Finite {
		Finite {
			size: size
		}
	}
}