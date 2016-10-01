use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

#[derive(Debug, Clone, Copy)]
pub struct Finite {
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
	pub fn new(size: usize) -> Finite {
		Finite {
			size: size
		}
	}
}