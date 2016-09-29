use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

pub struct Finite {
	size: u64,
}

impl Space for Finite {
	type Element = u64;

	fn sample(&self) -> u64 {
		let mut rng = thread_rng();
		rng.gen_range(0, self.size)
	}
}

impl FiniteSpace for Finite {
	fn enumerate(&self) -> Vec<u64> {
		(0..self.size).collect()
	}
}