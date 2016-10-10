use rand::{Rng, thread_rng};

use environment::Space;

/// Range
///
/// Represents a Space with elements drawn from some range [low, high)
#[derive(Debug, Clone, Copy)]
pub struct Range {
	/// Lower bound on elements
	low: f64,
	/// Upper bound on elements
	high: f64
}

impl Space for Range {
	type Element = f64;

	fn sample(&self) -> f64 {
		let mut rng = thread_rng();
		rng.gen_range(self.low, self.high)
	}
}

impl Range {
	/// Returns a new Range with values drawn from [low, high)
	pub fn new(low: f64, high: f64) -> Range {
		Range {
			low: low,
			high: high
		}
	}
}