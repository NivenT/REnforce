//! Statistics Module

use num::Float;
use num::cast::NumCast;

/// Calculates the mean and variance of a set of numbers
pub fn mean_var<T: Float>(nums: &[T]) -> (T, T) {
	let (mut sum, mut sq_sum) = (T::zero(), T::zero());
	for &x in nums {
		sum = sum + x;
		sq_sum = sq_sum + x*x;
	}
	let mean = sum/NumCast::from(nums.len()).unwrap();
	let variance = sq_sum/NumCast::from(nums.len()).unwrap() - mean*mean;
	(mean, variance)
}

/// Calculates the mean of a set of numbers
pub fn mean<T: Float>(nums: &[T]) -> T {
	let mut sum = T::zero();
	for &x in nums {
		sum = sum + x;
	}
	sum/NumCast::from(nums.len()).unwrap()
}

/// Calculates the variance of a set of numbers
pub fn variance<T: Float>(nums: &[T]) -> T {
	mean_var(nums).1
}

/// Calculates the standard deviation of a set of numbers
pub fn stddev<T: Float>(nums: &[T]) -> T {
	variance(nums).sqrt()
}

#[cfg(test)]
mod test {
	use super::mean_var;

	const EPSILON: f64 = 0.000001;

	#[test]
	fn mean_var_simple() {
		let nums = vec![0.0, 1.0];
		let (mean, var) = mean_var(&nums);

		assert!(mean - 0.5 < EPSILON);
		assert!(var - 0.25 < EPSILON);
	}
	#[test]
	fn mean_var_big_nums() {
		let nums = vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0];
		let (mean, var) = mean_var(&nums);

		assert!(mean - 18518.5 < EPSILON);
		assert!(var - 1608680209.5 < EPSILON);
	}
	#[test]
	fn stddev_range_0_100() {
		let nums: Vec<_> = (0..100).map(|n| n as f64).collect();
		let stddev = mean_var(&nums).1.sqrt();

		assert!(stddev - 29.0115 < EPSILON);
	}
}