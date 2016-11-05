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
	let mean = sum / NumCast::from(nums.len()).unwrap();
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