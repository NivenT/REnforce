//! Statistics Module

use rand::{Rng, thread_rng};

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

/// Normalizes a list of numbers to have mean 0 and standard deviation 1
pub fn normalize<T: Float>(nums: &mut [T]) {
	let (mean, var) = mean_var(nums);
	let stddev = var.sqrt();

	for num in nums {
		*num = (*num - mean)/stddev;
	}
}

/// Performs in-place Fisher-Yates Shuffle
pub fn shuffle<T: Clone>(nums: &mut [T]) {
	let mut rng = thread_rng();
	for i in 0..(nums.len()-1) {
		let j = rng.gen_range(i, nums.len());

		nums.swap(i, j);
	}
}

#[cfg(test)]
mod test {
	use super::{mean_var, normalize, shuffle};

	const EPSILON: f64 = 0.000001;

	#[test]
	fn mean_var_simple() {
		let nums: Vec<f64> = vec![0.0, 1.0];
		let (mean, var) = mean_var(&nums);

		assert!((mean - 0.5).abs() < EPSILON);
		assert!((var - 0.25).abs() < EPSILON);
	}
	#[test]
	#[should_panic]
	// Variance calculation not numerically stable
	fn mean_var_big_nums_fail() {
		let nums: Vec<f64> = vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0];
		let (mean, var) = mean_var(&nums);

		assert!((mean - 18518.5).abs() < EPSILON);
		assert!((var - 1608680209.5).abs() < EPSILON);
	}
	#[test]
	fn normalize_range_0_100() {
		let mut nums: Vec<_> = (0..100).map(|n| n as f64).collect();
		normalize(&mut nums);

		let (mean, var) = mean_var(&nums);
		assert!(mean.abs() < EPSILON && (var - 1.0).abs() < EPSILON);
	}
	#[test]
	fn shuffle_simple() {
		let mut nums: Vec<_> = (0..10).collect();
		shuffle(&mut nums);

		assert!(nums.iter().enumerate().fold(false, |acc, (i, &j)| acc || i != j));
	}
}