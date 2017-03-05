//! Gradient Descent Module

use num::Float;

use util::GradientDescAlgo;

/// Simplest possible Gradient Descent algorithm
/// Gradient step is just gradient * learning_rate
#[derive(Clone, Copy, Debug)]
pub struct GradientDesc;

// Up to clinet to negate step if needed
impl<T: Float> GradientDescAlgo<T> for GradientDesc {
	fn calculate(&mut self, mut grad: Vec<T>, lr: T) -> Vec<T> {
		for x in &mut grad {
			*x = *x * lr;
		}
		grad
	}
}