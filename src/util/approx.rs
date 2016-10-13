//! Function Approximator Module

use std::marker::PhantomData;
use std::fmt::Debug;

use rand::{Rng, thread_rng};

use environment::Space;

use util::QFunction;

/// QLinear
///
/// Represents a linear function approximator
/// f(x) = w^T x + b
/// Weights updated using squared error cost
/// C = 1/2(w^T x + b - y)^2
#[derive(Debug, Clone)]
pub struct QLinear<T: Into<f64>> {
	weights: Vec<f64>,
	bias: f64,
	phantom: PhantomData<T>,
}

impl<T: Into<f64> + Debug, S: Space, A: Space> QFunction<S, A> for QLinear<T>
	where S::Element: Into<Vec<T>>, A::Element: Into<Vec<T>> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64 {
		let mut index = 0;
		let mut ret = self.bias;
		for s in state.into() {
			ret += self.weights[index]*s.into();
			index += 1;
		}
		for a in action.into() {
			ret += self.weights[index]*a.into();
			index += 1;
		}
		ret
	}
	fn update(&mut self, state: S::Element, action: A::Element, new_val: f64, alpha: f64) {
		let cost_grad = {
			let func: &mut QFunction<S, A> = self;
			func.eval(state.clone(), action.clone()) - new_val
		};
		let mut index = 0;
		for s in state.into() {
			self.weights[index] -= alpha*cost_grad*s.into();
			index += 1;
		}
		for a in action.into() {
			self.weights[index] -= alpha*cost_grad*a.into();
			index += 1;
		}
		self.bias -= alpha*cost_grad;
	}
}

impl<T: Into<f64>> QLinear<T> {
	/// Creates a new Linear Q-Function Approximator
	pub fn new(n: usize) -> QLinear<T> {
		let mut rng = thread_rng();
		QLinear {
			weights: (0..n).map(|_| rng.gen_range(0.0, 1.0)).collect(),
			bias: rng.gen_range(0.0, 1.0),
			phantom: PhantomData
		}
	}
	/// Returns a clone of the weights of this function
	pub fn get_weights(&self) -> Vec<f64> {
		self.weights.clone()
	}
}