//! Function Approximator Module

use rand::{Rng, thread_rng};

use environment::Space;

use util::QFunction;

/// QLinear
///
/// Represents a linear function approximator
/// f(x) = w^T x
/// Weights updated using squared error cost
/// C = 1/2(w^T x - y)^2
#[derive(Debug, Clone)]
pub struct QLinear {
	weights: Vec<f64>,
}

impl<S: Space, A: Space> QFunction<S, A> for QLinear
	where S::Element: Into<Vec<f64>>, A::Element: Into<Vec<f64>> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64 {
		let mut ret = 0.0;
		let mut index = 0;
		for s in state.into() {
			ret += self.weights[index]*s;
			index += 1;
		}
		for a in action.into() {
			ret += self.weights[index]*a;
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
			self.weights[index] -= alpha*cost_grad*s;
			index += 1;
		}
		for a in action.into() {
			self.weights[index] -= alpha*cost_grad*a;
			index += 1;
		}
	}
}

impl QLinear {
	/// Creates a new Linear Q-Function Approximator
	pub fn new(n: usize) -> QLinear {
		let mut rng = thread_rng();
		QLinear {
			weights: (0..n).map(|_| rng.gen_range(0.0,1.0)).collect()
		}
	}
}