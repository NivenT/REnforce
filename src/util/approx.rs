//! Function Approximator Module

use rand::{Rng, thread_rng};

use environment::Space;

use util::{Feature, QFunction};

/// QLinear
///
/// Represents a linear function approximator
/// f(x) = w^T g(x) + b
/// 	where g: S x A -> R^n maps state, action pairs to a vector of features
/// Weights updated using squared error cost
/// C = 1/2(w^T g(x) + b - y)^2
#[derive(Debug)]
pub struct QLinear<S: Space, A: Space> {
	features: Vec<Box<Feature<S, A>>>,
	weights: Vec<f64>,
	bias: f64,
}

impl<S: Space, A: Space> QFunction<S, A> for QLinear<S, A> {
	fn eval(&self, state: &S::Element, action: &A::Element) -> f64 {
		let mut ret = self.bias;
		for (i, feat) in self.features.iter().enumerate() {
			ret += self.weights[i]*feat.extract(&state, &action);
		}
		ret
	}
	fn update(&mut self, state: &S::Element, action: &A::Element, new_val: f64, alpha: f64) {
		let cost_grad = {
			let func: &mut QFunction<S, A> = self;
			func.eval(&state, &action) - new_val
		};
		for (i, feat) in self.features.iter().enumerate() {
			self.weights[i] -= alpha*cost_grad*feat.extract(&state, &action);
		}
		self.bias -= alpha*cost_grad;
	}
}

impl<S: Space, A: Space> QLinear<S, A> {
	/// Creates a new Linear Q-Function Approximator
	pub fn new() -> QLinear<S, A> {
		let mut rng = thread_rng();
		QLinear {
			features: vec![],
			weights: vec![],
			bias: rng.gen_range(-1.0, 1.0)
		}
	}
	/// Returns a clone of the weights of this function
	pub fn get_weights(&self) -> Vec<f64> {
		self.weights.clone()
	}
	/// Adds the specified feature to the end of the feature vector, giving it a random weight
	pub fn add_feature<'a>(&'a mut self, feature: Box<Feature<S, A>>) -> &'a mut QLinear<S, A> {
		let mut rng = thread_rng();
		self.weights.push(rng.gen_range(-1.0, 1.0));
		self.features.push(feature);
		self
	}
}