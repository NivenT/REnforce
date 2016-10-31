//! Function Approximator Module

use rand::{Rng, thread_rng};

use environment::Space;

use util::{Feature, VFunction};

/// QLinear
///
/// Represents a linear function approximator
/// f(x) = w^T g(x) + b
/// 	where g: S -> R^n maps states to a vector of features
/// Weights updated using squared error cost
/// C = 1/2(w^T g(x) + b - y)^2
#[derive(Debug)]
pub struct VLinear<S: Space> {
	features: Vec<Box<Feature<S>>>,
	weights: Vec<f64>,
	bias: f64,
}

impl<S: Space> VFunction<S> for VLinear<S> {
	fn eval(&self, state: &S::Element) -> f64 {
		let mut ret = self.bias;
		for (i, feat) in self.features.iter().enumerate() {
			ret += self.weights[i]*feat.extract(&state);
		}
		/*
		println!("features: {:?}", self.features.iter().map(|feat| {
			feat.extract(&state)
		}).collect::<Vec<_>>());
		println!("weights: {:?}", self.weights);
		println!("bias: {}", self.bias);
		println!("val: {}\n", ret);
		*/
		ret
	}
	fn update(&mut self, state: &S::Element, new_val: f64, alpha: f64) {
		let cost_grad = {
			let func: &mut VFunction<S> = self;
			func.eval(&state) - new_val
		};
		for (i, feat) in self.features.iter().enumerate() {
			self.weights[i] -= alpha*cost_grad*feat.extract(&state);
		}
		self.bias -= alpha*cost_grad;
	}
}

impl<S: Space> VLinear<S> {
	/// Creates a new Linear Q-Function Approximator
	pub fn new() -> VLinear<S> {
		let mut rng = thread_rng();
		VLinear {
			features: vec![],
			weights: vec![],
			bias: rng.gen_range(-10.0, 10.0)
		}
	}
	/// Returns a clone of the weights of this function
	pub fn get_weights(&self) -> Vec<f64> {
		self.weights.clone()
	}
	/// Adds the specified feature to the end of the feature vector, giving it a random weight
	pub fn add_feature(mut self, feature: Box<Feature<S>>) -> VLinear<S> {
		let mut rng = thread_rng();
		self.weights.push(rng.gen_range(-10.0, 10.0));
		self.features.push(feature);
		self
	}
}