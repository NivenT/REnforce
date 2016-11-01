//! Function Approximator Module

use std::collections::HashMap;
use std::hash::Hash;

use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

use util::{Feature, VFunction, QFunction};

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
	/// Creates a new Linear V-Function Approximator
	pub fn new() -> VLinear<S> {
		let mut rng = thread_rng();
		VLinear {
			features: vec![],
			weights: vec![],
			bias: rng.gen_range(-10.0, 10.0)
		}
	}
	/// Creates a new Linear V-Function Approximator with the given features
	pub fn with_features(feats: Vec<Box<Feature<S>>>) -> VLinear<S> {
		let mut rng = thread_rng();
		let num_feats = feats.len();
		VLinear {
			features: feats,
			weights: (0..num_feats).map(|_| rng.gen_range(-10.0,10.0)).collect(),
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

/// Represents multiple linear function approximators, one for each action
#[derive(Debug)]
pub struct QLinear<S: Space, A: FiniteSpace> 
	where A::Element: Hash + Eq {
	functions: HashMap<A::Element, VLinear<S>>,
	/// Every linear function uses the same set of features with different weights
	features: Vec<Box<Feature<S>>>,
}

impl <S: Space, A: FiniteSpace> QFunction<S, A> for QLinear<S, A>
	where A::Element: Hash + Eq {
	fn eval(&self, state: &S::Element, action: &A::Element) -> f64 {
		if self.functions.contains_key(action) {
			self.functions[action].eval(state)
		} else {
			0.0
		}
	}
	fn update(&mut self, state: &S::Element, action: &A::Element, new_val: f64, alpha: f64) {
		let func = self.get_func(action);
		func.update(state, new_val, alpha);
	}
}

impl<S: Space, A: FiniteSpace> QLinear<S, A> where A::Element: Hash + Eq {
	/// Creates a new, empty QLinear
	pub fn new() -> QLinear<S, A> {
		QLinear {
			functions: HashMap::new(),
			features: Vec::new()
		}
	}
	/// Adds feat to list of features. Should not be called after any calls to eval or update
	pub fn add(&mut self, feat: Box<Feature<S>>) {
		self.features.push(feat);
	}
	/// Returns a mutable reference to the function for the corresponding action
	fn get_func(&mut self, action: &A::Element) -> &mut VLinear<S> {
		self.functions.entry(action.clone()).or_insert(VLinear::with_features(self.features.clone()))
	}
}