//! Function Approximator Module

use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;

use rand::{Rng, thread_rng};

use num::Float;
use num::cast::NumCast;

use environment::{Space, FiniteSpace};

use util::{VFunction, QFunction};
use util::{Feature, FeatureExtractor};
use util::{ParameterizedFunc, DifferentiableFunc};

/// Represents a linear function approximator
/// f(x) = w^T g(x) + b
/// 	where g: S -> R^n maps states to a vector of features
/// Weights updated using squared error cost
/// C = 1/2(w^T g(x) + b - y)^2
#[derive(Debug)]
pub struct VLinear<F: Float + Debug, S: Space> {
	features: Vec<Box<Feature<S, F>>>,
	/// 1st member of weights is bias
	weights: Vec<F>,
}

impl<F: Float + Debug, S: Space> VFunction<S> for VLinear<F, S> {
	fn eval(&self, state: &S::Element) -> f64 {
		let mut ret = self.weights[0];
		for (i, feat) in self.features.iter().enumerate() {
			ret = ret + self.weights[i+1]*feat.extract(&state);
		}
		ret.to_f64().unwrap()
	}
	fn update(&mut self, state: &S::Element, new_val: f64, alpha: f64) {
		let cost_grad = {
			let func: &mut VFunction<S> = self;
			func.eval(&state) - new_val
		};
		let lr = NumCast::from(alpha*cost_grad).unwrap();
		for (i, feat) in self.features.iter().enumerate() {
			self.weights[i+1] = self.weights[0] - lr*feat.extract(&state);
		}
		self.weights[0] = self.weights[0] - lr;
	}
}

impl<F: Float + Debug, S: Space> ParameterizedFunc<F> for VLinear<F, S> {
	fn num_params(&self) -> usize {
		self.weights.len()
	}
	fn get_params(&self) -> Vec<F> {
		self.weights.clone()
	}
	fn set_params(&mut self, params: Vec<F>) {
		self.weights = params;
	}
}

impl<S: Space, A: Space, F: Float + Debug> FeatureExtractor<S, A, F> for VLinear<F, S> {
	fn num_features(&self) -> usize {
		// One feature is constant 1
		self.weights.len()
	}
	fn extract(&self, state: &S::Element, _: &A::Element) -> Vec<F> {
		let mut feats: Vec<F> = self.features.iter().map(|feat| {
			NumCast::from(feat.extract(state)).unwrap()
		}).collect();
		feats.push(F::one());
		feats
	}
}

/*
impl<S: Space, A: Space, F: Float + Debug> DifferentiableFunc<S, A, F> for VLinear<F, S> {
	fn num_outputs(&self) -> usize {
		1
	}

	fn get_grad(&self, state: &S::Element, action: &A::Element, index: usize) -> Vec<F> {

	}
}
*/

impl<S: Space> Default for VLinear<f64, S> {
	/// Creates a new Linear V-Function Approximator
	fn default() -> VLinear<f64, S> {
		let mut rng = thread_rng();
		VLinear {
			features: vec![],
			weights: vec![rng.gen_range(-10.0, 10.0)]
		}
	}
}

impl<F: Float + Debug, S: Space> VLinear<F, S> {
	/// Creates a new Linear V-Function Approximator
	pub fn new() -> VLinear<F, S> {
		let mut rng = thread_rng();
		VLinear {
			features: vec![],
			weights: vec![NumCast::from(rng.gen_range(-10.0, 10.0)).unwrap()]
		}
	}
	/// Creates a new Linear V-Function Approximator with the given features
	pub fn with_features(feats: Vec<Box<Feature<S, F>>>) -> VLinear<F, S> {
		let mut rng = thread_rng();
		let num_feats = feats.len();
		VLinear {
			features: feats,
			weights: (0..num_feats+1).map(|_| NumCast::from(rng.gen_range(-10.0, 10.0)).unwrap()).collect()
		}
	}
	/// Adds the specified feature to the end of the feature vector, giving it a random weight
	pub fn add_feature(mut self, feature: Box<Feature<S, F>>) -> VLinear<F, S> {
		let mut rng = thread_rng();
		self.weights.push(NumCast::from(rng.gen_range(-10.0, 10.0)).unwrap());
		self.features.push(feature);
		self
	}
}

/// Represents multiple linear function approximators, one for each action
#[derive(Debug)]
pub struct QLinear<F: Float + Debug, S: Space, A: FiniteSpace> 
	where A::Element: Hash + Eq {
	functions: HashMap<A::Element, VLinear<F, S>>,
	actions: Vec<A::Element>,
	/// Indices of each action in the actions vector
	indices: HashMap<A::Element, usize>,
	/// Every linear function uses the same set of features with different weights
	features: Vec<Box<Feature<S, F>>>,
}

impl<F: Float + Debug, S: Space, A: FiniteSpace> QFunction<S, A> for QLinear<F, S, A>
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

impl<F: Float + Debug, S: Space, A: FiniteSpace> ParameterizedFunc<F> for QLinear<F, S, A>
	where A::Element: Hash + Eq {
	fn num_params(&self) -> usize {
		(self.features.len()+1)*self.actions.len()
	}
	fn get_params(&self) -> Vec<F> {
		let mut vec = Vec::with_capacity(self.num_params());
		for a in &self.actions {
			if self.functions.contains_key(a) {
				vec.extend_from_slice(&self.functions[a].get_params());
			} else {
				vec.extend_from_slice(&vec![F::zero(); self.features.len()+1]);
			}
		}
		vec
	}
	fn set_params(&mut self, params: Vec<F>) {
		let mut index = 0;
		let num_params = self.features.len()+1;
		for a in self.actions.clone() {
			let func = self.get_func(&a);
			func.set_params(params[index..index+num_params].to_vec());
			index += num_params;
		}
	}
}

impl<S: Space, A: FiniteSpace, F: Float + Debug> FeatureExtractor<S, A, F> for QLinear<F, S, A> 
	where A::Element: Hash + Eq {
	fn num_features(&self) -> usize {
		// Last feature is constant 1
		(self.features.len() + 1) * self.actions.len()
		//                          Technically, each action has its own unique set of feature
		//                          even though this set is shared across actions
	}
	fn extract(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
		let index = self.indices[action];
		let mut feats = vec![F::zero(); index*(self.features.len() + 1)];

		for feat in &self.features {
			feats.push(NumCast::from(feat.extract(state)).unwrap());
		}
		feats.push(F::one());

		feats.extend_from_slice(&vec![F::zero(); (self.actions.len()-index-1)*(self.features.len() + 1)]);
		feats
	}
}

impl<S: Space, A: FiniteSpace> QLinear<f64, S, A> where A::Element: Hash + Eq {
	/// Creates a new, empty QLinear
	pub fn default(action_space: &A) -> QLinear<f64, S, A> {
		QLinear::new(action_space)
	}
}

impl<F: Float + Debug, S: Space, A: FiniteSpace> QLinear<F, S, A> where A::Element: Hash + Eq {
	/// Creates a new, empty QLinear
	pub fn new(action_space: &A) -> QLinear<F, S, A> {
		let actions = action_space.enumerate();
		let mut indices = HashMap::new();
		for i in 0..actions.len() {
			indices.insert(actions[i].clone(), i);
		}
		QLinear {
			functions: HashMap::new(),
			actions: action_space.enumerate(),
			indices: indices,
			features: Vec::new()
		}
	}
	/// Adds feat to list of features. Should not be called after any calls to eval or update
	pub fn add(&mut self, feat: Box<Feature<S, F>>) {
		self.features.push(feat);
	}
	/// Returns a mutable reference to the function for the corresponding action
	fn get_func(&mut self, action: &A::Element) -> &mut VLinear<F, S> {
		self.functions.entry(action.clone()).or_insert(VLinear::with_features(self.features.clone()))
	}
}