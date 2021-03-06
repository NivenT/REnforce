//! Utilities Module

pub mod table;
pub mod chooser;
pub mod approx;
pub mod feature;
pub mod graddesc;

mod metric;

use std::fmt::Debug;

use num::Num;
use num::Float;

use environment::Space;

// Is there a clean way to reduce the number of traits?
// Are things progressing fine as is?

/// A function that evaluates its input by making use of some parameters
pub trait ParameterizedFunc<T: Num> {
	/// Returns number of parameters used by the function
	fn num_params(&self) -> usize;
	/// Returns the parameters used by the function
	fn get_params(&self) -> Vec<T>;
	/// Changes the parameters used by the function
	fn set_params(&mut self, params: Vec<T>);
}

/// A differentiable function taking in (state, action) pairs 
pub trait DifferentiableFunc<S: Space, A: Space, T: Num> : ParameterizedFunc<T> {
	/// Calculates the gradient of the output with respect to this function's parameters
	fn get_grad(&self, state: &S::Element, action: &A::Element) -> Vec<T>;
	/// Calculates the result of calling function on given input
	fn calculate(&self, state: &S::Element, action: &A::Element) -> T;
}

/// A differentiable function taking in a state and producing a vector output
pub trait DifferentiableVecFunc<S: Space, T: Num> : ParameterizedFunc<T> {
	/// Calculates the gradient of the output vector with respect to each parameter
	fn get_grads(&self, state: &S::Element) -> Vec<Vec<T>>; // gradient vector for each parameter
	/// Applies this function to the given state
	fn apply(&self, state: &S::Element) -> Vec<T>;
}

/// A function taking in (state, action) pairs whose log can be differentiated
pub trait LogDiffFunc<S: Space, A: Space, T: Num> : ParameterizedFunc<T> {
	/// The gradient of the log of the output with respect to the parameters
	fn log_grad(&self, state: &S::Element, action: &A::Element) -> Vec<T>;
}

/// Calculates gradient steps
pub trait GradientDescAlgo<F: Float> {
	/// Calculates local step for maximizing the function
	fn calculate(&mut self, grad: Vec<F>, lr: F) -> Vec<F>;
}

/// Represents something that extracts features from state-action pairs
pub trait FeatureExtractor<S: Space, A: Space, F: Float> {
	/// Number of features that can be calculated
	fn num_features(&self) -> usize;
	/// Vector containg the values of all the features for this state
	fn extract(&self, state: &S::Element, action: &A::Element) -> Vec<F>;
}

/// QFunction Trait
///
/// Represents a function Q: S x A -> R that takes in a (state, action) pair
/// and returns the value of that pair
pub trait QFunction<S: Space, A: Space> : Debug {
	/// Evaluate the function on the given state and action
	fn eval(&self, state: &S::Element, action: &A::Element) -> f64;
	/// Update the function using the given information (alpha is learning rate)
	fn update(&mut self, state: &S::Element, action: &A::Element, new_val: f64, alpha: f64);
}

/// VFunction Trait
///
/// Represents a function V: S -> R that takes in a state and returns its value
pub trait VFunction<S: Space> : Debug {
	/// Evaluate the function on the given state
	fn eval(&self, state: &S::Element) -> f64;
	/// Update the function using the given information (alpha is learning rate)
	fn update(&mut self, state: &S::Element, new_val: f64, alpha: f64);
}

/// Choose Trait
///
/// Represents a way to randomly choose an element of a list given some weights
pub trait Chooser<T> : Debug {
	/// returns an element of choices
	fn choose(&self, choices: &Vec<T>, weights: Vec<f64>) -> T;
}

/// A real-valued feature of elements of some state space
pub trait Feature<S: Space, F: Float> : Debug {
	/// Extracts some real-valued feature from a given state
	fn extract(&self, state: &S::Element) -> F;
	/// Creates a cloned trait object of self
	fn box_clone(&self) -> Box<Feature<S, F>>;
}

impl<F: Float, S: Space> Clone for Box<Feature<S, F>> {
	fn clone(&self) -> Self {
		self.box_clone()
	}
}

/// A type with a notion of distance
/// The distance function should satisfy the triangle inequality (and the other [metric](https://www.wikiwand.com/en/Metric_(mathematics)) properties)
///
/// d(x,z) <= d(x,y) + d(y,z)
pub trait Metric {
	/// Returns the distance between x and y
	fn dist(x: &Self, y: &Self) -> f64 {
		Metric::dist2(x, y).sqrt()
	}
	/// Returns the squared distance between x and y
	fn dist2(x: &Self, y: &Self) -> f64;
}

/// Some length of time experienced by an agent
#[derive(Debug, Clone)]
pub enum TimePeriod {
	/// A time period stored as a number of episodes
	EPISODES(usize),
	/// A time period stored as a number of individual timesteps
	TIMESTEPS(usize),
	/// Time period ends when first or second one ends
	OR(Box<TimePeriod>, Box<TimePeriod>),
}

impl TimePeriod {
	/// Returns whether or not self represents an empty time period
	pub fn is_none(&self) -> bool {
		match *self {
			TimePeriod::EPISODES(x) => x == 0,
			TimePeriod::TIMESTEPS(x) => x == 0,
			TimePeriod::OR(ref a, ref b) => a.is_none() || b.is_none(),
		}
	}
	/// Returns the time period remaing after one time step
	pub fn dec(&self, done: bool) -> TimePeriod {
		if self.is_none() {
			self.clone()
		} else {
			match *self {
				TimePeriod::EPISODES(x) => TimePeriod::EPISODES(if done {x-1} else {x}),
				TimePeriod::TIMESTEPS(x) => TimePeriod::TIMESTEPS(x-1),
				TimePeriod::OR(ref a, ref b) => TimePeriod::OR(Box::new(a.dec(done)), Box::new(b.dec(done))),
			}
		}
	}
}