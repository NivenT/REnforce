//! Utilities Module

pub mod table;
pub mod chooser;
pub mod approx;
pub mod feature;
mod metric;

use std::fmt::Debug;

use num::Num;

use environment::Space;

/// A function that evaluates its input by making use of some parameters
pub trait ParameterizedFunc<T: Num> {
	/// Returns number of parameters used by the agent
	fn num_params(&self) -> usize;
	/// Returns the parameters used by the agent
	fn get_params(&self) -> Vec<T>;
	/// Changes the parameters used by the agent
	fn set_params(&mut self, params: &Vec<T>);
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
	fn choose(&self, choices: Vec<T>, weights: Vec<f64>) -> T;
}

/// A feature real-valued feature of elements of some state space
pub trait Feature<S: Space> : Debug {
	/// Extracts some real-valued feature from a given state
	fn extract(&self, state: &S::Element) -> f64;
	/// Creates a cloned trait object of self
	fn box_clone(&self) -> Box<Feature<S>>;
}

impl<S: Space> Clone for Box<Feature<S>> {
	fn clone(&self) -> Self {
		self.box_clone()
	}
}

/// A type with a notion of distance
/// The distance function should satisfy the triangle inequality (and the other [metric](https://www.wikiwand.com/en/Metric_(mathematics)) properties)
/// d(x,z) <= d(x,y) + d(y,z)
pub trait Metric {
	/// Returns the distance between x and y
	fn dist(x: &Self, y: &Self) -> f64 {
		Metric::dist2(x, y).sqrt()
	}
	/// Returns the squared distance between x and y
	fn dist2(x: &Self, y: &Self) -> f64;
}