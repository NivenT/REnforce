//! Utilities Module

pub mod table;
pub mod chooser;
pub mod approx;
pub mod feature;

use environment::Space;

use std::fmt::Debug;

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
pub trait VFunction<S: Space, A: Space> : Debug {
	/// Evaluate the function on the given state
	fn eval(&self, state: &S::Element) -> f64;
}

/// Choose Trait
///
/// Represents a way to randomly choose an element of a list given some weights
pub trait Chooser<T> : Debug {
	/// returns an element of choices
	fn choose(&self, choices: Vec<T>, weights: Vec<f64>) -> T;
}

/// A feature of a state, action pair with a real value
pub trait Feature<S: Space, A: Space> : Debug {
	/// Extracts some real-valued feature from a given state, action pair
	fn extract(&self, state: &S::Element, action: &A::Element) -> f64;
}

/// A feature of a state, action pair with a binary value
pub trait BinaryFeature<S: Space, A: Space> : Debug {
	/// Extracts some binary feature from a given state, action pair
	fn b_extract(&self, state: &S::Element, action: &A::Element) -> bool;
}

impl<S: Space, A: Space> Feature<S, A> for BinaryFeature<S, A> {
	fn extract(&self, state: &S::Element, action: &A::Element) -> f64 {
		return self.b_extract(state, action) as u32 as f64;
	}
}