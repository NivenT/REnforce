//! Environment Module

pub mod finite;
mod empty;
mod tuple;

use std::fmt::Debug;

/// A transition experienced by the agent
pub type Transition<S: Space, A: Space> = (S::Element, A::Element, f64, S::Element);

/// Space Trait
///
/// Represents a {State, Action} Space
pub trait Space : Debug {
	//Should we require Copy?
	/// The type of members of the Space
	type Element : Debug + PartialEq + Clone + Copy;

	/// Returns a random element of this space
	fn sample(&self) -> Self::Element;
}

/// Finite Space Trait
///
/// Represents a space with finitely many members
pub trait FiniteSpace : Space {
	/// Returns a vector of all elements of this space
	fn enumerate(&self) -> Vec<Self::Element>;
}

/// Observation
///
/// Stores the information returned by the environment
#[derive(Debug, Clone)]
pub struct Observation<S: Space> {
	/// The state of the environment
	pub state: 	S::Element,
	/// The reward received by the agent
	pub reward: f64,
	/// Whether or not the episode has finished
	pub done: 	bool,
}

/// Environment Trait
///
/// Represents an interactive environment
pub trait Environment {
	/// The type of State Space used by this Environment	
	type State : Space;
	/// The type of Action Space used by this Environment
	type Action : Space;

	/// Performs action in environment and returns the observed result
	fn step(&mut self, action: <Self::Action as Space>::Element) -> Observation<Self::State>;
	/// Resets the environment to its initial configuration
	fn reset(&mut self) -> Observation<Self::State>;
	/// Displays the environment
	fn render(&self);
}