//! Environment Module

mod finite;
mod range;
mod empty;
mod tuple;
mod vector;

use std::fmt::Debug;

pub use self::finite::Finite;
pub use self::range::Range;

/// A transition experienced by the agent (s, a, r, s')
pub type Transition<S: Space, A: Space> = (S::Element, A::Element, f64, S::Element);

/// Space Trait
///
/// Represents a {State, Action} Space
pub trait Space : Debug {
	/// The type of members of the Space
	type Element : Debug + PartialEq + Clone;

	/// Returns a random element of this space
	fn sample(&self) -> Self::Element;
}

/// Finite Space Trait
///
/// Represents a space with finitely many members
pub trait FiniteSpace : Space {
	/// (Determistically) Returns a vector of all elements of this space
	fn enumerate(&self) -> Vec<Self::Element>;

	/// Returns the number of elements in this space
	// should this be called len of count or something else?
	fn size(&self) -> usize {
		self.enumerate().len()
	}
	/// Returns the index of an element in the vector returned by enumerate
	fn index(&self, elm: &Self::Element) -> isize {
		let all = self.enumerate();
		for i in 0..all.len() {
			if all[i] == *elm {
				return i as isize;
			}
		}
		return -1;
	}
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

	/// Returns the state space used by this Environment
	fn state_space(&self) -> Self::State;
	/// Returns the action space used by this Environment
	fn action_space(&self) -> Self::Action;
	/// Performs action in environment and returns the observed result
	fn step(&mut self, action: &<Self::Action as Space>::Element) -> Observation<Self::State>;
	/// Resets the environment to its initial configuration
	fn reset(&mut self) -> Observation<Self::State>;
	/// Displays the environment
	fn render(&self);
}