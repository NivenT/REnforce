//! Model Module

mod plain;

pub use self::plain::PlainModel;

use environment::{Space, Transition};

// Macro not tested, likely needs to be modified
macro_rules! implement_model_for_deterministicmodel {
    ($name:ident, $state:ident, $action:ident) => {
    	impl<S: $state, A: $action> Model<S, A> for $name<S, A> {
	    fn transition(&self, curr: &S::Element, action: &A::Element, next: &S::Element) -> f64 {
			let actual_next = self.transition2(curr, action);
			if *next == actual_next {1.0} else {0.0}
		}

		fn reward(&self, curr: &S::Element, action: &A::Element, next: &S::Element) -> f64 {
			let actual_next = self.transition2(curr, action);
			if *next == actual_next {self.reward2(curr, action)} else {0.0}
		}

		fn update(&mut self, transition: Transition<S, A>) {
			self.update(transition);
		}
	    	}
    }
}

/// Represents a (nondeterministic) model of an environment
/// The model itself is composed of the transition and reward functions
pub trait Model<S: Space, A: Space> {
	/// Returns the probabilty of moving from curr to next when performing action
	fn transition(&self, curr: &S::Element, action: &A::Element, next: &S::Element) -> f64;
	/// Returns the reward received when moving from curr to next when performing action
	fn reward(&self, curr: &S::Element, action: &A::Element, next: &S::Element) -> f64;
	/// Updates the model using information from the given transition
	fn update(&mut self, transition: Transition<S, A>);
}

/// Represents a deterministic model of an environment
/// When the agent performs a specified action in a specified state, there's only one possible next state
pub trait DeterministicModel<S: Space, A: Space> {
	/// Returns the new state of the agent after performing action in curr
	fn transition2(&self, curr: &S::Element, action: &A::Element) -> S::Element;
	/// Returns the reward received when performing action in curr
	fn reward2(&self, curr: &S::Element, action: &A::Element) -> f64;
	/// Upates the model using information from the given transition
	fn update(&mut self, transition: Transition<S, A>);
}