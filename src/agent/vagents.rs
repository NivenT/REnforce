//! V-Agents module

use environment::{Space, FiniteSpace};

use agent::Agent;

use util::VFunction;

/// Agent for environments where the action space has 2 members
/// Performs action 0 if state value is negative, and performs action 1 otherwise
#[derive(Debug)]
pub struct BinaryVAgent<S: Space, A: FiniteSpace> {
	/// Underlying value function
	v_func: Box<VFunction<S>>,
	/// Allowable actions
	action_space: A,
}

impl<S: Space, A: FiniteSpace> Agent<S, A> for BinaryVAgent<S, A> {
	fn get_action(&self, state: &S::Element) -> A::Element {
		let val = self.v_func.eval(state);
		let actions = self.action_space.enumerate();
		if val < 0.0 {actions[0].clone()} else {actions[1].clone()}
	}
}

impl<S: Space, A: FiniteSpace> VFunction<S> for BinaryVAgent<S, A> {
	fn eval(&self, state: &S::Element) -> f64 {
		self.v_func.eval(state)
	}
	fn update(&mut self, state: &S::Element, new_val: f64, alpha: f64) {
		self.v_func.update(state, new_val, alpha)
	}
}

impl<S: Space, A: FiniteSpace> BinaryVAgent<S, A> {
	/// Creates a new BinaryVAgent
	pub fn new(v_func: Box<VFunction<S>>, action_space: A) -> BinaryVAgent<S, A> {
		assert_eq!(action_space.enumerate().len(), 2, "BinaryVAgent can only be used on action spaces with 2 elements");
		BinaryVAgent {
			v_func: v_func,
			action_space: action_space
		}
	}
}