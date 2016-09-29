use environment::{Space, FiniteSpace};

use agent::Agent;

use util::QFunction;

pub struct GreedyQAgent<S: Space, A: FiniteSpace> {
	q_func:			Box<QFunction<S, A>>,
	action_space:	A,
}

impl<S: Space, A: FiniteSpace> Agent<S, A> for GreedyQAgent<S, A> {
	fn get_action(&self, state: S::Element) -> A::Element {
		let actions = self.action_space.enumerate();
		let (mut best_action, mut best_val) = (actions[0], self.q_func.eval(state, actions[0]));
		
		for a in actions.into_iter().skip(1) {
			let val = self.q_func.eval(state, a);
			if val > best_val {
				best_action = a;
				best_val = val;
			}
		}

		best_action
	}
}

impl<S: Space, A: FiniteSpace> QFunction<S, A> for GreedyQAgent<S, A> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64 {
		self.q_func.eval(state, action)
	}
	fn update(&mut self, state: S::Element, action: A::Element, new_val: f64, alpha: f64) {
		self.q_func.update(state, action, new_val, alpha)
	}
}

impl<S: Space, A: FiniteSpace> GreedyQAgent<S, A> {
	pub fn new(q_func: Box<QFunction<S, A>>, action_space: A) -> GreedyQAgent<S, A> {
		GreedyQAgent {
			q_func: q_func,
			action_space: action_space,
		}
	}
}