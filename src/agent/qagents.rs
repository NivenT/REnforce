use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

use agent::Agent;

use util::QFunction;
use util::Chooser;

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

pub struct EGreedyQAgent<S: Space, A: FiniteSpace, T: Chooser<A::Element>> {
	q_func:			Box<QFunction<S,A>>,
	action_space:	A,
	epsilon:		f64,
	chooser:		T,
}

impl<S: Space, A: FiniteSpace, T: Chooser<A::Element>> Agent<S, A> for EGreedyQAgent<S, A, T> {
	fn get_action(&self, state: S::Element) -> A::Element {
		let mut rng = thread_rng();
		let mut best_action;

		let actions = self.action_space.enumerate();
		if rng.gen_range(0.0, 1.0) < self.epsilon {
			let weights = actions.iter()
								 .map(|&a| self.q_func.eval(state, a))
								 .collect();
			best_action = self.chooser.choose(actions, weights);
		} else {
			let mut best_val = self.q_func.eval(state, actions[0]);
			
			best_action = actions[0];
			for a in actions.into_iter().skip(1) {
				let val = self.q_func.eval(state, a);
				if val > best_val {
					best_action = a;
					best_val = val;
				}
			}
		}
		best_action
	}
}

impl<S: Space, A: FiniteSpace, T: Chooser<A::Element>> QFunction<S, A> for EGreedyQAgent<S, A, T> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64 {
		self.q_func.eval(state, action)
	}
	fn update(&mut self, state: S::Element, action: A::Element, new_val: f64, alpha: f64) {
		self.q_func.update(state, action, new_val, alpha)
	}
}

impl<S: Space, A: FiniteSpace, T: Chooser<A::Element>> EGreedyQAgent<S, A, T> {
	pub fn new(q_func: Box<QFunction<S, A>>, action_space: A, epsilon: f64, chooser: T) -> EGreedyQAgent<S, A, T> {
		assert!(0.0 <= epsilon && epsilon <= 1.0, "epsilon must be between 0 and 1");

		EGreedyQAgent {
			q_func: q_func,
			action_space: action_space,
			epsilon: epsilon,
			chooser: chooser
		}
	}
}

#[cfg(test)]
mod test {
	use util::table::QTable;
	use util::chooser::Uniform;

	use super::EGreedyQAgent;

	#[test]
	#[should_panic]
	fn egreedy_invalid_epsilon() {
		let q_func: QTable<(), ()> = QTable::new();
		let _ = EGreedyQAgent::new(Box::new(q_func), (), -0.5, Uniform);
	}
}