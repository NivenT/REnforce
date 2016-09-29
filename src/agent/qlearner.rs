use std::f64;

use environment::Environment;
use environment::Transition;
use environment::{Space, FiniteSpace};

use agent::{Agent, OnlineTrainer};

use util::QFunction;

pub struct QLearner<A: FiniteSpace> {
	action_space: 	A,
	gamma:			f64,
	alpha:			f64,
}

impl<T, S: Space, A: FiniteSpace> OnlineTrainer<S, A, T> for QLearner<A>
	where T: QFunction<S, A> + Agent<S, A> {
	fn train_step(&self, agent: &mut T, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition;
		
		let mut max_next_val = f64::MIN;
		for a in self.action_space.enumerate() {
			max_next_val = max_next_val.max(agent.eval(next, a));
		}
		agent.update(state, action, reward + self.gamma*max_next_val, self.alpha);
	}
	fn train(&self, agent: &mut T, env: Box<Environment<State=S, Action=A>>) {
		//TODO
	}
}

impl<A: FiniteSpace> QLearner<A> {
	pub fn new(action_space: A, gamma: f64, alpha: f64) -> QLearner<A> {
		QLearner {
			action_space: action_space,
			gamma: gamma,
			alpha: alpha,
		}
	}
}