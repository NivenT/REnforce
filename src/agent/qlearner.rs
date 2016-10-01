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
	iters:			usize,
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
	fn train(&self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut obs = env.reset();
		for _ in 0..self.iters {
			let action = agent.get_action(obs.state);
			let new_obs = env.step(action);
			self.train_step(agent, (obs.state, action, new_obs.reward, new_obs.state));

			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl<A: FiniteSpace> QLearner<A> {
	pub fn new(action_space: A, gamma: f64, alpha: f64, iters: usize) -> QLearner<A> {
		QLearner {
			action_space: action_space,
			gamma: gamma,
			alpha: alpha,
			iters: iters
		}
	}
}