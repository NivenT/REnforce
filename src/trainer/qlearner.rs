//! QLearner Module

use std::f64;

use environment::Environment;
use environment::Transition;
use environment::{Space, FiniteSpace};

use trainer::OnlineTrainer;

use agent::Agent;

use util::QFunction;

/// QLearner
///
/// Represents an OnlineTrainer for Q-functions
/// Uses the [Q-learning algorithm](https://www.wikiwand.com/en/Q-learning)
#[derive(Debug)]
pub struct QLearner<A: FiniteSpace> {
	/// The action space used by the agent
	action_space: 	A,
	/// The discount factor
	gamma:			f64,
	/// The learning rate
	alpha:			f64,
	/// The number of steps to perform when calling train
	iters:			usize,
}

impl<T, S: Space, A: FiniteSpace> OnlineTrainer<S, A, T> for QLearner<A>
	where T: QFunction<S, A> + Agent<S, A> {
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition;
		
		let mut max_next_val = f64::MIN;
		for a in self.action_space.enumerate() {
			max_next_val = max_next_val.max(agent.eval(&next, &a));
		}
		agent.update(&state, &action, reward + self.gamma*max_next_val, self.alpha);
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut obs = env.reset();
		for _ in 0..self.iters {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);
			self.train_step(agent, (&obs.state, &action, new_obs.reward, &new_obs.state));

			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl<A: FiniteSpace> QLearner<A> {
	/// Returns a new QLearner with the given info
	pub fn new(action_space: A, gamma: f64, alpha: f64, iters: usize) -> QLearner<A> {
		QLearner {
			action_space: action_space,
			gamma: gamma,
			alpha: alpha,
			iters: iters
		}
	}
}

/// SARSALearner
///
/// Represents an OnlineTrainer for Q-functions
/// Uses the [SARSA algorithm](https://www.wikiwand.com/en/State-Action-Reward-State-Action)
#[derive(Debug)]
pub struct SARSALearner {
	/// The discount factor
	gamma:			f64,
	/// The learning rate
	alpha:			f64,
	/// The number of steps to perform when calling train
	iters:			usize,
}

impl<T, S: Space, A: Space> OnlineTrainer<S, A, T> for SARSALearner
	where T: QFunction<S, A> + Agent<S, A> {
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition;
		
		let next_action = agent.get_action(&next);
		let next_val = agent.eval(&next, &next_action);
		agent.update(&state, &action, reward + self.gamma*next_val, self.alpha);
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut obs = env.reset();
		for _ in 0..self.iters {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);
			self.train_step(agent, (&obs.state, &action, new_obs.reward, &new_obs.state));

			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl SARSALearner {
	/// Returns a new SARSALearner with the given info
	pub fn new(gamma: f64, alpha: f64, iters: usize) -> SARSALearner {
		SARSALearner {
			gamma: gamma,
			alpha: alpha,
			iters: iters
		}
	}
}