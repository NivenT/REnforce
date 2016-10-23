//! VLearner Module

use std::f64;

use environment::Environment;
use environment::Transition;
use environment::{Space, FiniteSpace};

use agent::{Agent, OnlineTrainer};
use agent::vagents::BinaryVAgent;

use util::VFunction;

/// Represents an OnlineTrainer for binary V-functions
/// Uses a SARSA-like algorithm.
///
/// Honestly, I'm not sure if this is a legitimate algorithm to use
/// It was just written to get an example off the ground
#[derive(Debug)]
pub struct BinaryVLearner {
	/// The discount factor
	gamma:			f64,
	/// The learning rate
	alpha:			f64,
	/// The number of steps to perform when calling train
	iters:			usize,
}

impl<S: Space, A: FiniteSpace> OnlineTrainer<S, A, BinaryVAgent<S, A>> for BinaryVLearner {
	fn train_step(&self, agent: &mut BinaryVAgent<S, A>, transition: Transition<S, A>) {
		let (state, _, reward, next) = transition;
		
		let next_val = agent.eval(&next);
		agent.update(&state, reward + self.gamma*next_val, self.alpha);
	}
	fn train(&self, agent: &mut BinaryVAgent<S, A>, env: &mut Environment<State=S, Action=A>) {
		let mut obs = env.reset();
		for _ in 0..self.iters {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);
			self.train_step(agent, (&obs.state, &action, new_obs.reward, &new_obs.state));

			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl BinaryVLearner {
	/// Returns a new BinaryVLearner with the given info
	pub fn new(gamma: f64, alpha: f64, iters: usize) -> BinaryVLearner {
		BinaryVLearner {
			gamma: gamma,
			alpha: alpha,
			iters: iters
		}
	}
}