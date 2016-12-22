use std::f64;

use environment::Environment;
use environment::Transition;
use environment::Space;
use trainer::OnlineTrainer;
use agent::Agent;
use util::{QFunction, TimePeriod};

/// Represents an OnlineTrainer for Q-functions
/// Uses the [SARSA algorithm](https://www.wikiwand.com/en/State-Action-Reward-State-Action)
#[derive(Debug)]
pub struct SARSALearner {
	/// The discount factor
	gamma: f64,
	/// The learning rate
	alpha: f64,
	/// The time period to train agent on when calling train
	train_period: TimePeriod,
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
		let mut time_remaining = self.train_period;
		while !time_remaining.is_none() {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);
			self.train_step(agent, (obs.state, action, new_obs.reward, new_obs.state.clone()));

			time_remaining = time_remaining.dec(new_obs.done);
			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl Default for SARSALearner {
	/// Creates a new SARSALearner with default values for gamma, alpha, and train_period
	fn default() -> SARSALearner {
		SARSALearner {
			gamma: 0.95,
			alpha: 0.1,
			train_period: TimePeriod::EPISODES(100)
		}
	}
}

impl SARSALearner {
	/// Returns a new SARSALearner with the given info
	pub fn new(gamma: f64, alpha: f64, train_period: TimePeriod) -> SARSALearner {
		SARSALearner {
			gamma: gamma,
			alpha: alpha,
			train_period: train_period
		}
	}
	/// Sets gamma field of self
	pub fn gamma(mut self, gamma: f64) -> SARSALearner {
		self.gamma = gamma;
		self
	}
	/// Sets alpha field of self
	pub fn alpha(mut self, alpha: f64) -> SARSALearner {
		self.alpha = alpha;
		self
	}
	/// Sets train_period field of self
	pub fn train_period(mut self, train_period: TimePeriod) -> SARSALearner {
		self.train_period = train_period;
		self
	}
}