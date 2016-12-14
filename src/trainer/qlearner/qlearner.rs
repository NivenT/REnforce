use std::f64;

use environment::Environment;
use environment::Transition;
use environment::{Space, FiniteSpace};
use trainer::OnlineTrainer;
use agent::Agent;
use util::{QFunction, TimePeriod};

/// Represents an OnlineTrainer for Q-functions
/// Uses the [Q-learning algorithm](https://www.wikiwand.com/en/Q-learning)
#[derive(Debug)]
pub struct QLearner<A: FiniteSpace> {
	/// The action space used by the agent
	action_space: A,
	/// The discount factor
	gamma: f64,
	/// The learning rate
	alpha: f64,
	/// The time period to train agent on when calling train
	train_period: TimePeriod,
}

impl<T, S: Space, A: FiniteSpace> OnlineTrainer<S, A, T> for QLearner<A>
	where T: QFunction<S, A> + Agent<S, A> {
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition;
		
		let mut max_next_val = f64::MIN;
		for a in self.action_space.enumerate() {
			max_next_val = max_next_val.max(agent.eval(&next, &a));
		}
		agent.update(state, action, reward + self.gamma*max_next_val, self.alpha);
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut obs = env.reset();
		let mut time_remaining = self.train_period;
		while !time_remaining.is_none() {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);
			self.train_step(agent, (&obs.state, &action, new_obs.reward, &new_obs.state));

			time_remaining = time_remaining.dec(new_obs.done);
			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl<A: FiniteSpace> QLearner<A> {
	/// Returns a new QLearner with the given info
	pub fn new(action_space: A, gamma: f64, alpha: f64, train_period: TimePeriod) -> QLearner<A> {
		QLearner {
			action_space: action_space,
			gamma: gamma,
			alpha: alpha,
			train_period: train_period
		}
	}
	/// Creates a new QLearner with default gamma, alpha, and train_period
	pub fn default(action_space: A) -> QLearner<A> {
		QLearner {
			action_space: action_space,
			gamma: 0.95,
			alpha: 0.1,
			train_period: TimePeriod::EPISODES(100)
		}
	}
	/// Sets gamma field of self
	pub fn gamma(mut self, gamma: f64) -> QLearner<A> {
		self.gamma = gamma;
		self
	}
	/// Sets alpha field of self
	pub fn alpha(mut self, alpha: f64) -> QLearner<A> {
		self.alpha = alpha;
		self
	}
	/// Sets train_period field of self
	pub fn train_period(mut self, train_period: TimePeriod) -> QLearner<A> {
		self.train_period = train_period;
		self
	}
}