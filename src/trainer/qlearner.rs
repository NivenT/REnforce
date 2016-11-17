use std::f64;

use rand::{thread_rng, Rng};

use environment::Environment;
use environment::Transition;
use environment::{Space, FiniteSpace};
use trainer::{OnlineTrainer, Model};
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
		agent.update(state, action, reward + self.gamma*next_val, self.alpha);
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

/// Represents an OnlineTrainer for Q-functions
/// Uses the Dyna-Q
#[derive(Debug)]
pub struct DynaQ<S: Space, A: FiniteSpace, M: Model<S, A>> {
	/// External trainer for updating Q 
	all_actions: Vec<A::Element>,
	/// The discount factor
	gamma: f64,
	/// The learning rate
	alpha: f64,
	/// The time period to train agent on when calling train
	train_period: TimePeriod,
	/// The number of (state, action) pairs to sample each train step
	num_samples: usize,
	/// The states observed by the agent
	states: Vec<S::Element>,
	/// The actions performed by the agent
	actions: Vec<A::Element>,
	model: M,
}

impl<T, S: Space, A: FiniteSpace, M: Model<S, A>> OnlineTrainer<S, A, T> for DynaQ<S, A, M>
	where T: QFunction<S, A> + Agent<S, A> {
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition;
		
		let mut max_next_val = f64::MIN;
		for a in &self.all_actions {
			max_next_val = max_next_val.max(agent.eval(&next, a));
		}

		agent.update(state, action, reward + self.gamma*max_next_val, self.alpha);
		self.model.update(transition);

		self.states.push(state.clone());
		self.actions.push(action.clone());

		let mut rng = thread_rng();
		for _ in 0..self.num_samples {
			let s = &self.states[rng.gen_range(0, self.states.len())];
			let a = &self.actions[rng.gen_range(0, self.actions.len())];
		}
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

impl<S: Space, A: FiniteSpace, M: Model<S, A>> DynaQ<S, A, M> {
	/// Returns a new DynaQ with the given parameters
	pub fn new(action_space: A, gamma: f64, alpha: f64, train_period: TimePeriod, 
				num_samples: usize, model: M) -> DynaQ<S, A, M> {
		DynaQ {
			all_actions: action_space.enumerate(),
			gamma: gamma,
			alpha: alpha,
			train_period: train_period,
			num_samples: num_samples,
			states: vec![],
			actions: vec![],
			model: model
		}
	}
	/// Sets gamma field of self
	pub fn gamma(mut self, gamma: f64) -> DynaQ<S, A, M> {
		self.gamma = gamma;
		self
	}
	/// Sets alpha field of self
	pub fn alpha(mut self, alpha: f64) -> DynaQ<S, A, M> {
		self.alpha = alpha;
		self
	}
	/// Sets train_period field of self
	pub fn train_period(mut self, train_period: TimePeriod) -> DynaQ<S, A, M> {
		self.train_period = train_period;
		self
	}
	/// Sets num_samples field of self
	pub fn num_samples(mut self, num_samples: usize) -> DynaQ<S, A, M> {
		self.num_samples = num_samples;
		self
	}

	fn sample_model(&self) -> (f64, S::Element) {
		(0.0, self.states[0].clone())
	}
}