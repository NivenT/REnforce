use std::collections::HashSet;
use std::hash::Hash;
use std::f64;

use rand::{thread_rng, Rng};

use environment::Environment;
use environment::Transition;
use environment::{Space, FiniteSpace};
use trainer::OnlineTrainer;
use model::Model;
use agent::Agent;
use util::{QFunction, TimePeriod, Chooser};
use util::chooser::Weighted;

/// Represents an OnlineTrainer for Q-functions
/// Uses the Dyna-Q algorithm
#[derive(Debug)]
pub struct DynaQ<S: Space, A: FiniteSpace, M: Model<S, A>> where
	S::Element: Hash + Eq, A::Element: Hash + Eq {
	/// All the possible performable actions
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
	states: HashSet<S::Element>,
	/// The actions performed by the agent
	actions: HashSet<A::Element>,
	model: M,
}

impl<T, S: Space, A: FiniteSpace, M: Model<S, A>> OnlineTrainer<S, A, T> for DynaQ<S, A, M>
	where T: QFunction<S, A> + Agent<S, A>,
		  S::Element: Hash + Eq,
		  A::Element: Hash + Eq {
	// Possibly improperly implemented
	// Updating Q maybe should make heavier use of model
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition.clone();
		
		let mut max_next_val = f64::MIN;
		for a in &self.all_actions {
			max_next_val = max_next_val.max(agent.eval(&next, a));
		}

		agent.update(&state, &action, reward + self.gamma*max_next_val, self.alpha);
		self.model.update(transition);

		self.states.insert(state.clone());
		self.states.insert(next.clone());
		self.actions.insert(action.clone());

		let mut rng = thread_rng();
		for _ in 0..self.num_samples {
			let state_index = rng.gen_range(0, self.states.len());
			let s0 = &self.states.iter().skip(state_index).next().unwrap();

			let action_index = rng.gen_range(0, self.actions.len());
			let a = &self.actions.iter().skip(action_index).next().unwrap();

			let (r, s1) = self.sample_model(s0, a);

			let mut max_next_val = f64::MIN;
			for a in &self.all_actions {
				max_next_val = max_next_val.max(agent.eval(s1, a));
			}

			agent.update(s0, a, r + self.gamma*max_next_val, self.alpha);			
		}
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut obs = env.reset();
		let mut time_remaining = self.train_period.clone();
		while !time_remaining.is_none() {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);
			self.train_step(agent, (obs.state, action, new_obs.reward, new_obs.state.clone()));

			time_remaining = time_remaining.dec(new_obs.done);
			obs = if new_obs.done {env.reset()} else {new_obs};
		}
	}
}

impl<S: Space, A: FiniteSpace, M: Model<S, A>> DynaQ<S, A, M> 
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	/// Returns a new DynaQ with the given parameters
	pub fn new(action_space: A, gamma: f64, alpha: f64, train_period: TimePeriod, 
				num_samples: usize, model: M) -> DynaQ<S, A, M> {
		DynaQ {
			all_actions: action_space.enumerate(),
			gamma: gamma,
			alpha: alpha,
			train_period: train_period,
			num_samples: num_samples,
			states: HashSet::new(),
			actions: HashSet::new(),
			model: model
		}
	}
	/// Creates a new DynaQ with default parameters
	pub fn default(action_space: A, model: M) -> DynaQ<S, A, M> {
		DynaQ {
			all_actions: action_space.enumerate(),
			gamma: 0.95,
			alpha: 0.1,
			train_period: TimePeriod::EPISODES(30),
			num_samples: 10,
			states: HashSet::new(),
			actions: HashSet::new(),
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

	fn sample_model(&self, s: &S::Element, a: &A::Element) -> (f64, &S::Element) {
		let weights = self.states.iter().map(|s1| {
			self.model.transition(s, a, s1)
		}).collect::<Vec<_>>();
		let states: Vec<_> = self.states.iter().collect();

		let chooser = Weighted;
		let s1 = chooser.choose(states, weights);
		let r = self.model.reward(s, a, s1);
		(r, s1)
	}
}