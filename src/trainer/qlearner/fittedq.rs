use std::f64;

use environment::Transition;
use environment::{Space, FiniteSpace};
use trainer::BatchTrainer;
use agent::Agent;
use util::QFunction;

/// BatchTrainer for Q-functions
/// Uses Fitted Q Iteration 
#[derive(Debug)]
pub struct FittedQIteration<A: FiniteSpace> {
	// Set of all possible actions
	actions: Vec<A::Element>,
	// Discount factor
	gamma: f64,
	// Learning rate
	alpha: f64,
	// Number of times to recalculate Q
	iters: usize,
}

impl<S: Space, A: FiniteSpace, T> BatchTrainer<S, A, T> for FittedQIteration<A>
	where T: QFunction<S, A> + Agent<S, A> {
	fn train(&mut self, agent: &mut T, transitions: Vec<Transition<S, A>>) {
		for _ in 0..self.iters {
			let mut patterns = Vec::with_capacity(transitions.len());
			for &(s0, a, r, s1) in &transitions {
				let mut max_next_val = f64::MIN;
				for a in &self.actions {
					max_next_val = max_next_val.max(agent.eval(&s1, a));
				}

				let target = r + self.gamma*max_next_val;
				patterns.push((s0, a, target));
			}

			for (s, a, q) in patterns {
				agent.update(s, a, q, self.alpha);
			}
		}
	}
}

impl<A: FiniteSpace> FittedQIteration<A> {
	/// Creates a new FittedQIteration with the given parameters
	pub fn new(action_space: A, gamma: f64, alpha: f64, iters: usize) -> FittedQIteration<A> {
		FittedQIteration {
			actions: action_space.enumerate(),
			gamma: gamma,
			alpha: alpha,
			iters: iters
		}
	}
	/// Creates a new FittedQIteration with default parameters
	pub fn default(action_space: A) -> FittedQIteration<A> {
		FittedQIteration {
			actions: action_space.enumerate(),
			gamma: 0.95,
			alpha: 0.1,
			iters: 10,
		}
	}
	/// Sets gamma field of self
	pub fn gamma(mut self, gamma: f64) -> FittedQIteration<A> {
		self.gamma = gamma;
		self
	}
	/// Sets alpha field of self
	pub fn alpha(mut self, alpha: f64) -> FittedQIteration<A> {
		self.alpha = alpha;
		self
	}
	/// Sets iters field of self
	pub fn iters(mut self, iters: usize) -> FittedQIteration<A> {
		self.iters = iters;
		self
	}
}