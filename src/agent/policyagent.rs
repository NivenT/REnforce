use std::marker::PhantomData;

use num::Float;

use environment::{Space, FiniteSpace};

use agent::{Agent, StochasticAgent};

use util::DifferentiableFunc;
use util::Chooser;
use util::chooser::Softmax;

/// Policy Agent
///
/// Explicitly stores a stochastic policy as the softmax of some differentiable function
#[derive(Debug, Clone)]
pub struct PolicyAgent<F: Float, S: Space, A: FiniteSpace, D: DifferentiableFunc<S, A, F>> { // TODO: Think of a better name
	/// The function used by this agent to calculate weights passed into Softmax
	pub log_func: D,

	/// The space the agent draws its actions from
	action_space: A, // Should probably just store Vec of actions instead of calling enumerate a lot but meh
	/// Temperature of associated softmax
	temp: F,
	phant1: PhantomData<F>,
	phant2: PhantomData<S>,
}

impl<F: Float, S: Space, A: FiniteSpace, D> StochasticAgent<F, S, A> for PolicyAgent<F, S, A, D>
	where D: DifferentiableFunc<S, A, F> {
	fn get_action_prob(&self, state: &S::Element, action: &A::Element) -> F {
		let mut weights = Vec::with_capacity(self.action_space.size());
		for a in self.action_space.enumerate() {
			weights.push(self.log_func.calculate(state, &a));
		}

		let mut total = F::zero();
		for w in &mut weights {
			*w = (*w/self.temp).exp();
			total = total + *w;
		}

		let index = self.action_space.index(action);
		debug_assert!(index != -1);

		weights[index as usize]/total
	}
}

impl<F: Float, S: Space, A: FiniteSpace, D> Agent<S, A> for PolicyAgent<F, S, A, D>
	where D: DifferentiableFunc<S, A, F> {
	fn get_action(&self, state: &S::Element) -> A::Element {
		let actions = self.action_space.enumerate();
		let mut weights = Vec::with_capacity(self.action_space.size());
		for a in &actions {
			weights.push(self.log_func.calculate(state, a).to_f64().unwrap());
		}

		Softmax::new(self.temp.to_f64().unwrap()).choose(actions, weights)
	}
}

impl<F: Float, S: Space, A: FiniteSpace, D: DifferentiableFunc<S, A, F>> PolicyAgent<F, S, A, D> {
	/// Creates a new PolicyAgent with given parameters
	pub fn new(action_space: A, log_func: D, temp: F) -> PolicyAgent<F, S, A, D> {
		PolicyAgent {
			log_func: log_func,
			action_space: action_space,
			temp: temp,
			phant1: PhantomData,
			phant2: PhantomData
		}
	}
	/// Updates temp field of self
	pub fn temp(mut self, temp: F) -> PolicyAgent<F, S, A, D> {
		self.temp = temp;
		self
	}
	/// Returns temperature used by agent
	pub fn get_temp(&self) -> F {
		self.temp
	}
}

impl<S: Space, A: FiniteSpace, D: DifferentiableFunc<S, A, f64>> PolicyAgent<f64, S, A, D> {
	/// Creates a new PolicyAgent with temperature 1.0 used in Softmax
	pub fn default(action_space: A, log_func: D) -> PolicyAgent<f64, S, A, D> {
		PolicyAgent {
			log_func: log_func,
			action_space: action_space,
			temp: 1.0,
			phant1: PhantomData,
			phant2: PhantomData
		}
	}
}