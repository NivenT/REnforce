use std::marker::PhantomData;

use num::Float;

use environment::{Space, FiniteSpace};

use agent::Agent;

use util::{LogDiffFunc, DifferentiableFunc, ParameterizedFunc};
use util::Chooser;
use util::chooser::Softmax;

/// Policy Agent
///
/// Explicitly stores a stochastic policy as the softmax of some differentiable function
#[derive(Debug, Clone)]
pub struct PolicyAgent<F: Float, S: Space, A: FiniteSpace, D: DifferentiableFunc<S, A, F>> { // TODO: Think of a better name
	/// The function used by this agent to calculate weights passed into Softmax
	log_func: D,
	/*
	/// The space the agent draws its actions from
	action_space: A, 
	*/
	/// All the actions performable by this agent
	actions: Vec<A::Element>,
	/// Temperature of associated softmax
	temp: F,
	phant: PhantomData<S>,
}

/*
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
*/

impl<F: Float, S: Space, A: FiniteSpace, D> ParameterizedFunc<F> for PolicyAgent<F, S, A, D> 
	where D: DifferentiableFunc<S, A, F> {
	fn num_params(&self) -> usize {
		self.log_func.num_params()
	}
	fn get_params(&self) -> Vec<F> {
		self.log_func.get_params()
	}
	fn set_params(&mut self, params: Vec<F>) {
		self.log_func.set_params(params)
	}
}

/* TODO: Implement this
impl<F: Float, S: Space, A: FiniteSpace, D> DifferentiableFunc<S, A, F> for PolicyAgent<F, S, A, D>
	where D: DifferentiableFunc<S, A, F> {
	fn get_grad(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
		Vec::new()
	}
	fn calculate(&self, state: &S::Element, action: &A::Element) -> F {
		F::zero()
	}
}
*/

impl<F: Float, S: Space, A: FiniteSpace, D> LogDiffFunc<S, A, F> for PolicyAgent<F, S, A, D>
	where D: DifferentiableFunc<S, A, F> {
	fn log_grad(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
		// Not 100% sure either of these are correct
		self.log_func.get_grad(state, action)//self.calc_log_grad(state, action)
	}
}

impl<F: Float, S: Space, A: FiniteSpace, D> Agent<S, A> for PolicyAgent<F, S, A, D>
	where D: DifferentiableFunc<S, A, F> {
	fn get_action(&self, state: &S::Element) -> A::Element {
		let mut weights = Vec::with_capacity(self.actions.len());
		for a in &self.actions {
			weights.push(self.log_func.calculate(state, a).to_f64().unwrap());
		}

		Softmax::new(self.temp.to_f64().unwrap()).choose(&self.actions, weights)
	}
}

impl<S: Space, A: FiniteSpace, D: DifferentiableFunc<S, A, f64>> PolicyAgent<f64, S, A, D> {
	/// Creates a new PolicyAgent with temperature 1.0 used in Softmax
	pub fn default(action_space: A, log_func: D) -> PolicyAgent<f64, S, A, D> {
		PolicyAgent {
			log_func: log_func,
			actions: action_space.enumerate(),
			//action_space: action_space,
			temp: 1.0,
			phant: PhantomData
		}
	}
}

impl<F: Float, S: Space, A: FiniteSpace, D: DifferentiableFunc<S, A, F>> PolicyAgent<F, S, A, D> {
	/// Creates a new PolicyAgent with given parameters
	pub fn new(action_space: A, log_func: D, temp: F) -> PolicyAgent<F, S, A, D> {
		PolicyAgent {
			log_func: log_func,
			actions: action_space.enumerate(),
			//action_space: action_space,
			temp: temp,
			phant: PhantomData
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
	/// Calculates the derivative of the log of this function
	// Can probably be calculated more efficiently
	// This function is correct assuming I correctly worked out the gradient
	// Would not put all my trust in it
	pub fn calc_log_grad(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
		let mut total = F::zero();

		let weights = self.actions.iter()
								  .map(|a| {
								  	let w = (self.log_func.calculate(state, a)/self.temp).exp();
								  	total = total + w;
								  	w
								  })
								  .collect::<Vec<_>>();

		let mut index = 0;
		let mut grad = vec![F::zero(); self.log_func.num_params()];
		for i in 0..self.actions.len() {
			if self.actions[i] != *action {
				let g = self.log_func.get_grad(state, &self.actions[i]);
				for j in 0..g.len() {
					grad[j] = grad[j] + g[j] * weights[i];
				}
			} else {
				index = i;
			}
		}

		let coeff = weights[index] - total;
		let g = self.log_func.get_grad(state, action);
		for i in 0..g.len() {
			grad[i] = -(grad[i] + coeff * g[i])/(total*self.temp);
		}

		grad
	}
}

