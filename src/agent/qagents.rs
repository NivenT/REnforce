//! Q-Agents module

use std::marker::PhantomData;

use num::Num;
use num::Float;

use rand::{Rng, thread_rng};

use environment::{Space, FiniteSpace};

use agent::Agent;

use util::{ParameterizedFunc, DifferentiableFunc, QFunction, FeatureExtractor};
use util::Chooser;

// TODO: Derive macro?

macro_rules! implement_qfunction {
    () => {
    	fn eval(&self, state: &S::Element, action: &A::Element) -> f64 {
			self.q_func.eval(state, action)
		}
		fn update(&mut self, state: &S::Element, action: &A::Element, new_val: f64, alpha: f64) {
			self.q_func.update(state, action, new_val, alpha)
		}
    }
}

macro_rules! implement_parameterizedfunc {
    () => {
    	fn num_params(&self) -> usize {
			self.q_func.num_params()
		}
		fn get_params(&self) -> Vec<N> {
			self.q_func.get_params()
		}
		fn set_params(&mut self, params: Vec<N>) {
			self.q_func.set_params(params)
		}
    }
}

macro_rules! implement_featureextractor {
    () => {
    	fn num_features(&self) -> usize {
			self.q_func.num_features()
		}
		fn extract(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
			self.q_func.extract(state, action)
		}
	}
}

macro_rules! implement_differentiablefunc {
    () => {
    	fn get_grad(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
			self.q_func.get_grad(state, action)
		}
		fn calculate(&self, state: &S::Element, action: &A::Element) -> F {
			self.q_func.calculate(state, action)
		}
	}
}

/// Greedy Q-Agent
///
/// Represents an agent that only performs the best action according to its QFunction
#[derive(Debug)]
pub struct GreedyQAgent<S: Space, A: FiniteSpace, Q: QFunction<S, A>> {
	/// The underlying QFunction used by the agent
	q_func:	Q,
	/// The agent's action space
	action_space: A,
	phantom: PhantomData<S>,
}

impl<S: Space, A: FiniteSpace, Q: QFunction<S, A>> Agent<S, A> for GreedyQAgent<S, A, Q> {
	fn get_action(&self, state: &S::Element) -> A::Element {
		let actions = self.action_space.enumerate();
		let (mut best_action, mut best_val) = (actions[0].clone(), self.q_func.eval(state, &actions[0]));
		
		for a in actions.into_iter().skip(1) {
			let val = self.q_func.eval(state, &a);
			if val > best_val {
				best_action = a;
				best_val = val;
			}
		}

		best_action
	}
}

impl<S: Space, A: FiniteSpace, Q: QFunction<S, A>> QFunction<S, A> for GreedyQAgent<S, A, Q> {
	implement_qfunction!();
}

impl<N: Num, S: Space, A: FiniteSpace, Q> ParameterizedFunc<N> for GreedyQAgent<S, A, Q>
	where Q: QFunction<S, A> + ParameterizedFunc<N> {
	implement_parameterizedfunc!();
}

impl<F: Float, A: FiniteSpace, S: Space, Q> FeatureExtractor<S, A, F> for GreedyQAgent<S, A, Q>
	where Q: QFunction<S, A> + FeatureExtractor<S, A, F> {
	implement_featureextractor!();
}

impl<F: Float, A: FiniteSpace, S: Space, Q> DifferentiableFunc<S, A, F> for GreedyQAgent<S, A, Q>
	where Q: QFunction<S, A> + DifferentiableFunc<S, A, F> {
	implement_differentiablefunc!();
}

impl<S: Space, A: FiniteSpace, Q: QFunction<S, A>> GreedyQAgent<S, A, Q> {
	/// Returns a new GreedyQAgent with the given function and action space
	pub fn new(q_func: Q, action_space: A) -> GreedyQAgent<S, A, Q> {
		GreedyQAgent {
			q_func: q_func,
			action_space: action_space,
			phantom: PhantomData
		}
	}
	/// Returns an EGreedyQAgent using this agent's Q function
	pub fn to_egreedy<T: Chooser<A::Element>>(self, eps: f64, chooser: T) -> EGreedyQAgent<S, A, Q, T> {
		EGreedyQAgent {
			q_func: self.q_func,
			action_space: self.action_space,
			epsilon: eps,
			chooser: chooser,
			phantom: PhantomData
		}
	}
}

/// Epsilon Greedy Q-Agent
///
/// Represents an agent that acts randomly with probabilty epsilon and
/// acts greedily with probabilty (1 - epsilon)
#[derive(Debug)]
pub struct EGreedyQAgent<S: Space, A: FiniteSpace, Q: QFunction<S, A>, T: Chooser<A::Element>> {
	/// Underlying QFunction
	q_func: Q,
	/// Agent's action space
	action_space: A,
	/// Probabilty of acting randomly
	epsilon: f64,
	/// Method for choosing a random action
	chooser: T,
	phantom: PhantomData<S>,
}

impl<S: Space, A: FiniteSpace, Q, T> Agent<S, A> for EGreedyQAgent<S, A, Q, T>
	where 	T: Chooser<A::Element>,
			Q: QFunction<S, A> {
	fn get_action(&self, state: &S::Element) -> A::Element {
		let mut rng = thread_rng();
		let mut best_action;

		let actions = self.action_space.enumerate();
		if rng.gen_range(0.0, 1.0) < self.epsilon {
			let weights = actions.iter()
								 .map(|a| self.q_func.eval(state, a))
								 .collect();
			best_action = self.chooser.choose(&actions, weights);
		} else {
			let mut best_val = self.q_func.eval(state, &actions[0]);
			
			best_action = actions[0].clone();
			for a in actions.into_iter().skip(1) {
				let val = self.q_func.eval(state, &a);
				if val > best_val {
					best_action = a;
					best_val = val;
				}
			}
		}
		best_action
	}
}

impl<S: Space, A: FiniteSpace, Q, T> QFunction<S, A> for EGreedyQAgent<S, A, Q, T> 
	where 	T: Chooser<A::Element>,
			Q: QFunction<S, A> {
	implement_qfunction!();
}

impl<N: Num, S: Space, A: FiniteSpace, Q, T> ParameterizedFunc<N> for EGreedyQAgent<S, A, Q, T>
	where 	T: Chooser<A::Element>,
			Q: QFunction<S, A> + ParameterizedFunc<N> {
	implement_parameterizedfunc!();
}

impl<F: Float, S: Space, A: FiniteSpace, Q, T> FeatureExtractor<S, A, F> for EGreedyQAgent<S, A, Q, T>
	where T: Chooser<A::Element>,
		  Q: QFunction<S, A> + FeatureExtractor<S, A, F> {
	implement_featureextractor!();
}

impl<F: Float, S: Space, A: FiniteSpace, Q, T> DifferentiableFunc<S, A, F> for EGreedyQAgent<S, A, Q, T> 
	where T: Chooser<A::Element>,
		  Q: QFunction<S, A> + DifferentiableFunc<S, A, F> {
	implement_differentiablefunc!();
}

impl<S: Space, A: FiniteSpace, Q, T> EGreedyQAgent<S, A, Q, T> 
	where	T: Chooser<A::Element>,
			Q: QFunction<S, A> {
	/// Returns a new EGreedyQAgent with the given information
	pub fn new(q_func: Q, action_space: A, epsilon: f64, chooser: T) -> EGreedyQAgent<S, A, Q, T> {
		assert!(0.0 <= epsilon && epsilon <= 1.0, "epsilon must be between 0 and 1");

		EGreedyQAgent {
			q_func: q_func,
			action_space: action_space,
			epsilon: epsilon,
			chooser: chooser,
			phantom: PhantomData
		}
	}
	/// Sets new value for epsilon 
	pub fn set_epsilon(&mut self, ep: f64) {
		self.epsilon = ep;
	}
	/// Returns a GreedyQAgent using this agent's q_function
	pub fn to_greedy(self) -> GreedyQAgent<S, A, Q> {
		GreedyQAgent {
			q_func: self.q_func,
			action_space: self.action_space,
			phantom: PhantomData
		}
	}
}

#[cfg(test)]
mod test {
	use util::table::QTable;
	use util::chooser::Uniform;

	use super::EGreedyQAgent;

	#[test]
	#[should_panic]
	fn egreedy_invalid_epsilon() {
		let q_func: QTable<(), ()> = QTable::new();
		let _ = EGreedyQAgent::new(q_func, (), -0.5, Uniform);
	}
}