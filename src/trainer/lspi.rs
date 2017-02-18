use num::Float;
use num::cast::NumCast;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

use environment::{Space, Transition};

use trainer::BatchTrainer;

use agent::Agent;

use util::{ParameterizedFunc, FeatureExtractor};

/// Least-squares Policy Iteration method
/// * Uses LSTD-Q for calculating the Q-function associated with a policy
/// * Only trains linear Q-functions (not currently enforced by library)
#[derive(Debug)]
pub struct LSPolicyIteration<F: Float> {
	/// discount factor
	gamma: F,
}

impl<F: Float + 'static, S: Space, A: Space, T> BatchTrainer<S, A, T> for LSPolicyIteration<F>
	where T: Agent<S, A> + ParameterizedFunc<F> + FeatureExtractor<S, A, F> {
	fn train(&mut self, agent: &mut T, transitions: Vec<Transition<S, A>>) {
		let num_features = agent.num_features();

		let mut mat: Matrix<F> = Matrix::zeros(num_features, num_features);
		let mut vec: Matrix<F> = Matrix::zeros(num_features, 1);

		// Divide by number of transitions in the end for numeric stability
		let num: F = NumCast::from(transitions.len()).unwrap();
		for transition in transitions {
			let (state, action, reward, next) = transition;
			let next_action = agent.get_action(&next);

			let feats = Matrix::new(num_features, 1, agent.extract(&state, &action));
			let next_feats = Matrix::new(num_features, 1, agent.extract(&next, &next_action));
			let feats_t = feats.clone().transpose();
			
			mat += &feats * &(&feats_t - &next_feats.transpose() * self.gamma);

			let reward: F = NumCast::from(reward).unwrap();
			vec += &feats * reward;
		}

		let vec = Vector::new(vec.into_vec());
		// Optimal weights w should satisfy mat * w = vec
		let weights = (&mat / num).solve(&vec / num).unwrap();
		agent.set_params(weights.into_vec());
	}
}

impl<F: Float> Default for LSPolicyIteration<F> {
	/// Creates a new LSPolicyIteration with gamma = 0.99
	fn default() -> LSPolicyIteration<F> {
		LSPolicyIteration {
			gamma: NumCast::from(0.99).unwrap()
		}
	}
}

impl<F: Float> LSPolicyIteration<F> {
	/// Constructs a new LSPolicyIteration with randomly initialized mean and deviation
	pub fn new(gamma: F) -> LSPolicyIteration<F> {
		assert!(F::zero() <= gamma && gamma <= F::one(), "elite must be between 0 and 1");

		LSPolicyIteration {
			gamma: gamma
		}
	}
	/// Updates gamma field of self
	pub fn gamma(mut self, gamma: F) -> LSPolicyIteration<F> {
		assert!(F::zero() <= gamma && gamma <= F::one(), "elite must be between 0 and 1");

		self.gamma = gamma;
		self
	}
}