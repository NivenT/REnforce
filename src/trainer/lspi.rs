use rand::{Rng, thread_rng};
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use num::Float;
use num::cast::NumCast;

use rulinalg::matrix::{Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};

use environment::{Space, Environment, Transition};

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

