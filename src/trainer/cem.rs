use num::Float;

use environment::{Space, Transition, Environment};
use trainer::OnlineTrainer;
use agent::ParameterizedAgent;

/// Cross Entropy method for parameter selection
#[derive(Debug)]
pub struct CrossEntropy<T: Float> {
	/// Mean value of Gaussian fitted to good parameter choices
	mean: Vec<T>,
	/// variance value of Gaussian
	variance: Vec<T>,
	/// Percent of top samples to use for Gaussian fit
	elite: f64,
	/// Number of samples to take
	num_samples: usize,
	/// Time period to evaluate each parameter sample on
	eval_period: usize,
	/// Number of training iterations to perform when calline `train`
	iters: usize,
}

impl<T, F: Float, S: Space, A: Space> OnlineTrainer<S, A, T> for CrossEntropy<F>
	where T: ParameterizedAgent<S, A, F> {
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>) {

	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {

	}
}

impl<T: Float> CrossEntropy<T> {

}