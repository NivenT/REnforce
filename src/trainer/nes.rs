use rand::{Rng, thread_rng};
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use num::Float;
use num::cast::NumCast;

use environment::{Space, Environment};

use trainer::EpisodicTrainer;

use agent::Agent;

use util::ParameterizedFunc;
use util::TimePeriod;

/// [Natural](https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d) [Evolution Strategies](https://blog.openai.com/evolution-strategies/)
#[derive(Debug)]
pub struct NaturalEvo<F: Float> {
	/// Learning rate
	alpha: F,
	/// The mean of the gaussian
	mean_params: Vec<F>,
	/// The standard deviation of the guassian
	deviation: F,
	/// Number of samples to take
	num_samples: usize,
	/// Time period to evaluate each parameter sample on
	eval_period: TimePeriod,
	/// Number of training iterations to perform when calling `train`
	iters: usize,
}
