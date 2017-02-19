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

use stat::{mean_var, normalize, shuffle};

/// A variation of the [Vanilla Policy Gradient](https://youtu.be/PtAIh9KSnjo?t=2590) algorithm
///
/// Instead of using a baseline, rewards are normalized to mean 0 and variance 1
#[derive(Debug)]
pub struct PolicyGradient<F: Float> {
	iters: usize,
	gamma: F,
}