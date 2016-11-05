use rand::{Rng, thread_rng};
use rand::distributions::IndependentSample;
use rand::distributions::range::SampleRange;
use rand::distributions::normal::Normal;
use num::Float;
use num::cast::NumCast;

use environment::{Space, Transition, Environment};
use trainer::OnlineTrainer;
use agent::ParameterizedAgent;

/// Cross Entropy method for parameter selection
#[derive(Debug)]
pub struct CrossEntropy<T: Float> {
	/// Mean value of Gaussian fitted to good parameter choices
	mean: Vec<T>,
	/// standard deviation value of Gaussian
	deviation: Vec<T>,
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
		let mut rng = thread_rng();
		let samples: Vec<Vec<F>> = (0..self.num_samples).map(|_| {
			(0..self.mean.len()).map(|i| {
				let normal = Normal::new(self.mean[i].to_f64().unwrap(),
										 self.deviation[i].to_f64().unwrap());
				NumCast::from(normal.ind_sample(&mut rng)).unwrap()
			}).collect()
		}).collect();

		let mut scored_samples: Vec<_> = samples.into_iter().map(|s| (self.eval(&s), s)).collect();
		scored_samples.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().reverse());

		let num_keep = (self.elite * self.num_samples as f64).floor() as usize;
		scored_samples = scored_samples[..num_keep].to_vec();

		
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {

	}
}

impl<T: Float + SampleRange> CrossEntropy<T> {
	/// Constructs a new CrossEntropy with randomly initialized mean and deviation
	pub fn new(num_params: usize, elite: f64, num_samples: usize, eval_period: usize, iters: usize) -> CrossEntropy<T> {
		let mut rng = thread_rng();
		CrossEntropy {
			mean: (0..num_params).map(|_| rng.gen_range(-T::one(),T::one())).collect(),
			deviation: (0..num_params).map(|_| rng.gen_range(-T::one(),T::one())).collect(),
			elite: elite,
			num_samples: num_samples,
			eval_period: eval_period,
			iters: iters
		}
	}
}

impl<T: Float> CrossEntropy<T> {
	fn eval(&self, params: &Vec<T>) -> T {
		NumCast::from(0.0).unwrap()
	}
}