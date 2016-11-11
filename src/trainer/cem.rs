use std::marker::PhantomData;

use rand::{Rng, thread_rng};
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;
use num::Float;
use num::cast::NumCast;

use environment::{Space, Transition, Environment};
use trainer::OnlineTrainer;
use agent::Agent;
use util::ParameterizedFunc;
use util::TimePeriod;
use stat::mean_var;

/// Cross Entropy method for parameter selection
#[derive(Debug)]
pub struct CrossEntropy<F: Float> {
	/// Percent of top samples to use for Gaussian fit
	elite: f64,
	/// Number of samples to take
	num_samples: usize,
	/// Time period to evaluate each parameter sample on
	eval_period: TimePeriod,
	/// Number of training iterations to perform when calling `train`
	iters: usize,
	phantom: PhantomData<F>,
}

impl<F: Float, S: Space, A: Space, T> OnlineTrainer<S, A, T> for CrossEntropy<F>
	where T: Agent<S, A> + ParameterizedFunc<F> {
	fn train_step(&mut self, _: &mut T, _: Transition<S, A>) {
		panic!("Cross Entropy can't be used to train on a single transition");
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut rng = thread_rng();
		let mut mean_params = agent.get_params();
		let mut deviation: Vec<F> = (0..mean_params.len()).map(|_| {
			let zero = F::zero().to_f64().unwrap();
			let one = F::one().to_f64().unwrap();
			NumCast::from(rng.gen_range(zero, one)).unwrap()
		}).collect();
		
		let num_keep = (self.elite * self.num_samples as f64).floor() as usize;
		for _ in 0..self.iters {
			let normals: Vec<_> = (0..mean_params.len()).map(|i| {
				Normal::new(mean_params[i].to_f64().unwrap(), 
							deviation[i].to_f64().unwrap())
			}).collect();

			let samples: Vec<Vec<F>> = (0..self.num_samples).map(|_| {
				normals.iter().map(|&distro| {
					NumCast::from(distro.ind_sample(&mut rng)).unwrap()
				}).collect()
			}).collect();

			let mut scored_samples: Vec<_> = samples.into_iter()
													.map(|s| (self.eval(s.clone(), agent, env), s))
													.collect();
			scored_samples.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().reverse());
			let scored_samples = &scored_samples[..num_keep];

			for i in 0..mean_params.len() {
				let dim_i: Vec<F> = scored_samples.iter().map(|s| s.1[i]).collect();
				let (mean, var) = mean_var(&dim_i);

				mean_params[i] = mean;
				deviation[i] = var.sqrt();
			}
		}

		agent.set_params(mean_params);
	}
}

impl<F: Float> Default for CrossEntropy<F> {
	/// Creates a new CrossEntropy with some default values
	fn default() -> CrossEntropy<F> {
		CrossEntropy {
			elite: 0.2,
			num_samples: 100,
			eval_period: TimePeriod::EPISODES(1),
			iters: 10,
			phantom: PhantomData
		}
	}
}

impl<F: Float> CrossEntropy<F> {
	/// Constructs a new CrossEntropy with randomly initialized mean and deviation
	pub fn new(elite: f64, num_samples: usize, eval_period: TimePeriod, iters: usize) -> CrossEntropy<F> {
		assert!(0.0 <= elite && elite <= 1.0, "elite must be between 0 and 1");

		CrossEntropy {
			elite: elite,
			num_samples: num_samples,
			eval_period: eval_period,
			iters: iters,
			phantom: PhantomData
		}
	}
	/// Updates elite field of self
	pub fn elite(mut self, elite: f64) -> CrossEntropy<F> {
		self.elite = elite;
		self
	}
	/// Updates num_samples field of self
	pub fn num_samples(mut self, num_samples: usize) -> CrossEntropy<F> {
		self.num_samples = num_samples;
		self
	}
	/// Updates eval_period field of self
	pub fn eval_period(mut self, eval_period: TimePeriod) -> CrossEntropy<F> {
		self.eval_period = eval_period;
		self
	}
	/// Updates iters field of self
	pub fn iters(mut self, iters: usize) -> CrossEntropy<F> {
		self.iters = iters;
		self
	}
	fn eval<T, S, A>(&self, params: Vec<F>, agent: &mut T, env: &mut Environment<State=S, Action=A>) -> F 
		where 	S: Space,
				A: Space,
				T: Agent<S, A> + ParameterizedFunc<F> {
		agent.set_params(params);

		let mut obs = env.reset();
		let mut reward = 0.0;
		let mut time_remaining = self.eval_period;
		while !time_remaining.is_none() {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);

			reward += new_obs.reward;
			time_remaining = time_remaining.dec(new_obs.done);
			obs = if new_obs.done {env.reset()} else {new_obs};
		}
		NumCast::from(reward).unwrap()
	}
}