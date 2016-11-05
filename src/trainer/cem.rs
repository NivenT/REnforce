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
use stat::mean_var;

/// Cross Entropy method for parameter selection
#[derive(Debug)]
pub struct CrossEntropy<F: Float> {
	/// Percent of top samples to use for Gaussian fit
	elite: f64,
	/// Number of samples to take
	num_samples: usize,
	/// Time period to evaluate each parameter sample on
	eval_period: usize,
	/// Number of training iterations to perform when calline `train`
	iters: usize,
	phantom: PhantomData<F>,
}

impl<F: Float, S: Space, A: Space, T> OnlineTrainer<S, A, T> for CrossEntropy<F>
	where T: Agent<S, A> + ParameterizedFunc<F> {
	fn train_step(&mut self, _: &mut T, _: Transition<S, A>) {
		panic!("Cross Entropy can't be used to train on a sinple transition");
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut rng = thread_rng();
		let mut mean_params = agent.get_params();
		let mut deviation: Vec<F> = (0..mean_params.len()).map(|_| {
			let one = F::one().to_f64().unwrap();
			NumCast::from(rng.gen_range(-one, one)).unwrap()
		}).collect();
		
		for _ in 0..self.iters {
			let samples: Vec<Vec<F>> = (0..self.num_samples).map(|_| {
				(0..mean_params.len()).map(|i| {
					let normal = Normal::new(mean_params[i].to_f64().unwrap(),
											 deviation[i].to_f64().unwrap());
					NumCast::from(normal.ind_sample(&mut rng)).unwrap()
				}).collect()
			}).collect();

			let mut scored_samples: Vec<_> = samples.into_iter()
													.map(|s| (self.eval(s.clone(), agent, env), s))
													.collect();
			scored_samples.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().reverse());

			let num_keep = (self.elite * self.num_samples as f64).floor() as usize;
			scored_samples = scored_samples[..num_keep].to_vec();

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
			num_samples: 10,
			eval_period: 1000,
			iters: 10,
			phantom: PhantomData
		}
	}
}

impl<F: Float> CrossEntropy<F> {
	/// Constructs a new CrossEntropy with randomly initialized mean and deviation
	pub fn new(elite: f64, num_samples: usize, eval_period: usize, iters: usize) -> CrossEntropy<F> {
		assert!(0.0 <= elite && elite <= 1.0, "elite must be between 0 and 1");

		CrossEntropy {
			elite: elite,
			num_samples: num_samples,
			eval_period: eval_period,
			iters: iters,
			phantom: PhantomData
		}
	}
	fn eval<T, S, A>(&self, params: Vec<F>, agent: &mut T, env: &mut Environment<State=S, Action=A>) -> F 
		where 	S: Space,
				A: Space,
				T: Agent<S, A> + ParameterizedFunc<F> {
		agent.set_params(params);

		let mut obs = env.reset();
		let mut reward = 0.0;
		for _ in 0..self.iters {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);

			reward += new_obs.reward;
			obs = if new_obs.done {env.reset()} else {new_obs};
		}
		NumCast::from(reward).unwrap()
	}
}