use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use num::Float;
use num::cast::NumCast;

use environment::{Space, Environment};

use trainer::EpisodicTrainer;

use agent::Agent;

use util::ParameterizedFunc;
use util::TimePeriod;

use stat::normalize;

/// [Natural](https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d) [Evolution Strategies](https://blog.openai.com/evolution-strategies/)
#[derive(Debug)]
pub struct NaturalEvo<F: Float> {
	/// Learning rate
	alpha: F,
	/// The mean of the gaussian
	mean_params: Vec<F>,
	/// The standard deviation of the guassian
	// Should this be a Vec<_>?
	deviation: F,
	/// Number of samples to take
	num_samples: usize,
	/// Time period to evaluate each parameter sample on
	eval_period: TimePeriod,
	/// Number of training iterations to perform when calling `train`
	iters: usize,
}

impl<F: Float, S: Space, A: Space, T> EpisodicTrainer<S, A, T> for NaturalEvo<F>
	where T: Agent<S, A> + ParameterizedFunc<F> {
	fn train_step(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		let mut rng = thread_rng();

		if self.mean_params.is_empty() {
			self.mean_params = agent.get_params();
		}

		let normals: Vec<_> = (0..self.mean_params.len()).map(|i| {
			Normal::new(self.mean_params[i].to_f64().unwrap(), 
						self.deviation.to_f64().unwrap())
		}).collect();

		let samples: Vec<Vec<F>> = (0..self.num_samples).map(|_| {
			normals.iter().map(|&distro| {
				NumCast::from(distro.ind_sample(&mut rng)).unwrap()
			}).collect()
		}).collect();

		let mut scores: Vec<_> = samples.iter()
										.map(|s| self.eval(s.clone(), agent, env))
										.collect();
		normalize(&mut scores);

		for d in 0..self.mean_params.len() {
			let mut delta = F::zero();
			for i in 0..self.num_samples {
				delta = delta + scores[i]*samples[i][d];
			}

			let size: F = NumCast::from(self.num_samples).unwrap();
			self.mean_params[d] = self.mean_params[d] + self.alpha*delta/(size * self.deviation);
		}

		agent.set_params(self.mean_params.clone());
	}
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>) {
		for _ in 0..self.iters {
			self.train_step(agent, env);		
		}
	}
}


impl Default for NaturalEvo<f64> {
	/// Creates a new NaturalEvo with some default values
	fn default() -> NaturalEvo<f64> {
		NaturalEvo {
			alpha: 0.001,
			mean_params: Vec::new(),
			deviation: 0.1,
			num_samples: 100,
			eval_period: TimePeriod::EPISODES(1),
			iters: 10
		}
	}
}

impl<F: Float> NaturalEvo<F> {
	/// Constructs a new NaturalEvo
	pub fn new(alpha: F, deviation: F, num_samples: usize, eval_period: TimePeriod, iters: usize) -> NaturalEvo<F> {
		assert!(F::zero() <= alpha && alpha <= F::one(), "elite must be between 0 and 1");
		assert!(deviation > F::zero(), "deviation must be greater than 0");

		NaturalEvo {
			alpha: alpha,
			mean_params: Vec::new(),
			deviation: deviation,
			num_samples: num_samples,
			eval_period: eval_period,
			iters: iters
		}
	}
	/// Updates alpha field of self
	pub fn alpha(mut self, alpha: F) -> NaturalEvo<F> {
		self.alpha = alpha;
		self
	}
	/// Updates deviation field of self
	pub fn deviation(mut self, deviation: F) -> NaturalEvo<F> {
		self.deviation = deviation;
		self
	}
	/// Updates num_samples field of self
	pub fn num_samples(mut self, num_samples: usize) -> NaturalEvo<F> {
		self.num_samples = num_samples;
		self
	}
	/// Updates eval_period field of self
	pub fn eval_period(mut self, eval_period: TimePeriod) -> NaturalEvo<F> {
		self.eval_period = eval_period;
		self
	}
	/// Updates iters field of self
	pub fn iters(mut self, iters: usize) -> NaturalEvo<F> {
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
		let mut time_remaining = self.eval_period.clone();
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