use num::Float;
use num::cast::NumCast;

use environment::{Space, FiniteSpace, Environment};

use trainer::EpisodicTrainer;

use agent::Agent;
use agent::PolicyAgent;

use util::{DifferentiableFunc, GradientDescAlgo};
use util::TimePeriod;

use stat::normalize;

/// A variation of the [Vanilla Policy Gradient](https://youtu.be/PtAIh9KSnjo?t=2590) algorithm
///
/// Instead of using a baseline, rewards are normalized to mean 0 and variance 1
#[derive(Debug)]
pub struct PolicyGradient<F: Float, G: GradientDescAlgo<F>, A: FiniteSpace> {
	/// Action Space 
	action_space: A,
	/// Gradient descent algorithm
	grad_desc: G,
	/// Discount factor
	gamma: f64,
	/// Learning rate
	lr: F,
	/// Number of training iterations to perform when calling `train`
	iters: usize,
	/// Time period to evaluate each parameter sample on
	eval_period: TimePeriod,
}

// Sometimes I wonder if I'm using traits how they were meant to be used, because this just looks ugly
// Honestly, even disregarding the trait boilerplate, this whole implementation is pretty messy
impl<F: Float, S: Space, A: FiniteSpace, G, D> EpisodicTrainer<S, A, PolicyAgent<F, S, A, D>> for PolicyGradient<F, G, A>
	where D: DifferentiableFunc<S, A, F>,
		  G: GradientDescAlgo<F> {
	fn train_step(&mut self, agent: &mut PolicyAgent<F, S, A, D>, env: &mut Environment<State=S, Action=A>) {
		let (xs, ys, mut rs) = self.collect_trajectory(agent, env);
		if rs.len() > 0 {
			normalize(&mut rs);

			let mut grad = vec![F::zero(); agent.log_func.num_params()];
			for i in 0..rs.len() {
				let g = agent.log_func.get_grad(&xs[i], &ys[i]);
				let r = NumCast::from(rs[i]).unwrap();

				for j in 0..g.len() {
					grad[j] = grad[j] + g[j] * r;
				}
			}

			let mut params = agent.log_func.get_params();
			let grad_step = self.grad_desc.calculate(grad, self.lr);
			for i in 0..params.len() {
				//println!("grad_step[{}] = {}", i, grad_step[i].to_f64().unwrap());
				params[i] = params[i] + grad_step[i];
			}
			//println!("");

			agent.log_func.set_params(params);
		}
	}
	fn train(&mut self, agent: &mut PolicyAgent<F, S, A, D>, env: &mut Environment<State=S, Action=A>) {
		for _ in 0..self.iters {
			self.train_step(agent, env);		
		}
	}
}

// Are these even good default values?
impl<G: GradientDescAlgo<f64>, A: FiniteSpace> PolicyGradient<f64, G, A> {
	/// Creates a PolicyGradient with default parameter values and given action space and gradient descent algorithm
	pub fn default(action_space: A, grad_desc: G) -> PolicyGradient<f64, G, A> {
		PolicyGradient {
			action_space: action_space,
			grad_desc: grad_desc,
			gamma: 0.99,
			lr: 0.005,
			iters: 100,
			eval_period: TimePeriod::EPISODES(10)
		}
	}
}

impl<F: Float, G: GradientDescAlgo<F>, A: FiniteSpace> PolicyGradient<F, G, A> {
	/// Constructs a new PolicyGradient with given information
	pub fn new(action_space: A, grad_desc: G, gamma: f64, lr: F, iters: usize, eval_period: TimePeriod) -> PolicyGradient<F, G, A> {
		assert!(0.0 < gamma && gamma <= 1.0, "gamma must be between 0 and 1");
		assert!(F::zero() < lr && lr <= F::one(), "learning rate must be between 0 and 1");

		PolicyGradient {
			action_space: action_space,
			grad_desc: grad_desc,
			gamma: gamma,
			lr: lr,
			iters: iters,
			eval_period: eval_period
		}
	}
	/// Updates gamma field of self
	pub fn gamma(mut self, gamma: f64) -> PolicyGradient<F, G, A> {
		assert!(0.0 <= gamma && gamma <= 1.0, "gamma must be between 0 and 1");

		self.gamma = gamma;
		self
	}
	/// Updates lr field of self
	pub fn lr(mut self, lr: F) -> PolicyGradient<F, G, A> {
		assert!(F::zero() <= lr && lr <= F::one(), "lr must be between 0 and 1");

		self.lr = lr;
		self
	}
	/// Updates iters field of self
	pub fn iters(mut self, iters: usize) -> PolicyGradient<F, G, A> {
		self.iters = iters;
		self
	}
	/// Updates eval_period field of self
	pub fn eval_period(mut self, eval_period: TimePeriod) -> PolicyGradient<F, G, A> {
		self.eval_period = eval_period;
		self
	}

	fn discount(&self, mut rewards: Vec<f64>) -> Vec<f64> {
		let mut running_sum = 0.0;
		for t in (0..rewards.len()).rev() {
			running_sum = running_sum * self.gamma + rewards[t];
			rewards[t] = running_sum;
		}
		return rewards;
	}
	fn collect_trajectory<S, D>(&self, agent: &PolicyAgent<F, S, A, D>, env: &mut Environment<State=S, Action=A>) -> (Vec<S::Element>, Vec<A::Element>, Vec<f64>)
		where S: Space,
			  D: DifferentiableFunc<S, A, F> {
		let (mut states, mut actions, mut rewards) = if let TimePeriod::TIMESTEPS(len) = self.eval_period.clone() {
			(Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len))
		} else {
			(Vec::new(), Vec::new(), Vec::new())
		};

		let mut ep_rewards = Vec::new();

		let mut obs = env.reset();
		let mut time_remaining = self.eval_period.clone();
		while !time_remaining.is_none() {
			let action = agent.get_action(&obs.state);
			let new_obs = env.step(&action);

			states.push(obs.state);
			actions.push(action);
			ep_rewards.push(obs.reward);

			time_remaining = time_remaining.dec(new_obs.done);
			obs = if new_obs.done {
				ep_rewards = self.discount(ep_rewards);
				rewards.extend_from_slice(&ep_rewards);

				ep_rewards.clear();
				env.reset()
			} else {new_obs};
		}

		rewards.extend_from_slice(&self.discount(ep_rewards));
		(states, actions, rewards)
	}
}