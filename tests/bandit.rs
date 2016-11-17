extern crate renforce as re;
extern crate rand;

use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use re::environment::{Environment, Observation};
use re::environment::Finite;
use re::trainer::{OnlineTrainer, EpisodicTrainer};
use re::trainer::{CrossEntropy, SARSALearner, QLearner};
use re::agent::Agent;
use re::agent::qagents::EGreedyQAgent;
use re::util::TimePeriod;
use re::util::table::QTable;
use re::util::approx::QLinear;
use re::util::chooser::Uniform;

struct NArmedBandit {
	arms: Vec<Normal>
}

impl Environment for NArmedBandit {
	type State = ();
	type Action = Finite;

	fn step(&mut self, action: &u32) -> Observation<()> {
		let mut rng = thread_rng();
		let action = *action as usize;
		let reward = if action < self.arms.len() {
			self.arms[action].ind_sample(&mut rng)
		} else {
			0.0
		};
		Observation {
			state: (), 
			reward: reward,
			done: false
		}
	}
	fn reset(&mut self) -> Observation<()> {
		Observation {
			state: (),
			reward: 0.0,
			done: false
		}
	}
	fn render(&self) {
	}
}

impl NArmedBandit {
	pub fn new() -> NArmedBandit {
		NArmedBandit{arms: vec![]}
	}
	pub fn add(&mut self, mean: f64, var: f64) {
		self.arms.push(Normal::new(mean, var));
	}
}

#[test]
fn qlearner_bandit() {
	let action_space = Finite::new(10);
	let q_func = QTable::new();
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);
	let mut trainer = QLearner::default(action_space).train_period(TimePeriod::TIMESTEPS(5000));

	let mut env = NArmedBandit::new();
	env.add(0.0, 1.0);
	env.add(2.0, 2.0);
	env.add(3.0, 3.0);
	env.add(-5.0, 1.0);
	env.add(10.0, 0.4);
	env.add(-3.0, 8.0);
	env.add(11.0, 5.0);
	env.add(7.0, 0.5);
	env.add(-10.0, 2.0);
	env.add(-6.0, 10.0);

	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	assert!(reward >= 500.0);
}

#[test]
fn sarsalearner_bandit() {
	let action_space = Finite::new(10);
	let q_func = QTable::new();
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);
	let mut trainer = SARSALearner::default().train_period(TimePeriod::TIMESTEPS(5000));

	let mut env = NArmedBandit::new();
	env.add(0.0, 1.0);
	env.add(2.0, 2.0);
	env.add(3.0, 3.0);
	env.add(-5.0, 1.0);
	env.add(10.0, 0.4);
	env.add(-3.0, 8.0);
	env.add(11.0, 5.0);
	env.add(7.0, 0.5);
	env.add(-10.0, 2.0);
	env.add(-6.0, 10.0);

	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	assert!(reward >= 500.0);
}

#[test]
fn cem_bandit() {
	let action_space = Finite::new(10);
	let q_func = QLinear::new(&action_space);
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);
	let mut trainer = CrossEntropy::default().eval_period(TimePeriod::TIMESTEPS(100));

	let mut env = NArmedBandit::new();
	env.add(0.0, 1.0);
	env.add(2.0, 2.0);
	env.add(3.0, 3.0);
	env.add(-5.0, 1.0);
	env.add(10.0, 0.4);
	env.add(-3.0, 8.0);
	env.add(11.0, 5.0);
	env.add(7.0, 0.5);
	env.add(-10.0, 2.0);
	env.add(-6.0, 10.0);

	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	assert!(reward >= 500.0);
}