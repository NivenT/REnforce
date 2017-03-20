// Tests various RL algorithms on simple environment
// These tests are generally fairly unstable which is a sign of poor implementation
// of the algorithms, or a sign of poor execution on this environment (hopefully the
// latter). I put little effort in the way of deliberately choosing good hyperparameters
// for this test. This is mainly just to confirm that some can be learned.

extern crate renforce as re;
extern crate rand;

use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use re::prelude::*;

use re::environment::Finite;

use re::trainer::*;

use re::model::PlainModel;

use re::util::table::QTable;
use re::util::approx::QLinear;
use re::util::chooser::Uniform;
use re::util::graddesc::GradientDesc;

const SOLVED_VALUE: f64 = 9000.0;

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
	pub fn num_arms(&self) -> usize {
		self.arms.len()
	}
}

fn test_env() -> NArmedBandit {
	let mut env = NArmedBandit::new();
	env.add(-7.43655309176, 2.1246500952);
	env.add(3.63386982772, 2.91132515333);
	env.add(-7.97146396603, 1.25623157209);
	env.add(-9.98975239925, 2.6061382877);
	env.add(-4.2958342745, 1.47647452872);
	env.add(-1.41255326365, 0.310501561125);
	env.add(-9.16529827385, 0.516568227624);
	env.add(-4.27497832924, 2.91926988686);
	env.add(-6.96468268963, 0.995747498586);
	env.add(-8.45172614267, 2.58484868519);
	env.add(-9.0064274943, 3.20837645281);
	env.add(-0.694385361059, 2.56132956562);
	env.add(0.655829601661, 2.95985113654);
	env.add(1.96045869416, 0.329262342405);
	env.add(-8.70994778115, 4.96518956329);
	env.add(-5.36724223125, 3.14902029655);
	env.add(16.0081918938, 2.75961525604);
	env.add(-6.0312618391, 0.459148128943);
	env.add(18.1171563576, 1.93440985725);
	env.add(19.8322749821, 0.917940489013);
	env.add(2.26223921448, 0.831387849263);
	env.add(19.2600114708, 1.23406519039);
	env.add(4.53694402425, 0.749525493972);
	env.add(4.34528984251, 0.504336403336);
	env.add(18.5630408545, 3.63891040085);
	env.add(1.73016020823, 4.03907898009);
	env.add(17.1908124882, 1.42829702765);
	env.add(-3.34300831609, 0.849230362386);
	env.add(6.9381693627, 2.2583405271);
	env.add(-9.88611681399, 3.39622288703);
	env.add(-5.8975884947, 2.4567031603);
	env.add(7.03717316564, 4.33865652125);
	env.add(7.51430682603, 1.9009758178);
	env.add(16.8733232455, 1.78652883452);
	env.add(12.6002261563, 4.32907407187);
	env.add(13.6698733159, 4.53501880236);
	env.add(14.0483388949, 4.91356814619);
	env.add(1.68993684918, 0.268164731447);
	env.add(110.3700334743, 4.39645983743);
	env
}

#[test]
fn qlearner_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func = QTable::new();
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);

	let mut trainer = QLearner::default(action_space).train_period(TimePeriod::TIMESTEPS(10000));
	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	agent.set_epsilon(0.05);
	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	println!("Q Learning reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}

#[test]
fn sarsalearner_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func = QTable::new();
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);

	let mut trainer = SARSALearner::default().train_period(TimePeriod::TIMESTEPS(10000));
	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	agent.set_epsilon(0.05);
	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	println!("SARSA reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}

#[test]
fn cem_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func = QLinear::default(&action_space);
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);

	let mut trainer = CrossEntropy::default().eval_period(TimePeriod::TIMESTEPS(5));
	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	agent.set_epsilon(0.05);
	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	println!("CrossEntropy reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}

#[test]
fn dyna_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func = QTable::new();
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);
	let model = PlainModel::new();

	let mut trainer = DynaQ::default(action_space, model).train_period(TimePeriod::TIMESTEPS(500));
	trainer.train(&mut agent, &mut env);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	agent.set_epsilon(0.05);
	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	println!("Dyna-Q reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}

#[test]
fn fqi_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func: QTable<(), Finite> = QTable::new();
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);

	let mut trainer = FittedQIteration::default(action_space);

	// Collect transitions
	let mut transitions = Vec::new();
	for _ in 0..1000 {
		let action = agent.get_action(&());
		let obs = env.step(&action);

		transitions.push(((), action, obs.reward, obs.state));
	}

	trainer.train(&mut agent, transitions);

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	agent.set_epsilon(0.05);
	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	println!("FQI reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}

#[test]
fn lspi_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func: QLinear<f64, (), Finite> = QLinear::default(&action_space);
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.2, Uniform);

	let mut trainer = LSPolicyIteration::default();

	// Collect transitions
	let mut transitions = Vec::new();
	for _ in 0..1000 {
		let action = agent.get_action(&());
		let obs = env.step(&action);

		transitions.push(((), action, obs.reward, obs.state));
	}

	println!("training...");
	trainer.train(&mut agent, transitions);
	println!("finished");

	let mut obs = env.reset();
	let mut iters = 100;
	let mut reward = 0.0;

	agent.set_epsilon(0.05);
	while iters != 0 {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		reward += obs.reward;
		iters -= 1;
	}

	println!("LSPI reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}

#[test]
fn pg_bandit() {
	let mut env = test_env();

	let action_space = Finite::new(env.num_arms() as u32);
	let q_func = QLinear::default(&action_space);
	let mut agent = PolicyAgent::new(action_space, q_func, 0.01);

	let mut trainer = PolicyGradient::default(GradientDesc).eval_period(TimePeriod::TIMESTEPS(500));
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

	println!("PolicyGradient reward: {}", reward);
	assert!(reward >= SOLVED_VALUE);
}