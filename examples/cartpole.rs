// Here, we train an agent on the classic cartpole problem
// The agent does not exhibit optimal behavior, but it certainly learns something

extern crate renforce as re;
extern crate gym;

use std::io::stdin;

use re::environment::{Environment, Observation};
use re::environment::{Finite, Range};

use re::trainer::EpisodicTrainer;
use re::trainer::CrossEntropy;

use re::agent::Agent;
use re::agent::qagents::EGreedyQAgent;

use re::util::approx::QLinear;
use re::util::chooser::Uniform;
use re::util::feature::BSFeature;

use gym::GymClient;

struct CartPole {
	pub render: bool,

	env: gym::Environment,
}

impl Environment for CartPole {
	type State = Vec<Range>;
	type Action = Finite;

	fn step(&mut self, action: &u32) -> Observation<Vec<Range>> {
		let obs = self.env.step(vec![*action as f64], self.render).unwrap();
		Observation {
			state: obs.observation,
			reward: obs.reward,
			done: obs.done
		}
	}
	fn reset(&mut self) -> Observation<Vec<Range>> {
		let state = self.env.reset().unwrap();
		Observation {
			state: state,
			reward: 0.0,
			done: false
		}
	}
	fn render(&self) {}
}

impl CartPole {
	fn new() -> CartPole {
		let client = GymClient::new("http://localhost:5000".to_string());
		let mut env = match client.make("CartPole-v0") {
			Ok(env) => env,
			Err(msg) => panic!("Could not make environment because of error:\n{}", msg)
		};
		let _ = env.reset();
		CartPole {env: env, render: false}
	}
}

fn main() {
	// The agent has 2 actions: move {left, right}
	let action_space = Finite::new(2);

	let mut q_func = QLinear::new(&action_space);
	let num_ranges: u32 = 50;
	for d in 0..4 {
		for n in 0..num_ranges {
			let num_ranges = num_ranges as f64;
			q_func.add(Box::new(BSFeature::new(-3.0 + 6.0*(n as f64)/num_ranges,
											   -3.0 + 6.0*(n + 1) as f64/num_ranges,
											   d)));
		}
	}

	// Creates an epsilon greedy Q-agent
	// 20% of the time, the agent uniformly chooses a random action
	let mut agent = EGreedyQAgent::new(q_func, action_space.clone(), 0.20, Uniform);

	let mut env = CartPole::new();

	// Train agent using Cross Entropy Method with default parameters
	let mut trainer = CrossEntropy::default();

	println!("Training...");
	trainer.train(&mut agent, &mut env);
	println!("Done training (press enter)");

	let agent = agent.to_greedy();
	env.render = true;
	let _ = stdin().read_line(&mut String::new());

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	let mut reward = 0.0;
	while !obs.done {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);
		reward += obs.reward;
		println!("action: {:?}", action);
	}
	println!("total reward: {}", reward);

}