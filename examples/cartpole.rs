extern crate renforce as re;
extern crate gym;

use std::io::stdin;

use re::environment::{Environment, Observation};
use re::environment::{Finite, Range};

use re::agent::{Agent, OnlineTrainer};
use re::agent::vagents::BinaryVAgent;
use re::agent::vlearner::BinaryVLearner;

use re::util::approx::VLinear;
use re::util::feature::*;

use gym::Client;

static mut render_cartpole: bool = false;

struct CartPole {
	pub env: gym::Environment
}

impl Environment for CartPole {
	type State = Vec<Range>;
	type Action = Vec<Finite>;

	fn step(&mut self, action: &Vec<u32>) -> Observation<Vec<Range>> {
		let obs = unsafe {
			self.env.step(vec![action[0] as f64], render_cartpole).unwrap()
		};
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
	fn render(&self) {
	}
}

impl CartPole {
	fn new() -> CartPole {
		let client = Client::new("http://localhost:5000".to_string());
		let mut env = match client.make("CartPole-v0") {
			Ok(env) => env,
			Err(msg) => panic!("Could not make environment because of error:\n{}", msg)
		};
		let _ = env.reset();
		CartPole {env: env}
	}
}

fn main() {
	// The agent has 2 actions: move {left, right}
	let action_space = vec![Finite::new(2)];

	let mut v_func = VLinear::new();
	let num_ranges = 20;
	for d in 0..4 {
		for n in 0..num_ranges {
			let num_ranges = num_ranges as f64;
			v_func = v_func.add_feature(Box::new(BSFeature::new(-3.0 + 6.0*(n as f64)/num_ranges,
																-3.0 + 6.0*(n + 1) as f64/num_ranges,
																d)));
		}
	}

	// Creates an epsilon greedy Q-agent
	// Agent will use softmax to act randomly 5% of the time
	let mut agent = BinaryVAgent::new(Box::new(v_func), action_space.clone());

	let mut env = CartPole::new();

	let mut trainer = BinaryVLearner::new(0.7, 0.1, 5000);

	println!("Training...");
	trainer.train(&mut agent, &mut env);
	println!("Done training");

	unsafe {
		render_cartpole = true;
	}
	let _ = stdin().read_line(&mut String::new());

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	let mut reward = 0.0;
	while !obs.done {
		env.render();

		let action = agent.get_action(&obs.state);
		obs = env.step(&action);
		reward += obs.reward;
		println!("action: {:?}", action);
	}
	env.render();
	println!("total reward: {}", reward);

}