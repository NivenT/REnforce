// Here, we train an agent on the classic cartpole problem

extern crate renforce as re;
extern crate gym;

use std::io::stdin;

use re::environment::{Environment, Observation};
use re::environment::{Finite, Range};

use re::trainer::EpisodicTrainer;
use re::trainer::CrossEntropy;

use re::agent::Agent;
use re::agent::qagents::EGreedyQAgent;

use re::util::TimePeriod;
use re::util::approx::QLinear;
use re::util::chooser::Uniform;
use re::util::feature::IFeature;

use gym::GymClient;

struct CartPole {
	pub render: bool,

	env: gym::Environment,
}

impl Environment for CartPole {
	type State = Vec<Range>;
	type Action = Finite;

	fn state_space(&self) -> Vec<Range> {
		vec![Range::new(-4.8, 4.8), Range::new(-3.4e38, 3.4e38), Range::new(-4.188e-1, 4.188e1), Range::new(-3.4e38,3.4e38)]
	}
	fn action_space(&self) -> Finite {
		// The agent has 2 actions: move {left, right}
		Finite::new(2)
	}
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
			Err(msg) => panic!("Could not make environment because of error:\n{}\n\nMake sure you have a [gym server](https://github.com/openai/gym-http-api) running.", msg)
		};
		let _ = env.reset();
		CartPole {env: env, render: false}
	}
}

fn main() {
	let mut env = CartPole::new();

	let mut q_func = QLinear::default(&env.action_space());
	for d in 0..4 {
		q_func.add(Box::new(IFeature::new(d)));
	}

	// Creates an epsilon greedy Q-agent
	// 20% of the time, the agent uniformly chooses a random action
	let mut agent = EGreedyQAgent::new(q_func, env.action_space(), 0.20, Uniform);

	// CrossEntropy will evalute agents for 1 episode or 300 time steps
	// whichever comes first
	let tp = {
		use TimePeriod::*;
		OR(Box::new(EPISODES(1)), Box::new(TIMESTEPS(300)))
	};
	// Train agent using Cross Entropy Method with default parameters
	let mut trainer = CrossEntropy::default().eval_period(tp);

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