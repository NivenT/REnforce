// Here, we train an agent on the OpenAI Gym's Acrobot envirnoment

// The trained agent is spastic, but it appears to be performing about as well
// as the most recent submissions to https://gym.openai.com/envs/Acrobot-v1

extern crate renforce as re;
extern crate gym;

use std::io::stdin;

use re::environment::{Finite, Range};

use re::trainer::NaturalEvo;

use re::util::approx::QLinear;
use re::util::chooser::Uniform;
use re::util::feature::*;

use re::prelude::*;

use gym::GymClient;

struct Acrobot {
	env: gym::Environment,
	render: bool,
}

impl Environment for Acrobot {
	type State = Vec<Range>;
	type Action = Finite;

	fn state_space(&self) -> Vec<Range> {
		vec![Range::sym(1.0), Range::sym(1.0), Range::sym(1.0), Range::sym(1.0), Range::sym(12.566), Range::sym(28.274)]
	}
	fn action_space(&self) -> Finite {
		Finite::new(3)
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

impl Acrobot {
	fn new() -> Acrobot {
		let client = GymClient::new("http://localhost:5000".to_string());
		let mut env = match client.make("Acrobot-v1") {
			Ok(env) => env,
			Err(msg) => panic!("Could not make environment because of error:\n{}\n\nMake sure you have a [gym server](https://github.com/openai/gym-http-api) running.", msg)
		};
		let _ = env.reset();
		Acrobot {env: env, render: false}
	}
	fn toggle_render(&mut self) {
		self.render = !self.render;
	}
}

fn main() {
	let mut env = Acrobot::new();

	let mut q_func = QLinear::default(&env.action_space());
	for d in 0..env.state_space().len() {
		q_func.add(Box::new(IFeature::new(d)));
	}

	// Creates an epsilon greedy Q-agent
	// 20% of the time, the agent uniformly chooses a random action
	let mut agent = EGreedyQAgent::new(q_func, env.action_space(), 0.20, Uniform);

	let tp = TimePeriod::OR(Box::new(TimePeriod::EPISODES(1)), Box::new(TimePeriod::TIMESTEPS(300)));
	// Train agent using Natural Evolution Strategies
	let mut trainer = NaturalEvo::default().eval_period(tp)
										   .deviation(0.01);

	println!("Training...");
	trainer.train(&mut agent, &mut env);
	println!("Done training (press enter)");

	env.toggle_render();
	let agent = agent.to_greedy();
	let _ = stdin().read_line(&mut String::new());

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	let mut reward = 0.0;
	while !obs.done {
		let action = agent.get_action(&obs.state);
		obs = env.step(&action);
		reward += obs.reward;
	}
	println!("total reward: {}", reward);

}