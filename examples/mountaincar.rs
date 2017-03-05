// Here, we train an agent on the mountain car problem

// Currently, the agent does learn to make it to the top of the hill
// (which I think is a nontrivial accomplishment, but I'm not 100% sure)
// However, it takes it much longer than it should to get there
// At some point, I'll need to take some time to verify that I'm doing things
// correctly, and do more exploring with hyper parameters

extern crate renforce as re;
extern crate gym;

use std::io::stdin;

use re::environment::{Environment, Observation};
use re::environment::{Finite, Range};

use re::trainer::EpisodicTrainer;
use re::trainer::PolicyGradient;

use re::agent::Agent;
use re::agent::PolicyAgent;

use re::util::TimePeriod;
use re::util::approx::QLinear;
use re::util::feature::BBFeature;
use re::util::graddesc::GradientDesc;

use gym::GymClient;

struct MountainCar {
	pub render: bool,

	env: gym::Environment,
}

impl Environment for MountainCar {
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

impl MountainCar {
	fn new() -> MountainCar {
		let client = GymClient::new("http://localhost:5000".to_string());
		let mut env = match client.make("MountainCar-v0") {
			Ok(env) => env,
			Err(msg) => panic!("Could not make environment because of error:\n{}\n\nMake sure you have a [gym server](https://github.com/openai/gym-http-api) running.", msg)
		};
		let _ = env.reset();
		MountainCar {env: env, render: false}
	}
}

fn main() {
	// The agent has 3 actions
	let action_space = Finite::new(3);

	let mut log_prob_func = QLinear::default(&action_space);
	for i in 0..18 {
		for j in 0..14 {
			let (x, y) = (-1.2 + 0.1 * i as f64, -0.07 + 0.01 * j as f64);
			log_prob_func.add(Box::new(BBFeature::new(vec![x, y], 0.2)));
		}
	}

	// Creates an agent that acts randomly
	// The (log of the) probability of each action is determined by log_prob_func
	let mut agent = PolicyAgent::default(action_space, log_prob_func);

	let mut env = MountainCar::new();

	// PolicyGradient will evalute agents for 1 episode or 1000 time steps
	// whichever comes first
	let tp = TimePeriod::OR(Box::new(TimePeriod::EPISODES(1)), Box::new(TimePeriod::TIMESTEPS(1000)));
	// Train agent using Policy gradients with (mostly) default parameters
	let mut trainer = PolicyGradient::default(action_space, GradientDesc).eval_period(tp)
																		 .update_delay(10);
																		 
	println!("Training...");
	trainer.train(&mut agent, &mut env);
	println!("Done training (press enter)");

	env.render = true;
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
