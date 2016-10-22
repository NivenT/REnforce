// This will test a trainer's ability to work in a simple environment
// The agent needs to learn to output an odd number

extern crate renforce as re;

use re::environment::{Environment, Observation};
use re::environment::Finite;

use re::agent::{Agent, OnlineTrainer};
use re::agent::qagents::GreedyQAgent;
use re::agent::qlearner::QLearner;

use re::util::table::QTable;

struct NumberChooser;

impl Environment for NumberChooser {
	type State = ();
	type Action = Finite;

	fn step(&mut self, action: &u32) -> Observation<()> {
		Observation {
			state: (), 
			reward: if *action%2 == 1 {1.0} else {-1.0},
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

#[test]
fn learn_to_choose_odd() {
	let action_space = Finite::new(10);
	let q_func = QTable::new();
	let mut agent = GreedyQAgent::new(Box::new(q_func), action_space);
	let mut env = NumberChooser;
	let trainer = QLearner::new(action_space, 0.9, 0.9, 100);

	trainer.train(&mut agent, &mut env);

	assert!(agent.get_action(&())%2 == 1, "The agent should have learned to pick odd numbers");
}