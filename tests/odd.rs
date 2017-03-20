// This will test a trainer's ability to work in a simple environment
// The agent needs to learn to output an odd number

extern crate renforce as re;

use re::environment::{Environment, Observation};
use re::environment::Finite;

use re::trainer::OnlineTrainer;
use re::trainer::QLearner;

use re::agent::Agent;
use re::agent::qagents::GreedyQAgent;

use re::util::TimePeriod;
use re::util::table::QTable;

struct NumberChooser;

impl Environment for NumberChooser {
	type State = ();
	type Action = Finite;

	fn state_space(&self) {}
	fn action_space(&self) -> Finite {
		Finite::new(10)
	}
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
	let mut env = NumberChooser;
	let mut agent = GreedyQAgent::new(QTable::new(), env.action_space());
	let mut trainer = QLearner::new(env.action_space(), 0.9, 0.9, TimePeriod::TIMESTEPS(100));

	trainer.train(&mut agent, &mut env);

	assert!(agent.get_action(&())%2 == 1, "The agent should have learned to pick odd numbers");
}