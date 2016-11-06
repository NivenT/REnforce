// In this example, we will train an agent to find the shorted path through a maze

extern crate renforce as re;

use std::io::stdin;

use re::environment::{Environment, Observation};
use re::environment::Finite;

use re::trainer::OnlineTrainer;
use re::trainer::QLearner;

use re::agent::Agent;
use re::agent::qagents::EGreedyQAgent;

use re::util::TimePeriod;
use re::util::table::QTable;
use re::util::chooser::Softmax;

#[derive(PartialEq)]
enum MazeSpot{EMTY, WALL, GOAL}

use MazeSpot::*;

struct Maze {
	grid: Vec<Vec<MazeSpot>>,
	loc: (u32, u32),
}

impl Environment for Maze {
	type State = (Finite, Finite);
	type Action = Finite;

	fn step(&mut self, action: &u32) -> Observation<(Finite, Finite)> {
		let action = *action;
		if action < 4 {
			self.move_agent(action);
		}
		// End episode when agent reaches goal
		let done = self.grid[self.loc.0 as usize][self.loc.1 as usize] == GOAL;
		Observation {
			state: self.loc, 
			// Punish agent for every step it takes, but reward it when it reaches the goal
			// The optimal strategy is then to take the shortest path to the goal
			reward: if done {1.0} else {-1.0},
			done: done
		}
	}
	fn reset(&mut self) -> Observation<(Finite, Finite)> {
		*self = Maze::new();
		Observation {
			state: (0, 0),
			reward: 0.0,
			done: false
		}
	}
	fn render(&self) {
		println!("===============");
		for r in 0..8 {
			for c in 0..8 {
				let ch = match self.grid[r as usize][c as usize] {
					EMTY => " ",
					WALL => "#",
					GOAL => "_",
				};
				print!("{} ", if (r, c) == self.loc {"A"} else {ch});
			}
			println!("");
		}
		println!("===============");
	}
}

impl Maze {
	fn new() -> Maze {
		Maze {
			grid: vec![vec![EMTY, WALL, WALL, WALL, WALL, EMTY, EMTY, EMTY],
					   vec![EMTY, WALL, WALL, WALL, WALL, EMTY, WALL, EMTY],
					   vec![EMTY, WALL, EMTY, EMTY, WALL, EMTY, WALL, EMTY],
					   vec![EMTY, WALL, EMTY, WALL, WALL, EMTY, WALL, EMTY],
					   vec![EMTY, EMTY, EMTY, EMTY, EMTY, EMTY, WALL, EMTY],
					   vec![EMTY, WALL, WALL, WALL, WALL, WALL, WALL, EMTY],
					   vec![EMTY, WALL, EMTY, EMTY, EMTY, EMTY, WALL, EMTY],
					   vec![EMTY, EMTY, EMTY, EMTY, EMTY, EMTY, WALL, GOAL]],
			loc: (0, 0)
		}
	}
	fn is_in_bounds(&self, loc: (usize, usize)) -> bool {
		loc.0 < 8 && loc.1 < 8
	}
	fn move_agent(&mut self, action: u32) {
		if action >= 4 {
			return;
		}
		let dir = match action {
			0 => (0, 1),
			1 => (0, -1),
			2 => (-1, 0),
			3 => (1, 0),
			_ => (0, 0),
		};

		let new_loc = (self.loc.0 as isize + dir.0, self.loc.1 as isize + dir.1);
		let new_loc = (new_loc.0 as usize, new_loc.1 as usize);

		if self.is_in_bounds(new_loc) && self.grid[new_loc.0][new_loc.1] != WALL {
			self.loc = (new_loc.0 as u32, new_loc.1 as u32);
		}
	}
}

fn main() {
	// The agent has 4 actions: move {up, down, left, right}
	let action_space = Finite::new(4);
	// The agent will use a table as its Q-function
	let q_func = QTable::new();
	// Creates an epsilon greedy Q-agent
	// Agent will use softmax to act randomly 5% of the time
	let mut agent = EGreedyQAgent::new(q_func, action_space, 0.05, Softmax::new(1.0));
	let mut env = Maze::new();
	// We will use Q-learning to train the agent with
	// discount factor and learning rate both 0.9 and
	// 10000 training iterations
	let mut trainer = QLearner::new(action_space, 0.9, 0.9, TimePeriod::TIMESTEPS(10000));

	// Magic happens
	trainer.train(&mut agent, &mut env);

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	while !obs.done {
		env.render();

		let action = agent.get_action(&obs.state);
		obs = env.step(&action);

		let _ = stdin().read_line(&mut String::new());
	}
	env.render();
}