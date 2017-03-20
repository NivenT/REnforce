// In this example, we will train an agent to play tic-tac-toe against a random player

extern crate renforce as re;
extern crate rand;

use std::io::stdin;

use rand::{Rng, thread_rng};

use re::environment::{Environment, Observation, Space};
use re::environment::Finite;

use re::trainer::OnlineTrainer;
use re::trainer::SARSALearner;

use re::agent::Agent;
use re::agent::qagents::EGreedyQAgent;

use re::util::table::QTable;
use re::util::chooser::Softmax;

#[derive(Debug, PartialEq, Clone, Copy)]
enum Cell{E, X, O} //E for empty

use Cell::*;

struct Board {
	cells:	[Cell; 9]
}

impl Environment for Board {
	type State = Vec<Finite>;
	type Action = Finite;

	fn state_space(&self) -> Vec<Finite> {
		vec![Finite::new(3); 9]
	}
	fn action_space(&self) -> Finite {
		// The agent has 9 spots to play an X in
		Finite::new(9)
	}
	fn step(&mut self, action: &<Finite as Space>::Element) -> Observation<Self::State> {
		let mut winner = 0;
		let mut valid_move = false;
		let action = *action as usize;
		if action < 9 && self.cells[action] == E {
			self.cells[action] = X;
			winner = self.get_winner();
			valid_move = true;
			if winner == 0 {
				let mut rng = thread_rng();
				let empty_cells: Vec<_> = (0..9).filter(|&i| self.cells[i] == E).collect();
				if empty_cells.len() != 0 {
					let cell = empty_cells[rng.gen_range(0, empty_cells.len())];
					self.cells[cell] = O;
				}
			}
		}
		Observation {
			state: self.cells.iter().map(|&c| c as u32).collect(),
			reward: if valid_move {winner as f64} else {-0.5},
			done: if winner == 0 {(0..9).filter(|&i| self.cells[i] == E).count() == 0} else {true}
		}
	}
	fn reset(&mut self) -> Observation<Self::State> {
		*self = Board::new();
		Observation {
			state: vec![0; 9],
			reward: 0.0,
			done: false
		}
	}
	fn render(&self) {
		println!(" {:?} | {:?} | {:?} ", self.cells[0], self.cells[1], self.cells[2]);
		println!("---|---|---");
		println!(" {:?} | {:?} | {:?} ", self.cells[3], self.cells[4], self.cells[5]);
		println!("---|---|---");
		println!(" {:?} | {:?} | {:?} ", self.cells[6], self.cells[7], self.cells[8]);
		println!("");
	}
}

impl Board {
	fn new() -> Board {
		Board {
			cells: [E; 9]
		}
	}
	fn get_winner(&self) -> i8 {
		let board = &self.cells;
		for i in 0..3 {
			if board[i] == board[i+3] && board[i+3] == board[i+6] && board[i+6] != E {
				return -2*(board[i] as i8) + 3;
			} else if board[i*3] == board[i*3+1] && board[i*3+1] == board[i*3+2] && board[i*3+2] != E {
				return -2*(board[i*3] as i8) + 3;
			}
		}
		if board[0] == board[4] && board[4] == board[8] && board[8] != E {
			return -2*(board[0] as i8) + 3;
		} else if board[2] == board[4] && board[4] == board[6] && board[6] != E {
			return -2*(board[2] as i8) + 3;
		}
		0
	}
}

fn main() {
	let mut env = Board::new();

	let q_func = QTable::new();
	// Creates an epsilon greedy Q-agent
	// Agent will use softmax to act randomly 15% of the time
	let mut agent = EGreedyQAgent::new(q_func.clone(), env.action_space(),
										0.15, Softmax::new(1.0));
	

	// We will use SARSA learning to train the agent
	// In this example, default values are used for the parameters
	// In general, you can set the learning rate, discount factor,
	// 	and training period manually
	let mut trainer = SARSALearner::default();

	// Magic happens
	trainer.train(&mut agent, &mut env);
	// Agent will no longer explore, only exploit
	let agent = agent.to_greedy();

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	let mut reward = 0.0;
	while !obs.done {
		env.render();

		let action = agent.get_action(&obs.state);
		obs = env.step(&action);
		reward += obs.reward;
		println!("action: {:?}", action);

		let _ = stdin().read_line(&mut String::new());
	}
	env.render();
	println!("total reward: {}", reward);
}