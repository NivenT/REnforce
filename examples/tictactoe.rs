// In this example, we will train an agent to play tic-tac-toe

extern crate renforce as re;
extern crate rand;

use std::io::stdin;

use rand::{Rng, thread_rng};

use re::environment::{Environment, Observation, Space, FiniteSpace};
use re::environment::{Finite, Range};

use re::agent::{Agent, OnlineTrainer};
use re::agent::qagents::EGreedyQAgent;
use re::agent::qlearner::QLearner;

use re::util::table::QTable;
use re::util::approx::QLinear;
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

	fn step(&mut self, action: <Finite as Space>::Element) -> Observation<Self::State> {
		let mut winner = 0;
		let action = action as usize;
		if action < 9 && self.cells[action] == E {
			self.cells[action] = X;
			winner = self.get_winner();
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
			state: self.cells.iter().map(|&c| c as i64 as usize).collect(),
			reward: winner as f64,
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
				return -2*(board[i] as i8) + 3;
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
	// The agent has 9 spots to play an X in
	let action_space = Finite::new(9);
	// The agent will use a table as its Q-function
	let q_func = QTable::new();
	// Creates an epsilon greedy Q-agent
	// Agent will use softmax to act randomly 5% of the time
	let mut agent = EGreedyQAgent::new(Box::new(q_func), action_space, 0.05, Softmax::new(1.0));
	let mut env = Board::new();
	// We will use Q-learning to train the agent with
	// discount factor and learning rate both 0.9 and
	// 100000 training iterations
	let trainer = QLearner::new(action_space, 0.9, 0.9, 100000);

	// Magic happens
	trainer.train(&mut agent, &mut env);

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	while !obs.done {
		env.render();

		let action = agent.get_action(obs.state);
		obs = env.step(action);

		let _ = stdin().read_line(&mut String::new());
	}
	env.render();
}