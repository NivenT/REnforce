//! Trainer Module

mod qlearner;
mod cem;

pub use self::qlearner::{QLearner, SARSALearner, DynaQ, FittedQIteration};
pub use self::cem::CrossEntropy;

use environment::{Space, Environment, Transition};

use agent::Agent;

/// Represents a way to train an agent online (by interacting with the environment)
pub trait OnlineTrainer<S: Space, A: Space, T: Agent<S, A>> {
	/// Performs one training iteration using the given transition
	fn train_step(&mut self, agent: &mut T, transition: Transition<S, A>);
	/// Automatically trains the agent to perform well in the environment
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>);
}

/// Trains agents 1 "episode" at a time
pub trait EpisodicTrainer<S: Space, A: Space, T: Agent<S, A>> {
	/// Trains agent using 1 "episodes" worth of exploration
	fn train_step(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>);
	/// Trains agent to perform well in the environment, potentially acting out multiple episodes
	fn train(&mut self, agent: &mut T, env: &mut Environment<State=S, Action=A>);
}

/// Represents a way to train an agent from a set of transitions
pub trait BatchTrainer<S: Space, A: Space, T: Agent<S, A>> {
	/// Trains agent based on the observed transitions
	fn train(&mut self, agent: &mut T, transitions: Vec<Transition<S, A>>);
}