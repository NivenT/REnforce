//! Agent Module

pub mod qagents;
pub mod vagents;
mod randagent;

pub use self::randagent::RandomAgent;

use num::Num;

use environment::Space;

/// Represents an agent acting in an environment
pub trait Agent<S: Space, A: Space> {
	/// Returns the actions the agent should perform in the given state
	fn get_action(&self, state: &S::Element) -> A::Element;
}

/// An agent whose actions are determined by some parameters
pub trait ParameterizedAgent<S: Space, A: Space, T: Num> : Agent<S, A> {
	/// Returns the parameters used by the agent
	fn get_params(&self) -> Vec<T>;
	/// Changes the parameters used by the agent
	fn set_params(&self, params: &Vec<T>);
}