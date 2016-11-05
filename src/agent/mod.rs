//! Agent Module

pub mod qagents;
pub mod vagents;
mod randagent;

pub use self::randagent::RandomAgent;

use environment::Space;

/// Agent Trait
///
/// Represents an agent acting in an environment
pub trait Agent<S: Space, A: Space> {
	/// Returns the function the agent should perform in the given state
	fn get_action(&self, state: &S::Element) -> A::Element;
}