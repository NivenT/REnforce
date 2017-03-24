//! Agent Module

pub mod qagents;
pub mod vagents;
mod randagent;
mod policyagent;

pub use self::randagent::RandomAgent;
pub use self::policyagent::PolicyAgent;

use environment::Space;

/// Represents an agent acting in an environment
pub trait Agent<S: Space, A: Space> {
	/// Returns the actions the agent should perform in the given state
	fn get_action(&self, state: &S::Element) -> A::Element;
}

/*
/// An agent that produces probabilites instead of deterministic actions
// Should this derive Agent?
pub trait StochasticAgent<F: Float, S: Space, A: Space> {
	/// Returns the probability of this agent performing an action in a given state
	fn get_action_prob(&self, state: &S::Element, action: &A::Element) -> F;
}
*/