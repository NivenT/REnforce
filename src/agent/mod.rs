//! Agent Module

pub mod qlearner;
pub mod qagents;
mod randagent;

use environment::{Space, Environment, Transition};

pub use self::randagent::RandomAgent;

/// Agent Trait
///
/// Represents an agent acting in an environment
pub trait Agent<S: Space, A: Space> {
	/// Returns the function the agent should perform in the given state
	fn get_action(&self, state: S::Element) -> A::Element;
}

/// Model Trait
///
/// Represents a (nondeterministic) model of an environment
/// The model itself is composed of the transition and reward functions
pub trait Model<S: Space, A: Space> {
	/// Returns the probabilty of moving from curr to next when performing action
	fn transition(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64;
	/// Returns the reward received when moving from curr to next when performing action
	fn reward(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64;
	/// Updates the model using information from the given transition
	fn update_model(&mut self, transition: Transition<S, A>);
}

/// Deterministic Model Trait
///
/// Represents a deterministic model of an environment
/// When the agent performs a specified action in a specified state, there's only one possible next state
pub trait DeterministicModel<S: Space, A: Space> : Model<S, A> {
	/// Returns the new state of the agent after performing action in curr
	fn transition2(&self, curr: S::Element, action: A::Element) -> S::Element;
	/// Returns the reward received when performing action in curr
	fn reward2(&self, curr: S::Element, action: A::Element) -> f64;

	/// Same as Model's transtion function
	fn transition(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64 {
		let actual_next = self.transition2(curr, action);
		if next == actual_next {1.0} else {0.0}
	}

	/// Same as Model's reward function
	fn reward(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64 {
		let actual_next = self.transition2(curr, action);
		if next == actual_next {self.reward2(curr, action)} else {0.0}
	}
}

/// Online Trainer Trait
///
/// Represents a way to train an agent online (by interacting with the environment)
pub trait OnlineTrainer<S: Space, A: Space, T: Agent<S, A>> {
	/// Performs one training iteration using the given transition
	fn train_step(&self, agent: &mut T, transition: Transition<S, A>);
	/// Automatically trains the agent to perform well in the environment
	fn train(&self, agent: &mut T, env: &mut Environment<State=S, Action=A>);
}

/// Batch Trainer Trait
///
/// Represents a way to train an agent from a set of transitions
pub trait BatchTrainer<S: Space, A: Space, T: Agent<S, A>> {
	/// Trains agent based on the observed transitions
	fn train(&self, agent: &mut T, transitions: Vec<Transition<S, A>>);
}