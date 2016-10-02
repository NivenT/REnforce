use environment::Space;

use agent::Agent;

/// Random Agent
///
/// Represents an agent that acts randomly
#[derive(Debug)]
pub struct RandomAgent<A: Space> {
	/// The space the agent draws its actions from
	action_space: A
}

impl<S: Space, A: Space> Agent<S, A> for RandomAgent<A> {
	fn get_action(&self, _: S::Element) -> A::Element {
		self.action_space.sample()
	}
}

impl<A: Space> RandomAgent<A> {
	/// Creates a new random agent that performs actions from the given space
	pub fn new(action_space: A) -> RandomAgent<A> {
		RandomAgent {
			action_space: action_space
		}
	}
}