use environment::Space;

use agent::Agent;

pub struct RandomAgent<A: Space> {
	action_space: A
}

impl<S: Space, A: Space> Agent<S, A> for RandomAgent<A> {
	fn get_action(&self, _: S::Element) -> A::Element {
		self.action_space.sample()
	}
}

impl<A: Space> RandomAgent<A> {
	pub fn new(action_space: A) -> RandomAgent<A> {
		RandomAgent {
			action_space: action_space
		}
	}
}