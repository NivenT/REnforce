use super::environment::{Space, Environment};

pub trait Agent<S: Space, A: Space> {
	fn get_action(&self, env: Box<Environment<State=S, Action=A>>) -> A;
}

pub trait Model<S: Space, A: Space> {
	fn transition(&self, curr: S::Element, action: A, next: S::Element) -> f64;
	fn reward(&self, curr: S::Element, action: A, next: S::Element) -> f64;
	fn update_model(&self, old: S::Element, action: A, new: S::Element, reward: f64);
}

pub trait DeterministicModel<S: Space, A: Space> : Model<S, A> {
	fn transition2(&self, curr: S::Element, action: A::Element) -> S::Element;
	fn reward2(&self, curr: S::Element, action: A::Element) -> f64;

	fn transition(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64 {
		let actual_next = self.transition2(curr, action);
		if next == actual_next {1.0} else {0.0}
	}

	fn reward(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64 {
		let actual_next = self.transition2(curr, action);
		if next == actual_next {self.reward2(curr, action)} else {0.0}
	}
}