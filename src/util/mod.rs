pub mod table;

use environment::{Space, Transition};

pub trait QFunction<S: Space, A: Space + Clone> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64;
	fn update(&mut self, transition: Transition<S, A>, action_space: A, gamma: f64, alpha: f64);
	fn batch_update(&mut self, transitions: Vec<Transition<S, A>>, action_space: A, gamma: f64, alpha: f64) {
		for t in transitions {
			self.update(t, action_space.clone(), gamma, alpha);
		}
	}
}

pub trait VFunction<S: Space, A: Space> {
	fn eval(&self, state: S::Element) -> f64;
}