use environment::{Space, Transition};

pub trait QFunction<S: Space, A: Space> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64;
	fn update(&self, transition: Transition<S, A>);
	fn batch_update(&self, transitions: Vec<Transition<S, A>>) {
		for t in transitions {
			self.update(t);
		}
	}
}

pub trait VFunction<S: Space, A: Space> {
	fn eval(&self, state: S::Element) -> f64;
	fn update(&self, transition: Transition<S, A>);
	fn batch_update(&self, transitions: Vec<Transition<S, A>>) {
		for t in transitions {
			self.update(t);
		}
	}
}