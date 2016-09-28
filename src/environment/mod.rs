pub type Transition<S: Space, A: Space> = (S::Element, A::Element, f64, S::Element);

pub trait Space {
	//Should we require Copy?
	type Element : PartialEq + Clone + Copy;

	fn sample(&self) -> Self::Element;
}

pub trait FiniteSpace : Space {
	fn enumerate(&self) -> Vec<Self::Element>;
}

pub struct Observation<S: Space> {
	state: 	S::Element,
	reward: f64,
	done: 	bool
}

pub trait Environment {
	type State : Space;
	type Action : Space;

	fn step(&self, action: <Self::Action as Space>::Element) -> Observation<Self::State>;
	fn reset(&self) -> Observation<Self::State>;
	fn render(&self);
}