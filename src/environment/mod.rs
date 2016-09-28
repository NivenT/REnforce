pub trait Space {
	type Element;

	fn sample(&self) -> Self::Element;
}

pub trait FiniteSpace : Space {
	fn enumerate(&self) -> Vec<Self::Element>;
}

pub struct Observation<S: Space> {
	state: 	S,
	reward: f64,
	done: 	bool
}

pub trait Environment {
	type State : Space;
	type Action : Space;

	fn step(action: Self::Action) -> Observation<Self::State>;
	fn reset() -> Observation<Self::State>;
	fn render();
}