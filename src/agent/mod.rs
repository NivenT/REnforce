use environment::{Space, Environment, Transition};

pub trait Agent<S: Space, A: Space> {
	fn get_action(&self, env: Box<Environment<State=S, Action=A>>) -> A;
}

pub trait Model<S: Space, A: Space> {
	fn transition(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64;
	fn reward(&self, curr: S::Element, action: A::Element, next: S::Element) -> f64;
	fn update_model(&self, old: S::Element, action: A::Element, new: S::Element, reward: f64);
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

pub trait OnlineTrainer<S: Space, A: Space> {
	type AgentType : Agent<S, A>;

	fn train(&self, env: Box<Environment<State=S, Action=A>>) -> Self::AgentType;
	fn train_step(&self, agent: Self::AgentType, transition: Transition<S, A>) -> Self::AgentType;
}

pub trait BatchTrainer<S: Space, A: Space> {
	type AgentType : Agent<S, A>;

	fn train(&self, transitions: Vec<Transition<S, A>>) -> Self::AgentType;
}