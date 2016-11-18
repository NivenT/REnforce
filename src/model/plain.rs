use std::collections::HashMap;
use std::hash::Hash;

use environment::{FiniteSpace, Transition};
use model::Model;

// TODO: Come up with a better name
/// StraightForward model that approximates probabilities can counting observations
#[derive(Debug)]
pub struct PlainModel<S: FiniteSpace, A: FiniteSpace>
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	trans: HashMap<(S::Element, A::Element), Vec<S::Element>>,
	rewards: HashMap<(S::Element, A::Element, S::Element), f64>,
}

impl<S: FiniteSpace, A: FiniteSpace> Model<S, A> for PlainModel<S, A>
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	fn transition(&self, curr: &S::Element, action: &A::Element, next: &S::Element) -> f64 {
		let key = (curr.clone(), action.clone());
		let poss = if self.trans.contains_key(&key) {&self.trans[&key]} else {return 0.0};

		let total = poss.len();
		let count = poss.into_iter().filter(|&s| s == next).count();
		(count as f64)/(total as f64)
	}
	fn reward(&self, curr: &S::Element, action: &A::Element, next: &S::Element) -> f64 {
		let key = (curr.clone(), action.clone(), next.clone());
		if self.rewards.contains_key(&key) {
			self.rewards[&key]
		} else {
			0.0
		}
	}
	fn update(&mut self, transition: Transition<S, A>) {
		let (state, action, reward, next) = transition;

		let key = (state.clone(), action.clone());
		self.trans.entry(key).or_insert(Vec::new()).push(next.clone());

		let key = (state.clone(), action.clone(), next.clone());
		self.rewards.insert(key, reward);
	}
}

impl<S: FiniteSpace, A: FiniteSpace> PlainModel<S, A>
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	/// Creates a new PlainModel
	pub fn new() -> PlainModel<S, A> {
		PlainModel {
			trans: HashMap::new(),
			rewards: HashMap::new()
		}
	}
}