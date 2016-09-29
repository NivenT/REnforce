use std::collections::HashMap;
use std::hash::Hash;
use std::f64;

use environment::{FiniteSpace, Space, Transition};

use util::{QFunction, VFunction};

pub struct QTable<S: FiniteSpace, A: FiniteSpace> {
	map: HashMap<(S::Element, A::Element), f64>
}

impl<S: FiniteSpace, A: FiniteSpace + Clone> QFunction<S, A> for QTable<S, A> 
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	fn eval(&self, state: S::Element, action: A::Element) -> f64 {
		if !self.map.contains_key(&(state, action)) {
			self.map[&(state, action)]
		} else {
			0.0
		}
	}
	fn update(&mut self, transition: Transition<S, A>, action_space: A, gamma: f64, alpha: f64) {
		let (state, action, reward, next) = transition;
		let old_val = self.eval(state, action);

		let mut max_next_val = f64::MIN;
		for a in action_space.enumerate() {
			max_next_val = max_next_val.max(self.eval(next, a));
		}
		self.map.insert((state, action), old_val + alpha*(reward + gamma*max_next_val - old_val));
	}
}

pub struct VTable<S: FiniteSpace> {
	map: HashMap<S::Element, f64>
}

impl<S: FiniteSpace, A: Space> VFunction<S, A> for VTable<S> where S::Element: Hash + Eq {
	fn eval(&self, state: S::Element) -> f64 {
		//*self.map.entry(state).or_insert(0.0)
		if self.map.contains_key(&state) {
			self.map[&state]
		} else {
			0.0
		}
	}
}