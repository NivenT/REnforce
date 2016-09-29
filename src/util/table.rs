use std::collections::HashMap;
use std::hash::Hash;
use std::f64;

use environment::{FiniteSpace, Space};

use util::{QFunction, VFunction};

pub struct QTable<S: FiniteSpace, A: FiniteSpace> {
	map: HashMap<(S::Element, A::Element), f64>
}

impl<S: FiniteSpace, A: FiniteSpace> QFunction<S, A> for QTable<S, A> 
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	fn eval(&self, state: S::Element, action: A::Element) -> f64 {
		if self.map.contains_key(&(state, action)) {
			self.map[&(state, action)]
		} else {
			0.0
		}
	}
	fn update(&mut self, state: S::Element, action: A::Element, new_val: f64, alpha: f64) {
		let old_val = self.eval(state, action);
		self.map.insert((state, action), old_val + alpha*(new_val - old_val));
	}
}

impl<S: FiniteSpace, A: FiniteSpace> QTable<S, A> 
	where S::Element: Hash + Eq, A::Element: Hash + Eq {
	pub fn new() -> QTable<S, A> {
		QTable {
			map: HashMap::new()
		}
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