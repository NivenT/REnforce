pub mod table;
pub mod chooser;

use environment::Space;

pub trait QFunction<S: Space, A: Space> {
	fn eval(&self, state: S::Element, action: A::Element) -> f64;
	fn update(&mut self, state: S::Element, action: A::Element, new_val: f64, alpha: f64);
}

pub trait VFunction<S: Space, A: Space> {
	fn eval(&self, state: S::Element) -> f64;
}

pub trait Chooser<T> {
	fn choose(&self, choices: Vec<T>, weights: Vec<f64>) -> T;
}