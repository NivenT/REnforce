//! Feature Extraction Module

use std::marker::PhantomData;
use std::fmt::Debug;

use environment::Space;

use util::Feature;

/// Identity Feature
///
/// Attempts to convert (state, action) pair into a Vec and returns the ith component
#[derive(Debug)]
pub struct IFeature<T: Debug> {
	index: usize,
	phantom: PhantomData<T>,
}

impl<S: Space, A: Space, T: Into<f64> + Debug + Clone> Feature<S, A> for IFeature<T>
	where S::Element: Into<Vec<T>>, A::Element: Into<f64> {
	fn extract(&self, state: &S::Element, action: &A::Element) -> f64 {
		let state_vec: Vec<T> = state.clone().into();
		if self.index < state_vec.len() {
			state_vec[self.index].clone().into()
		} else {
			action.clone().into()
		}
	}
}

impl<T: Debug> IFeature<T> {
	/// Creates a new Identity Feature
	pub fn new(index: usize) -> IFeature<T> {
		IFeature {index: index, phantom: PhantomData}
	}
}