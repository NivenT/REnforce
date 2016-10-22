//! Feature Extraction Module

use std::marker::PhantomData;
use std::fmt::Debug;

use environment::Space;

use util::{Feature, Metric};

/// Identity Feature
///
/// Attempts to convert (state, action) pair into a Vec and returns the ith component
#[derive(Debug)]
pub struct IFeature<T: Debug + Into<f64>> {
	index: usize,
	phantom: PhantomData<T>,
}

impl<S: Space, T> Feature<S> for IFeature<T>
	where T: Into<f64> + Debug + Clone,
		  S::Element: Into<Vec<T>> {
	fn extract(&self, state: &S::Element) -> f64 {
		state.clone().into()[self.index].clone().into()
	}
}

impl<T: Debug + Into<f64>> IFeature<T> {
	/// Creates a new Identity Feature
	pub fn new(index: usize) -> IFeature<T> {
		IFeature {index: index, phantom: PhantomData}
	}
}

/// Radial Basis (Function) Feature
///
/// Computes feature exp(-||s-s'||^2/(2u^2))
/// Represents a gaussian centered at s' with standard deviation u
#[derive(Debug)]
pub struct RBFeature<S: Space> {
	center: S::Element,
	variation: f64,
}

impl <S: Space> Feature<S> for RBFeature<S> where S::Element: Metric {
	fn extract(&self, state: &S::Element) -> f64 {
		(-Metric::dist2(state, &self.center)/(2.0*self.variation)).exp()
	}
}

impl<S: Space> RBFeature<S> {
	/// Creates a new RBFeature with given center and standard deviation
	pub fn new(center: S::Element, deviation: f64) -> RBFeature<S> {
		RBFeature {
			center: center,
			variation: deviation*deviation
		}
	}
}