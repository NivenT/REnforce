//! Feature Extraction Module

use std::marker::PhantomData;
use std::fmt::Debug;

use environment::Space;

use util::{Feature, Metric};

/// Identity Feature
///
/// Attempts to convert (state, action) pair into a Vec and returns the ith component
#[derive(Debug, Clone)]
pub struct IFeature<T: Debug + Into<f64>> {
	index: usize,
	phantom: PhantomData<T>,
}

impl<S: Space, T> Feature<S> for IFeature<T>
	where T: Into<f64> + Debug + Clone + 'static,
		  S::Element: Into<Vec<T>> {
	fn extract(&self, state: &S::Element) -> f64 {
		state.clone().into()[self.index].clone().into()
	}
	fn box_clone(&self) -> Box<Feature<S>> {
		Box::new(self.clone())
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
#[derive(Debug, Clone)]
pub struct RBFeature<S: Space> {
	center: S::Element,
	variation: f64,
}

impl<S: Space + Clone + 'static> Feature<S> for RBFeature<S> where S::Element: Metric {
	fn extract(&self, state: &S::Element) -> f64 {
		(-Metric::dist2(state, &self.center)/(2.0*self.variation)).exp()
	}
	fn box_clone(&self) -> Box<Feature<S>> {
		Box::new((*self).clone())
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

/// Binary Ball Feature
///
/// 1 iff the state is close enough to the center
#[derive(Debug, Clone)]
pub struct BBFeature<S: Space> {
	center: S::Element,
	radius: f64,
}

impl<S: Space + Clone + 'static> Feature<S> for BBFeature<S> where S::Element: Metric {
	fn extract(&self, state: &S::Element) -> f64 {
		if Metric::dist2(state, &self.center) <= self.radius*self.radius {1.0} else {0.0}
	}
	fn box_clone(&self) -> Box<Feature<S>> {
		Box::new(self.clone())
	}
}

impl<S: Space> BBFeature<S> {
	/// Creates a new BBFeature
	pub fn new(center: S::Element, radius: f64) -> BBFeature<S> {
		BBFeature {
			center: center,
			radius: radius
		}
	}
}

/// Binary Slice Feature
///
/// 1 iff the value in the specified dimension is in the given range
#[derive(Debug, Clone)]
pub struct BSFeature<T: Debug + Into<f64>> {
	min: f64,
	max: f64,
	dim: usize,
	phantom: PhantomData<T>,
}

impl<S: Space, T> Feature<S> for BSFeature<T> 
	where T: Into<f64> + Debug + Clone + 'static,
		  S::Element: Into<Vec<T>> {
	fn extract(&self, state: &S::Element) -> f64 {
		let val = state.clone().into()[self.dim].clone().into();
		if self.min <= val && val <= self.max {1.0} else {0.0}
	}
	fn box_clone(&self) -> Box<Feature<S>> {
		Box::new(self.clone())
	}
}

impl<T: Debug + Into<f64>> BSFeature<T> {
	/// Creates a new BSFeature
	pub fn new(min: f64, max: f64, dim: usize) -> BSFeature<T> {
		BSFeature {
			min: min,
			max: max,
			dim: dim,
			phantom: PhantomData
		}
	}
}