//! Feature Extraction Module

use std::marker::PhantomData;
use std::fmt::Debug;

use environment::Space;

use util::{BinaryFeature, Feature, Metric};

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

impl<S: Space> Feature<S> for RBFeature<S> where S::Element: Metric {
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

/// Binary Ball Feature
///
/// 1 iff the state is close enough to the center
#[derive(Debug)]
pub struct BBFeature<S: Space> {
	center: S::Element,
	radius: f64,
}

impl<S: Space> BinaryFeature<S> for BBFeature<S> where S::Element: Metric {
	fn b_extract(&self, state: &S::Element) -> bool {
		Metric::dist2(state, &self.center) <= self.radius*self.radius
	}
}

impl<S: Space> Feature<S> for BBFeature<S> where S::Element: Metric {
	fn extract(&self, state: &S::Element) -> f64 {
		if self.b_extract(state) {1.0} else {0.0}
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

/// Bineary Slice Feature
///
/// 1 iff the value in the specified dimension is in the given range
#[derive(Debug)]
pub struct BSFeature<T: Debug + Into<f64>> {
	min: f64,
	max: f64,
	dim: usize,
	phantom: PhantomData<T>,
}

impl<S: Space, T> BinaryFeature<S> for BSFeature<T>
	where T: Into<f64> + Debug + Clone,
		  S::Element: Into<Vec<T>> {
	fn b_extract(&self, state: &S::Element) -> bool {
		let val = state.clone().into()[self.dim].clone().into();
		self.min <= val && val <= self.max
	}
}

impl<S: Space, T> Feature<S> for BSFeature<T> 
	where T: Into<f64> + Debug + Clone,
		  S::Element: Into<Vec<T>> {
	fn extract(&self, state: &S::Element) -> f64 {
		let bin: &BinaryFeature<S> = self;
		if bin.b_extract(state) {1.0} else {0.0}
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