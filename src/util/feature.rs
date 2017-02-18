//! Feature Extraction Module

use std::marker::PhantomData;
use std::fmt::Debug;

use num::Float;
use num::cast::NumCast;

use environment::Space;

use util::{Feature, Metric};

/// Identity Feature
///
/// Attempts to convert (state, action) pair into a Vec and returns the ith component
#[derive(Debug, Clone)]
pub struct IFeature<F: Float + Debug + 'static, T: Debug + Into<F>> {
	index: usize,
	phant1: PhantomData<T>,
	phant2: PhantomData<F>,
}

impl<F: Float + Debug + 'static, S: Space, T> Feature<S, F> for IFeature<F, T>
	where T: Into<F> + Debug + Clone + 'static,
		  S::Element: Into<Vec<T>> {
	fn extract(&self, state: &S::Element) -> F {
		state.clone().into()[self.index].clone().into()
	}
	fn box_clone(&self) -> Box<Feature<S, F>> {
		Box::new(self.clone())
	}
}

impl<F: Float + Debug + 'static, T: Debug + Into<F>> IFeature<F, T> {
	/// Creates a new Identity Feature
	pub fn new(index: usize) -> IFeature<F, T> {
		IFeature {index: index, phant1: PhantomData, phant2: PhantomData}
	}
}

/// Radial Basis (Function) Feature
///
/// Computes feature exp(-||s-s'||^2/(2u^2))
/// Represents a gaussian centered at s' with standard deviation u
#[derive(Debug, Clone)]
pub struct RBFeature<F:Float + Debug + 'static, S: Space> {
	center: S::Element,
	variation: F,
}

impl<F: Float + Debug + 'static, S: Space + Clone + 'static> Feature<S, F> for RBFeature<F, S> 
	where S::Element: Metric {
	fn extract(&self, state: &S::Element) -> F {
		let two = F::one() + F::one();
		let dist2: F = NumCast::from(Metric::dist2(state, &self.center)).unwrap();
		(-dist2/(two*self.variation)).exp()
	}
	fn box_clone(&self) -> Box<Feature<S, F>> {
		Box::new((*self).clone())
	}
}

impl<F: Float + Debug + 'static, S: Space> RBFeature<F, S> {
	/// Creates a new RBFeature with given center and standard deviation
	pub fn new(center: S::Element, deviation: F) -> RBFeature<F, S> {
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
pub struct BBFeature<F: Float + Debug + 'static, S: Space> {
	center: S::Element,
	radius: F,
}

impl<F: Float + Debug + 'static, S: Space + Clone + 'static> Feature<S, F> for BBFeature<F, S> 
	where S::Element: Metric {
	fn extract(&self, state: &S::Element) -> F {
		let dist2: F = NumCast::from(Metric::dist2(state, &self.center)).unwrap();
		if dist2 <= self.radius*self.radius {F::one()} else {F::zero()}
	}
	fn box_clone(&self) -> Box<Feature<S, F>> {
		Box::new(self.clone())
	}
}

impl<F: Float + Debug + 'static, S: Space> BBFeature<F, S> {
	/// Creates a new BBFeature
	pub fn new(center: S::Element, radius: F) -> BBFeature<F, S> {
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
pub struct BSFeature<F: Float + Debug + 'static, T: Debug + Into<F>> {
	min: F,
	max: F,
	dim: usize,
	phantom: PhantomData<T>,
}

impl<F: Float + Debug + 'static + 'static, S: Space, T> Feature<S, F> for BSFeature<F, T> 
	where T: Into<F> + Debug + Clone + 'static,
		  S::Element: Into<Vec<T>> {
	fn extract(&self, state: &S::Element) -> F {
		let val = state.clone().into()[self.dim].clone().into();
		if self.min <= val && val <= self.max {F::one()} else {F::zero()}
	}
	fn box_clone(&self) -> Box<Feature<S, F>> {
		Box::new(self.clone())
	}
}

impl<F: Float + Debug + 'static, T: Debug + Into<F>> BSFeature<F, T> {
	/// Creates a new BSFeature
	pub fn new(min: F, max: F, dim: usize) -> BSFeature<F, T> {
		BSFeature {
			min: min,
			max: max,
			dim: dim,
			phantom: PhantomData
		}
	}
}