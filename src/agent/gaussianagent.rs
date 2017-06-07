use std::marker::PhantomData;

use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

use num::Float;
use num::cast::NumCast;

use environment::Space;

use agent::Agent;

use util::{LogDiffFunc, DifferentiableVecFunc, ParameterizedFunc};

/// An agent that samples actions from a Normal distribution
/// * The mean of the distribution is dependent upon the agent's state
/// * The variance of the distribution is fixed
#[derive(Debug, Clone)]
// Can't tell if I hate or love traits
pub struct GaussianAgent<F: Float, T: Into<F> + Clone, S: Space, A: Space, D: DifferentiableVecFunc<S, F>> 
	where A::Element: Into<Vec<T>> {
	/// The function used to calculate the mean value of the Gaussian the agent draws actions from
	pub mean_func: D,
	/// The standard deviation of the Gaussian used by the agent
	deviation: F,

	phant1: PhantomData<S::Element>,
	phant2: PhantomData<A::Element>,
	phant3: PhantomData<T>,
}

impl<F: Float, T: Into<F> + Clone, S: Space, A: Space, D> ParameterizedFunc<F> for GaussianAgent<F, T, S, A, D> 
	where D: DifferentiableVecFunc<S, F>,
		  A::Element: Into<Vec<T>> {
	fn num_params(&self) -> usize {
		self.mean_func.num_params()
	}
	fn get_params(&self) -> Vec<F> {
		self.mean_func.get_params()
	}
	fn set_params(&mut self, params: Vec<F>) {
		self.mean_func.set_params(params)
	}
}

impl<F: Float, T: Into<F> + Clone, S: Space, A: Space, D> LogDiffFunc<S, A, F> for GaussianAgent<F, T, S, A, D>
	where D: DifferentiableVecFunc<S, F>,
		  A::Element: Into<Vec<T>> {
	fn log_grad(&self, state: &S::Element, action: &A::Element) -> Vec<F> {
		let action: Vec<_> = action.clone().into();
		let (val, grads) = (self.mean_func.apply(state), self.mean_func.get_grads(state));

		let coeffs: Vec<_> = (0..action.len()).map(|i| {
			(action[i].clone().into() - val[i])/(self.deviation * self.deviation)
		}).collect();

		grads.into_iter().map(|grad| {
			let mut sum = F::zero();
			for j in 0..coeffs.len() {
				sum = sum + coeffs[j]*grad[j];
			}
			sum
		}).collect()
	}
}

impl<F: Float, T: Into<F> + Clone, S: Space, A: Space, D> Agent<S, A> for GaussianAgent<F, T, S, A, D>
	where D: DifferentiableVecFunc<S, F>,
		  A::Element: Into<Vec<T>> + From<Vec<F>> {
	fn get_action(&self, state: &S::Element) -> A::Element {
		let mut rng = thread_rng();
		let mean = self.mean_func.apply(state);

		let action: Vec<_> = (0..mean.len()).map(|i| {
			let normal = Normal::new(mean[i].to_f64().unwrap(), 
									 self.deviation.to_f64().unwrap());
			NumCast::from(normal.ind_sample(&mut rng)).unwrap()
		}).collect();

		action.into()
	}
}

impl<F: Float, T: Into<F> + Clone, S: Space, A: Space, D> GaussianAgent<F, T, S, A, D> 
	where D: DifferentiableVecFunc<S, F>,
		  A::Element: Into<Vec<T>> {
	/// Creates a new GaussianAgent
	pub fn new(mean_func: D, deviation: F) -> GaussianAgent<F, T, S, A, D> {
		GaussianAgent {
			mean_func: mean_func,
			deviation: deviation,
			phant1: PhantomData,
			phant2: PhantomData,
			phant3: PhantomData
		}
	}
}