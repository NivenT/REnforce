//! Gradient Descent Module

use num::Float;

use util::GradientDescAlgo;

/// Simplest possible Gradient Descent algorithm
/// Gradient step is just gradient * learning_rate
#[derive(Clone, Copy, Debug)]
pub struct GradientDesc;

// Up to clinet to negate step if needed
impl<F: Float> GradientDescAlgo<F> for GradientDesc {
	fn calculate(&mut self, mut grad: Vec<F>, lr: F) -> Vec<F> {
		for x in &mut grad {
			*x = *x * lr;
		}
		grad
	}
}

/// Basic gradient descent with momentum
#[derive(Clone, Debug)]
pub struct GradDescMomentum<F: Float> {
	momentum: F,
	cache: Option<Vec<F>>,
}

impl<F: Float> GradientDescAlgo<F> for GradDescMomentum<F> {
    fn calculate(&mut self, grad: Vec<F>, lr: F) -> Vec<F> {
    	if self.cache.is_none() {
    		self.cache = Some(vec![F::zero(); grad.len()]);
    	}

    	// Probably not the cleanest way to do this
    	self.cache = Some(
    		self.cache.as_ref()
    				  .unwrap()
    				  .iter()
    				  .zip(grad.into_iter())
    				  .map(|(&x, y)| self.momentum*x + lr*y)
    				  .collect());
    	self.cache.clone().unwrap()
    }
}

impl Default for GradDescMomentum<f64> {
	fn default() -> GradDescMomentum<f64> {
		GradDescMomentum {
			momentum: 0.9,
			cache: None
		}
	}
}

impl<F: Float> GradDescMomentum<F> {
	/// Creates a new GradDescMomentum
	pub fn new(momentum: F) -> GradDescMomentum<F> {
		GradDescMomentum {
			momentum: momentum,
			cache: None
		}
	}
}

/// The RMSProp algorithm (Hinton et al. 2012).
#[derive(Debug, Clone)]
pub struct RMSProp<F: Float> {
    /// Rate at which running total of average square gradients decays
    decay_rate: F,
    /// Small value used to avoid divide by zero
    epsilon: F,
    cache: Option<Vec<F>>,
}

impl<F: Float> GradientDescAlgo<F> for RMSProp<F> {
    fn calculate(&mut self, grad: Vec<F>, lr: F) -> Vec<F> {
    	if self.cache.is_none() {
    		self.cache = Some(vec![F::zero(); grad.len()]);
    	}

    	self.cache = Some(
    		self.cache.as_ref()
    				  .unwrap()
    				  .iter()
    				  .zip(grad.iter())
    				  .map(|(&x, &y)| x * self.decay_rate + y*y*(F::one() - self.decay_rate))
    				  .collect()
        );

    	self.cache.as_ref()
    			  .unwrap()
    			  .iter()
    			  .zip(grad.into_iter())
    			  .map(|(&x, y)| y * lr / (x.sqrt() + self.epsilon))
    			  .collect()
    }
}

impl Default for RMSProp<f64> {
	fn default() -> RMSProp<f64> {
		RMSProp {
			decay_rate: 0.90,
			epsilon: 0.00001,
			cache: None
		}
	}
}

impl<F: Float> RMSProp<F> {
	/// Creates a new RMSProp
	pub fn new(decay_rate: F, epsilon: F) -> RMSProp<F> {
		RMSProp {
			decay_rate: decay_rate,
			epsilon: epsilon,
			cache: None
		}
	}
}