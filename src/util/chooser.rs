use rand::{Rng, thread_rng};

use util::Chooser;

pub struct Uniform;

impl<T: Clone> Chooser<T> for Uniform {
	fn choose(&self, choices: Vec<T>, _: Vec<f64>) -> T {
		let mut rng = thread_rng();
		rng.choose(&choices).unwrap().clone()
	}
}

pub struct Softmax {
	temp: f64
}

impl<T: Clone> Chooser<T> for Softmax {
	fn choose(&self, choices: Vec<T>, weights: Vec<f64>) -> T {
		let mut total = 0.0;
		let new_weights: Vec<_> = weights.into_iter()
								 		 .map(|w| {
								 			let u = (w/self.temp).exp();
										 	total += u;
										 	u
										 })
										 .collect();
		let mut rng = thread_rng();
		let mut index = 0;
		let mut choice = rng.gen_range(0.0, total);
		while choice > new_weights[index] {
			choice -= new_weights[index];
			index = index + 1;
		}
		choices[index].clone()
	}
}

impl Softmax {
	pub fn new(temp: f64) -> Softmax {
		Softmax {
			temp: temp
		}
	}
}