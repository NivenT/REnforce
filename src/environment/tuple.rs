use std::iter;

use environment::{Space, FiniteSpace};

impl<T: Space, U: Space> Space for (T, U) {
	type Element = (T::Element, U::Element);

	fn sample(&self) -> Self::Element {
		(self.0.sample(), self.1.sample())
	}
}

impl <T: FiniteSpace, U: FiniteSpace> FiniteSpace for (T, U) {
	fn enumerate(&self) -> Vec<Self::Element> {
		let (x_enum, y_enum) = (self.0.enumerate(), self.1.enumerate());
		x_enum.into_iter()
			  .flat_map(|x| iter::repeat(x.clone()).zip(y_enum.clone().into_iter()))
			  .collect()
	}
}