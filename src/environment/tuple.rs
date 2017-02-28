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

	fn size(&self) -> usize {
		self.0.size() * self.1.size()
	}

	fn index(&self, elm: Self::Element) -> isize {
		let i = self.0.index(elm.0);
		if i == -1 {-1} else {
			let j = self.1.index(elm.1);
			if j == -1 {-1} else {i*self.1.size() as isize + j}
		}
	}
}