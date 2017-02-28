use std::iter;

use environment::{Space, FiniteSpace};

impl<T: Space> Space for Vec<T> {
	type Element = Vec<T::Element>;

	fn sample(&self) -> Self::Element {
		self.iter()
			.map(|s| s.sample())
			.collect()
	}
}

impl<T: FiniteSpace + Clone> FiniteSpace for Vec<T> {
	fn enumerate(&self) -> Vec<Self::Element> {
		match self.len() {
			0 => vec![],
			1 => self[0].enumerate().into_iter().map(|x| vec![x]).collect(),
			_ => {
				let (head, tail) = self.split_first().unwrap();
				head.enumerate()
					.into_iter()
					.flat_map(|x| iter::repeat(x).zip(tail.to_vec().enumerate())
												 .map(|(h, mut t)| {
												 	t.insert(0, h);
												 	t
												 }))
					.collect()
			}
		}
	}

	fn size(&self) -> usize {
		(0..self.len()).map(|i| self[i].size()).product()
	}

	// TODO: index
}