use environment::Space;

impl<T: Space> Space for Vec<T> {
	type Element = Vec<T::Element>;

	fn sample(&self) -> Self::Element {
		self.iter()
			.map(|s| s.sample())
			.collect()
	}
}

//TODO: impl<T: FiniteSpace> FiniteSpace for Vec<T>