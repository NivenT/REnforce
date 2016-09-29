use environment::{Space, FiniteSpace};

impl Space for () {
	type Element = ();

	fn sample(&self) -> () {
		()
	}
}

impl FiniteSpace for () {
	fn enumerate(&self) -> Vec<()> {
		Vec::new()
	}
}