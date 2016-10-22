use util::Metric;

impl Metric for u32 {
	fn dist2(x: &u32, y: &u32) -> f64 {
		let (x, y) = (*x as f64, *y as f64);
		((x-y)*(x-y))
	}
}

impl Metric for f64 {
	fn dist2(x: &f64, y: &f64) -> f64 {
		let (x, y) = (*x, *y);
		((x-y)*(x-y))
	}
}

impl<T: Metric> Metric for Vec<T> {
	fn dist2(x: &Vec<T>, y: &Vec<T>) -> f64 {
		assert_eq!(x.len(), y.len(), "You can only take the distance between vectors of equal length");
		x.iter().zip(y.iter())
				.map(|(x, y)| Metric::dist2(x, y))
				.sum()
	}
}

impl<T: Metric, U: Metric> Metric for (T, U) {
	fn dist2(x: &(T, U), y: &(T, U)) -> f64 {
		Metric::dist2(&x.0, &y.0) + Metric::dist2(&x.1, &y.1)
	}
}