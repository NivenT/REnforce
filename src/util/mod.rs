pub trait FunctionApproximator {
	fn eval(&self, input: Vec<f64>) -> f64;
	fn train(&self, inputs: Vec<Vec<f64>>, targets: Vec<f64>);
}