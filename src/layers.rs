use ndarray::{Array1, Array2, Axis};
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

// A type of layer, such as a linear layer or a ReLU layer
pub trait Layer {
    // Forward pass on this layer
    fn forward(&self, x: &Array2<f64>) -> Array2<f64>;
    // Backward pass on this layer WRT the upstream gradient
    // also gets as input its original input data
    // Returns the gradients of the loss WRT x, w, and b
    fn backward(
        &self,
        dy: Array2<f64>,
        x: Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>, Option<Array1<f64>>);
    // How to update the parameters of this layer based on its gradients (if there are any)?
    fn update_params(&mut self, dw: &Array2<f64>, db: &Array1<f64>, learning_rate: f64);
    // The shape of this layer, if it has any (i.e. activation layers do not have any shape)
    fn shape(&self) -> Option<(usize, usize)>;
}

//-----------------START TYPES OF LAYERS-------------------------
// Linear layer of the form L(x) = WX + b
pub struct Linear {
    w: Array2<f64>,
    b: Array1<f64>,
}

// ReLU(z) = max(0, z)
pub struct ReLU {}

// Sigmoid function: \sigma(z) = 1 / (1 + e^(-z))
pub struct Sigmoid {}
//-----------------END TYPES OF LAYERS---------------------------

impl Linear {
    // Create a new linear layer with Xavier initialization
    // Fan in is the number of inputs to this layer (i.e. it gets an input of shape (batch_size, n))
    // Fan out is the number of outputs of this layer, i.e. the number of neurons
    pub fn new(fan_in: usize, fan_out: usize) -> Linear {
        let w = xavier_init(fan_in, fan_out);
        let b = Array1::zeros(fan_out);

        Linear { w, b }
    }
}

impl Layer for Linear {
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let (w, b) = (&self.w, &self.b);
        // Compute the linear transformation
        x.dot(w) + b
    }

    fn backward(
        &self,
        dy: Array2<f64>,
        x: Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>, Option<Array1<f64>>) {
        // Borrow w to not have to type the full "self.w" each time
        let w = &self.w;

        // The gradient of the output of this layer (y) WRT its input (x)
        let dy_dx = w.t();
        // The gradient of the loss WRT the input of this layer, acc. to the chain rule
        let dx = dy.dot(&dy_dx);

        // The gradient of the output of this layer (y) WRT its weights (w)
        let dy_dw = x.t();
        // The gradient of the loss WRT the weights of this layer
        let dw = dy_dw.dot(&dy);
        // The gradient of loss WRT the biases of this layer
        let db = dy.mean_axis(Axis(0)).unwrap();

        (dx, Some(dw), Some(db))
    }

    // Update the parameters of this layer based on the gradients
    fn update_params(&mut self, dw: &Array2<f64>, db: &Array1<f64>, learning_rate: f64) {
        self.w = &self.w - learning_rate * dw;
        self.b = &self.b - learning_rate * db;
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((self.w.nrows(), self.w.ncols()))
    }
}

impl Layer for ReLU {
    // ReLU function
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|x| x.max(0f64))
    }

    // The derivative of ReLU
    fn backward(
        &self,
        dy: Array2<f64>,
        x: Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>, Option<Array1<f64>>) {
        // We only return the gradient WRT the input, since ReLU layers don't have any parameters
        let dy_dx = x.map(|x| if *x <= 0f64 { 0f64 } else { 1f64 });
        let dx = dy * dy_dx;

        (dx, None, None)
    }

    fn update_params(&mut self, _: &Array2<f64>, _: &Array1<f64>, _: f64) {
        // No params to be updated
        ()
    }

    fn shape(&self) -> Option<(usize, usize)> {
        None
    }
}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {}
    }
}

impl Layer for Sigmoid {
    // The sigmoid function
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|x| (1f64 + (-x).exp()).recip())
    }

    // The derivative of the sigmoid function
    fn backward(
        &self,
        dy: Array2<f64>,
        x: Array2<f64>,
    ) -> (Array2<f64>, Option<Array2<f64>>, Option<Array1<f64>>) {
        let dy_dx = x.map(|x| {
            let s_x = (1f64 + (-x).exp()).recip();

            // Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
            s_x * (1f64 - s_x)
        });
        let dx = dy * dy_dx;

        (dx, None, None)
    }

    fn update_params(&mut self, _: &Array2<f64>, _: &Array1<f64>, _: f64) {
        // No params to be updated
        ()
    }

    fn shape(&self) -> Option<(usize, usize)> {
        None
    }
}

impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {}
    }
}

// Xavier initialization of a layer where the previous layer had fan_in neurons
// and the current layer has fan_out neurons
fn xavier_init(fan_in: usize, fan_out: usize) -> Array2<f64> {
    let mut rng = thread_rng();
    let param = (6f64 / (fan_in + fan_out) as f64).sqrt();
    // Distribution to sample the weights from
    let dist = Uniform::new(-param, param);
    // Create the weight matrix
    Array2::from_shape_fn((fan_in, fan_out), |_| dist.sample(&mut rng))
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng,
    };

    use crate::layers::Layer;

    use super::Linear;

    // Helper trait for generating random matrices
    trait RandomInit {
        fn random(n: usize, m: usize) -> Self;
    }

    impl RandomInit for Array2<f64> {
        fn random(n: usize, m: usize) -> Self {
            let mut rng = thread_rng();
            let dist = Uniform::new(-0.3, 0.3);
            // Create the weight matrix
            Array2::from_shape_fn((n, m), |_| dist.sample(&mut rng))
        }
    }

    #[test]
    fn test_linear() {
        // Random input data
        let x = Array2::random(64, 200);
        let layer = Linear::new(200, 100);

        assert_eq!(layer.forward(&x.clone()), x.dot(&layer.w) + layer.b);
    }
}
