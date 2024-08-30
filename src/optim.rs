use std::collections::VecDeque;
use crate::neural_net::NeuralNet;
use ndarray::{Array1, Array2};
use rand_distr::num_traits::Pow;

// So that we won't divide by zero in Adam
const EPS: f64 = 1e-6;

// Stochastic GD optimizer
pub struct SGD {
    learning_rate: f64,
}

// Adam optimizer: https://arxiv.org/abs/1412.6980
pub struct Adam {
    learning_rate: f64,
    // Estimates of the first (mean) and second (zero-biased variance) moments of the gradients
    // We store these as fields since we have to update them at each optimization step (they're exp. moving avgs.)
    first_moment: VecDeque<(Option<Array2<f64>>, Option<Array1<f64>>)>,
    second_moment: VecDeque<(Option<Array2<f64>>, Option<Array1<f64>>)>,
    // Parameters related to the exp. moving avgs: \beta_1 is the update rate for the first moment estimates
    // and \beta_2 is the update rate for the second moment estimates
    beta1: f64,
    beta2: f64,
    // Timestamp
    t: usize,
}

impl SGD {
    pub fn new(learning_rate: f64) -> SGD {
        SGD { learning_rate }
    }

    pub fn optimize(
        &self,
        net: &mut NeuralNet,
        gradients: Vec<(Option<Array2<f64>>, Option<Array1<f64>>)>,
    ) {
        // Perform GD step
        for i in 0..net.layers.len() {
            // The current gradient
            let grad = &gradients[i];

            // Proceed only if there are parameter gradients
            // layers such as ReLU don't have any parameters, so we don't need to update anything
            if let (Some(dw), Some(db)) = grad {
                net.layers[i].update_params(&dw, &db, self.learning_rate);
            }
        }
    }
}

impl Adam {
    // Construct a new Adam optimizer for NeuralNet net
    // We require a ref to net as a parameter since we need to know
    // what shapes the graidents take on (this can, of course, be derived from the parameters of the net)
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, net: &NeuralNet) -> Adam {
        let mut first_moment = VecDeque::new();
        let mut second_moment = VecDeque::new();

        for layer in &net.layers {
            if let Some(shape) = layer.shape() {
                // Linear layer has a (nrows, ncols) gradient for the weight and an (ncols) gradient for the bias
                first_moment.push_back((Some(Array2::zeros(shape)), Some(Array1::zeros(shape.1))));
                second_moment.push_back((Some(Array2::zeros(shape)), Some(Array1::zeros(shape.1))));
            } else {
                // Empty layers don't have gradients; added to make the code simpler when we perform
                // an optimization step
                first_moment.push_back((None, None));
                second_moment.push_back((None, None));
            }
        }

        Adam {
            learning_rate,
            first_moment,
            second_moment,
            beta1,
            beta2,
            t: 0,
        }
    }

    // Perform an optimization step on net, given the current gradients of the loss WRT the model parameters
    // We require a mut ref to self since we update the estimates of the first and second moments
    pub fn optimize(
        &mut self,
        net: &mut NeuralNet,
        gradients: Vec<(Option<Array2<f64>>, Option<Array1<f64>>)>,
    ) {
        let mut new_first_moment = vec![];
        let mut new_second_moment = vec![];

        // Increment the timestamp
        self.t += 1;

        // Compute the updates for both first & second moment estimates. We do this in reverse order
        // to not have to convert the moments into a VecDeque
        for grad in gradients.iter() {
            // If there's a gradient for the current layer, we compute the bias-corrected new moving avg.
            // Otherwise, we push a (None, None) tuple to the moving avgs.
            // indicating that the layer doesn't have any gradients
            if let (Some(grad_w), Some(grad_b)) = grad {
                // Estimation of the first moment (the mean)
                let m_prev_first = self.first_moment.pop_front().unwrap();
                let m_prev_first_w = m_prev_first.0.unwrap();
                let m_prev_first_b = m_prev_first.1.unwrap();
                let new_w = self.beta1 * m_prev_first_w + (1f64 - self.beta1) * grad_w;
                let new_b = self.beta1 * m_prev_first_b + (1f64 - self.beta1) * grad_b;
                // Bias-correct the estimation
                // We could do it in a different pass, but doing this here is better
                let new_w_bias_corrected = new_w / (1f64 - self.beta1.powi(self.t.try_into().unwrap()));
                let new_b_bias_corrected = new_b / (1f64 - self.beta1.powi(self.t.try_into().unwrap()));
                new_first_moment.push((Some(new_w_bias_corrected), Some(new_b_bias_corrected))); 
                // Estimation of the second moment (the variance)
                let m_prev_second = self.second_moment.pop_front().unwrap();
                let m_prev_second_w = m_prev_second.0.unwrap();
                let m_prev_second_b = m_prev_second.1.unwrap();
                let new_w_second = self.beta2 * m_prev_second_w + (1f64 - self.beta2) * grad_w * grad_w;
                let new_b_second = self.beta2 * m_prev_second_b + (1f64 - self.beta2) * grad_b * grad_b;
                // Bias-correct the estimation
                let new_w_bias_corrected_second = new_w_second / (1f64 - self.beta2.powi(self.t.try_into().unwrap()));
                let new_b_bias_corrected_second = new_b_second / (1f64 - self.beta2.powi(self.t.try_into().unwrap()));
                new_second_moment.push((Some(new_w_bias_corrected_second), Some(new_b_bias_corrected_second)));
            } else {
                // Pop the layer from the moment deques
                self.first_moment.pop_front();
                self.second_moment.pop_front();
                new_first_moment.push((None, None));
                new_second_moment.push((None, None));
            }
        }

        // Update the first & second moment estimations
        self.first_moment = VecDeque::from(new_first_moment);
        self.second_moment = VecDeque::from(new_second_moment);

        // Update model parameters
        for i in 0..net.layers.len() {
            // Pull the first & second moment estimations of the gradient WRT current parameter
            if let (Some(first_w), Some(first_b), Some(second_w), Some(second_b)) = (&self.first_moment.get(i).unwrap().0, &self.first_moment.get(i).unwrap().1, &self.second_moment.get(i).unwrap().0, &self.second_moment.get(i).unwrap().1) {
                // Compute the actual gradient from the moment estimates
                let dw = first_w / (second_w.sqrt() + EPS);
                let db = first_b / (second_b.sqrt() + EPS);

                // Proceed only if there are parameter gradients
                // layers such as ReLU don't have any parameters, so we don't need to update anything
                net.layers[i].update_params(&dw, &db, self.learning_rate);
            }
        }
    }
}
