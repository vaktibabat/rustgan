use ndarray::{Array1, Array2, ArrayView2};

use crate::layers::Layer;

/// Types of activation functions
pub enum Activation {
    ReLU,
    Sigmoid,
}

/// Types of loss functions
#[derive(Copy, Clone)]
pub enum Loss {
    GanDiscriminator,
    GanGenerator,
    Mse,
    CrossEntropy,
}

// Sequence of layers + training hyperparameters
pub struct NeuralNet {
    pub layers: Vec<Box<dyn Layer>>,
    num_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    // Cache to store the intermediate outputs of the hidden layers, used for backprop
    pub cache: Vec<Array2<f64>>,
    // Loss function, i.e. cross entropy or GAN loss
    loss: Loss,
}

// Getters
impl NeuralNet {
    pub fn num_epochs(&self) -> usize {
        return self.num_epochs;
    }

    pub fn batch_size(&self) -> usize {
        return self.batch_size;
    }

    pub fn learning_rate(&self) -> f64 {
        return self.learning_rate;
    }

    pub fn loss(&self) -> Loss {
        return self.loss;
    }

    // The last item in the cache
    pub fn cache_last(&self) -> &Array2<f64> {
        self.cache.last().unwrap()
    }
}

impl NeuralNet {
    // This only initializes the hyperparameters
    // Layers need to be added manually using the add_layer function,
    // which adds its argument to be the last layer
    pub fn new(num_epochs: usize, batch_size: usize, learning_rate: f64, loss: Loss) -> NeuralNet {
        let layers = vec![];
        let cache = vec![];

        NeuralNet {
            layers,
            num_epochs,
            batch_size,
            learning_rate,
            cache,
            loss,
        }
    }

    // Append a new layer to the neural network
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    // Forward pass
    pub fn forward(&mut self, x: &ArrayView2<f64>, is_training: bool) -> Array2<f64> {
        let mut curr = x.to_owned();

        self.cache.push(curr.to_owned());

        for layer in self.layers.iter() {
            // Compute the output of this layer
            curr = layer.forward(&curr);
            // Save it to the cache
            if is_training {
                self.cache.push(curr.clone());
            }
        }

        curr
    }

    // Backward pass - Return the gradients of the loss WRT the model parameters
    // also returns the gradient of the loss WRT the input, which is useful when training some types of models
    // such as GANs
    pub fn backward(
        &mut self,
        dy: Array2<f64>,
    ) -> (Vec<(Option<Array2<f64>>, Option<Array1<f64>>)>, Array2<f64>) {
        // Run the forward pass to get the hidden layer outputs
        // self.forward(batch, true);
        // Output of the net
        let _ = self.cache.pop().unwrap();
        let mut dy = dy.clone();
        // Gradients WRT each layer
        let mut gradients = vec![];

        // Backprop
        for layer in self.layers.iter().rev() {
            // Output of the current layer
            let x = self.cache.pop().unwrap();
            let curr_grad = layer.backward(dy.clone(), x);
            // Change the upstream
            dy = curr_grad.0;
            // Push to the gradient vector
            gradients.push((curr_grad.1, curr_grad.2));
        }

        (gradients, dy)
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::Linear;

    use super::NeuralNet;

    #[test]
    fn test_forward_pass() {
        let mut network = NeuralNet::new(10, 64, 0.003, super::Loss::GanDiscriminator);
        network.add_layer(Box::new(Linear::new(100, 50)));
        network.add_layer(Box::new(Linear::new(50, 1)));
    }
}
