extern crate core;

mod math;
mod transform;

use std::io::{ErrorKind, Error};
use rand::{Rng, thread_rng};
use std::f64::consts::E as EXPONENT;
use std::ops::Mul;

use crate::math::matrix::Matrix;

struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(features_count: usize, mut layers: Vec<Layer>) -> Self {
        let mut units_count = features_count;
        for i in 0..layers.len() {
            layers[i].init_weights(units_count);
            units_count = layers[i].units_count;
        }

        Self { layers }
    }

    pub fn predict(&mut self, inputs: Vec<f64>) -> Matrix<f64> {
        let mut predictions = Matrix::new(1, inputs.len());
        predictions.assign(inputs).unwrap();

        for i in 1..self.layers.len() {
            predictions = self.layers[i].forward(predictions.clone());
        }

        predictions
    }

    pub fn train(
        &mut self,
        inputs: Vec<f64>,
        outputs: Vec<f64>,
        learning_rate: f64,
        iterations_count: usize,
    ) {
        for k in 0..iterations_count {
            let mut predictions = Matrix::new(1, inputs.len());
            predictions.assign(inputs.clone()).unwrap();

            let mut expected = Matrix::new(1, inputs.len());
            expected.assign(outputs.clone()).unwrap();

            for i in 1..self.layers.len() {
                predictions = self.layers[i].forward(predictions.clone());
            }

            if k % 100 == 0 {
                let cost = Self::compute_cost(&mut predictions, &mut expected);
                println!("Cost after: {:?} iteration: {:?}", k, cost);
            }

            // cross entropy
            let a = expected.clone() / predictions.clone();
            let b = (1. - expected.clone()) / (1. - predictions.clone());
            // d_cross_entropy
            let mut d_activation = -(a - b);

            for i in (1..self.layers.len()).rev() {
                let (d_activation_prev, d_weights, d_biases) = self.layers[i].backward(
                    &mut d_activation
                );

                d_activation = d_activation_prev;

                self.layers[i].update_parameters(d_weights, d_biases, learning_rate);
            }
        }
    }

    fn compute_cost(
        // AL
        predictions: &mut Matrix<f64>,
        // Y
        outputs: &mut Matrix<f64>
    ) -> f64 {
        - (outputs.clone() * predictions.clone().map(Box::new(Self::log))
            + (1. - outputs.clone()) * (1. - predictions.clone()).map(Box::new(Self::log))).sum()
            / outputs.shape.cols as f64
    }

    fn log(x: f64) -> f64 {
        let result = x.ln();
        if result.is_nan() || result.is_infinite() { 0. } else { result }
    }
}

#[derive(Debug, Default)]
struct Layer {
    pub units_count: usize,
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    activation_type: ActivationType,
    cache: Cache,
}

impl Layer {
    pub fn new(units_count: usize, activation_type: ActivationType) -> Self {
        Self {
            units_count,
            weights: Matrix::default(),
            biases: Matrix::new(units_count, 1).randomize(0., 1.),
            activation_type,
            cache: Cache::default(),
        }
    }

    pub fn init_weights(&mut self, prev_units_count: usize) {
        self.weights = Matrix::new(self.units_count, prev_units_count).randomize(0., 1.);
    }

    pub fn update_parameters(
        &mut self,
        d_weights: Matrix<f64>,
        d_biases: Matrix<f64>,
        learning_rate: f64
    ) {
        self.weights -= d_weights * learning_rate;
        self.biases -= d_biases * learning_rate;
    }

    pub fn forward(&mut self, mut input: Matrix<f64>) -> Matrix<f64> {
        let mut linear_activation = self.weights.dot(&mut input).unwrap() + self.biases.clone();

        let output = match self.activation_type {
            ActivationType::Sigmoid => {
                Layer::sigmoid(linear_activation)
            },
            ActivationType::Tanh => {
                Layer::tanh(linear_activation)
            },
            ActivationType::Relu => {
                Layer::relu(linear_activation)
            },
            ActivationType::Linear => {
                linear_activation
            },
        };

        self.cache = Cache {
            // prev activation
            input,
            weights: self.weights.clone(),
            // current activation
            output: output.clone(),
        };

        output
    }

    pub fn backward(&mut self, activation: &mut Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
        let m = activation.shape.cols as f64;

        let mut d_activation = match self.activation_type {
            ActivationType::Sigmoid => {
                activation.clone() * (1. - activation.clone())
            },
            ActivationType::Tanh => {
                1. - activation.clone().map(Box::new(|x| x * x))
            },
            ActivationType::Relu => {
                activation.clone().map(Box::new(|x| if x > 0. { x } else { 0. }))
            },
            ActivationType::Linear => {
                activation.clone().map(Box::new(|_| 1.))
            },
        };

        // dW
        let d_weights = d_activation.dot(&mut self.cache.input.transpose()).unwrap() / m;
        // db
        let d_biases = d_activation.x_axis_sum() / m;
        let d_activation_prev = self.weights.transpose().dot(&mut d_activation).unwrap();

        (d_activation_prev, d_weights, d_biases)
    }

    fn relu(input: Matrix<f64>) -> Matrix<f64> {
        input.clone().map(Box::new(|x| if x >= 0. { x } else { 0. }))
    }

    fn leaky_relu(input: Matrix<f64>) -> Matrix<f64> {
        input.clone().map(Box::new(|x| if x >= 0. { x } else { 0.1 * x }))
    }

    fn sigmoid(input: Matrix<f64>) -> Matrix<f64> {
        input.clone().map(Box::new(|x| 1. / ( 1. + (-x).exp())))
    }

    fn tanh(input: Matrix<f64>) -> Matrix<f64> {
        input.clone().map(Box::new(|x| x.tanh()))
    }
}

#[derive(Debug, Clone, Default)]
struct Cache {
    pub weights: Matrix<f64>,
    pub input: Matrix<f64>,
    pub output: Matrix<f64>,
}

#[derive(Debug, Clone)]
enum ActivationType {
    Sigmoid,
    Tanh,
    Relu,
    Linear,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::Linear
    }
}

fn main() {
    // training on factorial dataset
    let input = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let output = vec![1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880., 3628800.];

    let learning_rate = 0.0075;
    let iterations_count = 1;
    let features_count = 1;

    let mut nn = NeuralNetwork::new(features_count, vec![
        Layer::new(features_count, ActivationType::Relu),
        Layer::new(5, ActivationType::Relu),
        Layer::new(30, ActivationType::Relu),
        Layer::new(1, ActivationType::Sigmoid),
    ]);

    nn.train(input, output, learning_rate, iterations_count);
    // let prediction = nn.predict(vec![0., 1., 2., 3., 4.]);
}