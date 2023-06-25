mod math;
mod transform;
mod dataset;

use std::io::{ErrorKind, Error};
use rand::{Rng, thread_rng};
use std::f64::consts::E as EXPONENT;
use std::ops::Mul;
use crate::dataset::{generate_image_dataset, read_vector_from_file, write_vector_to_file};

use crate::math::matrix::Matrix;
use crate::transform::image::image_to_vector;

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

    pub fn predict(&mut self, input: Matrix<f64>, mut output: Matrix<f64>) -> Vec<f64> {
        let examples_count = input.shape.cols;
        let mut normalized_predictions = Matrix::new_empty(1, examples_count);
        let mut predictions = input.clone();

        for i in 0..self.layers.len() {
            predictions = self.layers[i].forward(predictions);
        }

        let mut data = vec![];
        for i in 0..predictions.shape.cols {
            data.push(if *predictions.get_item(0, i).unwrap() > 0.5 { 1. } else { 0. });
        }
        normalized_predictions.assign(data);

        let accuracy_data = (0..predictions.size)
            .map(|i| {
                let a = normalized_predictions.get_item(0, i).unwrap();
                let b = output.get_item(0, i).unwrap();
                if a == b { 1. } else { 0. }
            }).collect::<Vec<f64>>();

        let mut accuracy = Matrix::new(1, normalized_predictions.shape.cols);
        accuracy.assign(accuracy_data.to_vec());

        println!("Accuracy: {:?}", (accuracy / examples_count as f64).sum());

        accuracy_data
    }

    pub fn train(
        &mut self,
        inputs: Matrix<f64>,
        outputs: Matrix<f64>,
        learning_rate: f64,
        iterations_count: usize,
    ) {
        let mut prev_cost = 0.0;

        for k in 0..iterations_count {
            let mut predictions = inputs.clone();
            let mut expected = outputs.clone();

            for i in 0..self.layers.len() {
                predictions = self.layers[i].forward(predictions);
            }

            // if k % 100 == 0 {
            //     let cost = Self::compute_cost(&mut predictions, &mut expected);
            //     println!("Cost after: {:?} iteration: {:?}", k, cost);
            // }

            let current_cost = Self::compute_cost(&mut predictions, &mut expected);
            println!(
                "Cost after: {:?} iteration: {:?} [{:?}]",
                k,
                current_cost,
                if current_cost >= prev_cost { "+" } else {"-"}
            );
            prev_cost = current_cost;

            // cross entropy
            let epsilon = 1e-8;
            // let epsilon = 0.;
            let a = expected.clone() / predictions.clone();
            let b = ((1. - expected.clone()) + epsilon) / ((1. - predictions.clone()) + epsilon);
            // d_cross_entropy
            let mut d_activation = -(a - b);

            for i in (0..self.layers.len()).rev() {
                let (d_activation_prev, d_weights, d_biases) = self.layers[i].backward(
                    d_activation
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
            biases: Matrix::new(units_count, 1),//.randomize(-1., 1.),
            activation_type,
            cache: Cache::default(),
        }
    }

    pub fn init_weights(&mut self, prev_units_count: usize) {
        self.weights = Matrix::new(self.units_count, prev_units_count).randomize(0., 1.) * 0.01;
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
        let mut linear_activation = self.weights.dot(&mut input) + self.biases.clone();

        let output = match self.activation_type {
            ActivationType::Sigmoid => {
                Layer::sigmoid(linear_activation.clone())
            },
            ActivationType::Tanh => {
                Layer::tanh(linear_activation.clone())
            },
            ActivationType::Relu => {
                Layer::relu(linear_activation.clone())
            },
            ActivationType::Linear => {
                linear_activation.clone()
            },
        };

        self.cache = Cache {
            // prev activation
            input,
            // current activation
            output: output.clone(),
        };

        output
    }

    pub fn backward(&mut self, mut activation: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
        let m = activation.shape.cols as f64;

        let mut d_activation = match self.activation_type {
            ActivationType::Sigmoid => {
                let output = self.cache.output.clone();
                activation * output.clone() * (1. - output)
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
        let d_weights = d_activation.dot(&mut self.cache.input.transpose()) / m;
        // db
        let d_biases = d_activation.x_axis_sum() / m;
        let d_activation_prev = self.weights.transpose().dot(&mut d_activation);

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
    // let input_data: Vec<Vec<f64>> = read_vector_from_file("./src/input_128x128.bin").unwrap();
    // let output_data: Vec<f64> = read_vector_from_file("./src/output_128x128.bin").unwrap();

    // let (input_data, output_data) = generate_image_dataset(vec![
    //     "./src/dataset/test/cats",
    //     "./src/dataset/test/dogs"
    // ]);
    //
    // write_vector_to_file(input_data.clone(), "input_128x128_test.bin").unwrap();
    // write_vector_to_file(output_data.clone(), "output_128x128_test.bin").unwrap();
    //
    // let (input_data, output_data) = generate_image_dataset(vec![
    //     "./src/dataset/train/cats",
    //     "./src/dataset/train/dogs"
    // ]);
    //
    // write_vector_to_file(input_data.clone(), "input_128x128.bin").unwrap();
    // write_vector_to_file(output_data.clone(), "output_128x128.bin").unwrap();

    let learning_rate = 0.075;
    let iterations_count = 1000;
    let features_count = input_data[1].len();

    let mut input: Matrix<f64> = Matrix::new(input_data[0].len(), input_data.len());
    input.assign(input_data.into_iter().flatten().collect());

    let mut output: Matrix<f64> = Matrix::new(1, output_data.len());
    output.assign(output_data);

    let mut nn = NeuralNetwork::new(features_count, vec![
        Layer::new(5, ActivationType::Relu),
        Layer::new(30, ActivationType::Relu),
        Layer::new(2, ActivationType::Relu),
        Layer::new(1, ActivationType::Sigmoid),
    ]);

    nn.train(input / 255., output.clone(), learning_rate, iterations_count);

    // check accuracy with test data
    // let input_data: Vec<Vec<f64>> = read_vector_from_file("./src/input_128x128_test.bin").unwrap();
    // let mut input: Matrix<f64> = Matrix::new(input_data[0].len(), input_data.len());
    // input.assign(input_data.into_iter().flatten().collect());
    //
    // let output_data: Vec<f64> = read_vector_from_file("./src/output_128x128_test.bin").unwrap();
    // let mut output: Matrix<f64> = Matrix::new(1, output_data.len());
    // output.assign(output_data);
    //
    // nn.predict(input / 255., output);

    // check accuracy on own data
    // let input_data: Vec<Vec<f64>> = vec![
    //     image_to_vector("./src/dataset/test_cat.jpg"),
    //     image_to_vector("./src/dataset/dog1.jpg"),
    //     image_to_vector("./src/dataset/test_cat2.jpeg"),
    //     image_to_vector("./src/dataset/test_cat3.jpg"),
    //     image_to_vector("./src/dataset/test_cat4.jpg"),
    //     image_to_vector("./src/dataset/test_cat5.jpg"),
    //     image_to_vector("./src/dataset/test_cat6.png"),
    // ];
    // let mut input: Matrix<f64> = Matrix::new(input_data[0].len(), input_data.len());
    // input.assign(input_data.into_iter().flatten().collect());
    //
    // let output_data: Vec<f64> = vec![1., 0., 1., 1., 0., 0., 1.];
    // let mut output: Matrix<f64> = Matrix::new(1, output_data.len());
    // output.assign(output_data);
    //
    // let predictions = nn.predict(input / 255., output);
    // println!("{:?}", predictions);
}