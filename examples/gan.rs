use clap::Parser;
use ndarray::Array2;
use ndarray::{Array, ArrayView};
use rand::{distributions::Distribution, thread_rng};
use rand_distr::Normal;
use rustgan::layers::{Linear, ReLU, Sigmoid};
use std::str::FromStr;
use std::{fs::File, io::Read};

#[derive(Parser, Debug)]
struct Args {
    /// The path to a CSV file containing MNIST training images of only one label (i.e. only 5s)
    #[arg(short, long)]
    mnist_path: String,
    /// The path under which to save intermediate training images
    #[arg(short, long)]
    inter_path: String,
}

use rustgan::networks::GenerativeAdverserial;
pub struct Dataset {
    pub data: Array2<f64>,
    pub target: Array2<f64>,
}

// Helper trait on Array2 in order to sample random batches from the dataset
// and to sample noise from the dataset
trait SampleRandom {
    // Sample m random gaussian noise vectors each of dimension n
    fn noise_sample(m: usize, n: usize) -> Self;
}

impl SampleRandom for Array2<f64> {
    fn noise_sample(m: usize, n: usize) -> Self {
        let dist = Normal::new(0f64, 1f64).unwrap();
        let mut rng = thread_rng();
        let noise = Array2::from_shape_fn((m, n), |_| dist.sample(&mut rng));

        noise
    }
}

const NUM_FEATURES: usize = 784;
const LINE_SIZE: usize = 785;
const NUM_CLASSES: usize = 10;
const GREYSCALE_SIZE: f64 = 255f64;

/// Parse a record (e.g. CSV record) of the form <x1><sep><x2><sep>...
/// Returns a vector of the xi's if the function was succesful
/// and None otherwise
fn parse_line<T: FromStr>(s: &str, seperator: char) -> Option<Vec<T>> {
    let mut record = Vec::<T>::new();

    for x in s.split(seperator) {
        match T::from_str(x) {
            Ok(val) => {
                record.push(val);
            }
            _ => return None,
        }
    }

    Some(record)
}

/// Parse a line in the dataset. Return the pixels and the label
/// Line is stored in the format: <label>,<pixel0x0>,<pixel0x1>,...
/// The dataset is taken from here https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
fn parse_dataset_line(line: &str) -> Option<(Vec<f64>, f64)> {
    match parse_line(line, ',') {
        Some(v) => match v.len() {
            // we divide by 255 to normalize
            LINE_SIZE => Some((
                v[1..LINE_SIZE].iter().map(|x| x / GREYSCALE_SIZE).collect(),
                v[0],
            )),
            _ => None,
        },
        _ => None,
    }
}

// Return matrix that represents the dataset
pub fn parse_dataset(path: &str) -> Dataset {
    let file = File::open(path);
    let mut data = Array::zeros((0, NUM_FEATURES));
    let mut target = Array::zeros((0, NUM_CLASSES));
    let mut contents = String::new();

    file.unwrap().read_to_string(&mut contents).unwrap();

    for line in contents.lines().skip(1).take_while(|x| !x.is_empty()) {
        let line = parse_dataset_line(line).unwrap();
        let pixels = line.0;
        let label = line.1 as usize;
        // Construct one-hot encoding for the label
        let one_hot_target: Vec<f64> = (0..NUM_CLASSES)
            .map(|idx| if idx == label { 1f64 } else { 0f64 })
            .collect();

        data.push_row(ArrayView::from(&pixels)).unwrap();
        target.push_row(ArrayView::from(&one_hot_target)).unwrap();
    }

    Dataset { data, target }
}

fn main() {
    let args = Args::parse();
    let dataset = parse_dataset(&args.mnist_path);

    let mut gan = GenerativeAdverserial::new(50, 64, 0.003, Some(args.inter_path));
    // Our discriminator
    gan.discriminator.add_layer(Box::new(Linear::new(784, 500)));
    gan.discriminator.add_layer(Box::new(ReLU::new()));
    gan.discriminator.add_layer(Box::new(Linear::new(500, 500)));
    gan.discriminator.add_layer(Box::new(ReLU::new()));
    gan.discriminator.add_layer(Box::new(Linear::new(500, 1)));
    gan.discriminator.add_layer(Box::new(Sigmoid::new()));
    // Our generator
    gan.generator.add_layer(Box::new(Linear::new(100, 500)));
    gan.generator.add_layer(Box::new(ReLU::new()));
    gan.generator.add_layer(Box::new(Linear::new(500, 500)));
    gan.generator.add_layer(Box::new(ReLU::new()));
    gan.generator.add_layer(Box::new(Linear::new(500, 784)));
    gan.generator.add_layer(Box::new(Sigmoid::new()));
    // Train it

    gan.fit(dataset.data);

    // Generate an image
    let noise_batch = Array2::noise_sample(1, 100);
    let gen = gan.generate(noise_batch);

    for x in gen {
        print!("{},", x);
    }
}
