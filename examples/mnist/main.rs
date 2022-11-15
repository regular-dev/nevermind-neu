extern crate rand; // For initializing weights.

use std::error::Error;
use std::vec::Vec;

use std::ops::DerefMut;

use regular_mind::dataloader::*;
use regular_mind::util::*;

use rust_mnist::{print_sample_image, Mnist};

fn main() -> Result<(), Box<dyn Error>> {
    let mut train_loader = ProtobufDataLoader::empty();
    let mut test_loader = ProtobufDataLoader::empty();

    let mnist = Mnist::new("mnist_data/");

    let mut c = 0;

    for (x, y) in mnist.train_data.iter().zip(mnist.train_labels) {
        if c < 20 {
            println!("Test first out : {}, label is {}", c, y);
            print_sample_image(x, y);
            c += 1;
        } else {
            return Ok(()); // TODO : refactor test code
        }

        let mut inp: Vec<f32> = Vec::new();
        inp.reserve(x.len());

        for j in x {
            inp.push(*j as f32);
        }

        let mut expected: Vec<f32> = Vec::new();
        expected.resize(10, 0.0);
        expected[y as usize] = 1.0;

        let mut b = DataBatch::new(inp, expected);

        // normalizing
        minmax_normalize_params(&mut b.input, 0.0, 255.0);

        train_loader.data.push(b);
    }

    for (x, y) in mnist.test_data.iter().zip(mnist.test_labels) {
        let mut inp: Vec<f32> = Vec::new();
        inp.reserve(x.len());

        for j in x {
            inp.push(*j as f32);
        }

        let mut expected: Vec<f32> = Vec::new();
        expected.resize(10, 0.0);
        expected[y as usize] = 1.0;

        let mut b = DataBatch::new(inp, expected);

        // normalizing
        minmax_normalize_params(&mut b.input, 0.0, 255.0);

        test_loader.data.push(b);
    }

    train_loader.to_file("mnist_train.proto")?;
    test_loader.to_file("mnist_test.proto")?;

    Ok(())
}
