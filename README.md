# nevermind-neu
[![crates.io](https://img.shields.io/crates/v/nevermind-neu.svg)](https://crates.io/crates/nevermind-neu)

Machine learning library and tool with terminal user interface written in rust. It supports OpenCL layers (beta WIP) and CPU layers.
Core math matrix library is **ndarray** which use **matrixmultiply** crate for CPU matrix multiplication.

## Design goals
  - Fast optimized computations
  - User-friendly API
  - Provide utility terminal application to create, train, manage models with user-friendly terminal interface

## OpenCL
OpenCL support is based on **ocl** crate. It is optional feature and enabled by default.

## Features
 - FullyConnected layer
 - Euclidean Loss, Softmax Loss
 - Optimizers: Adam, RMSProp, AdaGrad, AdaDelta
 - (De)Serializing neural network state to protobuf
 - (De)Serializing neural network configuration net yaml file
 - Activation functions : *sigmoid, tanh, relu, leaky_relu*

## Terminal user interface tool
![tui](https://github.com/regular-dev/nevermind-neu/blob/master/doc/tui_train.gif?raw=true)

## Roadmap
  - Conv2D layer
  - RNN + LSTM
  - Residual block
  - OpenCL optimization

## License
Apache License Version 2.0
