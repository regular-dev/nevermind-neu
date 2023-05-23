# MNIST digits classification example

MNIST example code just serializing data to protobuf, which could be passed to neural network with
compiled tool binary. The below instructions decribes how to create simple net architechture with 
softmax loss (cross-entropy), and train it to classify handwritten digits.

1) We need to create directory, where we will download mnist archives and extract them. In nevermind-neu root directory type:

`mkdir mnist_data`

`cd mnist_data`

`wget https://github.com/sunsided/mnist/raw/master/t10k-images-idx3-ubyte.gz`

`gzip -d t10k-images-idx3-ubyte.gz`

`wget https://github.com/sunsided/mnist/raw/master/t10k-labels-idx1-ubyte.gz`

`gzip -d t10k-labels-idx1-ubyte.gz`

`wget https://github.com/sunsided/mnist/raw/master/train-images-idx3-ubyte.gz`

`gzip -d train-images-idx3-ubyte.gz`

`wget https://github.com/sunsided/mnist/raw/master/train-labels-idx1-ubyte.gz`

`gzip -d train-labels-idx1-ubyte.gz`

You can download those archives from another location, but anyway put them into mnist_data/ directory.

2) Then serialize mnist data to protobuf

`cargo run --release --example mnist`

On successfull execution you will get two serialized protobuf files.

3) Create network architechture and optimizer configuration with tool binary
4) Train network
5) Test network

TODO ...
