# MNIST digits classification example

MNIST example code just serializing data to protobuf, which could be passed to neural network with
compiled tool binary. The below instructions decribes how to create simple net architechture with
softmax loss (cross-entropy), and train it to classify handwritten digits.

1) We need to create directory, where we will download mnist archives and extract them. In nevermind-neu project directory type:

`mkdir mnist_data`

`cd mnist_data`

`curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz`

`gunzip t*-ubyte.gz`

`cd ..`

You can download those archives from another location, but anyway put them into mnist_data/ directory.

2) Then serialize mnist data to protobuf

`cargo run --release --example mnist`

On successfull execution you will get two serialized protobuf files.

3) Create network architechture and optimizer configuration. The input size should be 784 as MNIST provides 24x24 images, and output should be 10.

`cargo run --release create_net --ocl`

You can omit the ocl flag if you don't want your model use OpenCL

![mnist_net_example](https://github.com/regular-dev/nevermind-neu/blob/master/doc/mnist_net.png?raw=true)

4) Train network

`cargo run --release  train --train_data=mnist_train.proto --model=net.cfg --epochs=5 --opt=optim.cfg`

5) Test network

`cargo run --release test --dataset=mnist_test.proto --model=net.cfg --state=network_9375_final.state --samples=32`

![mnist_tests](https://github.com/regular-dev/nevermind-neu/blob/master/doc/mnist_net_test.png?raw=true)
