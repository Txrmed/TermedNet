# TermedNet

Very basic implementation of a Multi-Layer Perceptron Neural Network.

Implements the backpropagation algorithm to train an arbitrary MLP network on an arbitrary dataset which is labeled and can be flattened to a X-Dimensional vector.

Backpropagation is implemented with stochastic gradient descent using a configurable mini batch size, which defaults to 10.

I achieved peak performance of around 97.5% on the MNIST dataset using a network with 784 in the input layer, 600, 400, 200, 50 neurons using ReLu in the hidden layers and 10 neurons in the output layer.

Matrix operations are provided by the Eigen library. The entire project is written in C++, but the goal is for the library to be accessible in Python.

# USAGE
## Loading the MNIST dataset
The network comes provided with a dataset loader and the MNIST data in the /data/mnist catalog.

    Dataset dataset();
Load all detected data.


    dataset.load_dataset();
The learning and training data comes in a matrix consisting of respectively 60k and 10k columns, each representing a flattened image, with the label in the 0th row and the pixels in rows 1 through 785.

    Eigen::MatrixXf data = dataset.get_learning_data(60000);
If you use validation data to prevent overfitting you can divide the dataset into learning and validation data, but first shuffle the `data` matrix because MNIST data comes grouped by classes.

    Dataset::shuffle(data);
    Eigen::MatrixXf learning_data = data.block(0, 0, 785, 50000);  
    Eigen::MatrixXf validation_data = data.block(0, 50000, 785, 10000); 
To load testing data do

    Eigen::MatrixXf testing_data = dataset.get_testing_data();

If you have created custom images (only PGM P5 format for now) you can load them using

    Eigen::MatrixXf custom_images = dataset.get_custom_data();
## Creating the network
All fields in the parameters struct have a default value but if you want to change them:

    MLP::NetworkParameters parameters{  
	  .regularization = MLP::Regularization::L2,  
	    .regularization_parameter = 0.01,  
	    .initializer = MLP::Initializer::HE,  
	    .dropout_regularization = true,  
	    .lambda = 0.1  
	};
Regularization options include:

    enum class Regularization {  
    NONE, ///< No regularization  
	  L1, ///< L1 regularization #Not implemented yet :( 
	  L2, ///< L2 regularization  
	};
The regularization parameter only applies if regularization other than NONE has been chose,
Initializer is used to initialize the weights accordingly, options include

    enum class Initializer {  
    RANDOM, ///< Randomly initialize weights to values between -1 and 1. Not recommended for ReLu  
	  HE, ///< He initialization 
	  GLOROT ///< Glorot initialization. Not implemented yet :(  
	};
Dropout regularization is separate from other regularization methods due to how it is implemented.
The NetworkParameters struct defaults to this:

    struct NetworkParameters {  
     
	  Optimizer optimizer = Optimizer::SGD;  // This will be implemented to use different optimizers like Adam etc.
	 
	  Regularization regularization = Regularization::NONE;  
	  
	  float regularization_parameter = 0;  
	  
	  Initializer initializer = Initializer::RANDOM;  
	  
	  bool dropout_regularization = true;  
	    
	  float lambda = 0.5;  
	};

To create the network use the only defined constructor, n is the amount of neurons in the created layer.

    MLP network(MLP::InputLayer(n), {MLP::DenseLayer(n, activation_function), ...}, 
    MLP::OutputLayer(n, activation_function, cost_function), parameters_struct));
You can create an arbitrary amount of DenseLayers, but performance does drop significantly after around a million parameters (weights and biases).

# Training the network
Train the network with

    network.train(MLP::Optimizer::SGD, learning_data, validation_data, testing_data, hyper_parameters);
Providing the optimizer here is due to my laziness of not removing it after experimenting a bit.
The hyperparameters struct looks like this:

    struct Hyperparameters {  
	    std::size_t mini_batch_size = 10; ///< Mini batch size (default is 10).  
  
		  float learning_rate; ///< Learning rate used when updating weights and biases.  
  
		  int epochs; ///< Amount of epochs.  
	};

# Other cool functions which  I implemented but regret because I could have implemented even cooler functions but the function which I was implementing seemed cool at the time but it really was not.
You can classify a vector containing an image with

    classify(Eigen::Ref<const Eigen::VectorXf> input)
Remember that the input to this has to come ***without*** the label.

You can also test the network with

    size_t test_network(Eigen::MatrixXf testing_data)
This has to come ***with*** the label, pretty self-explanatory as to why.
In the dataset class there's also a function to print a vector of pixels to stdout, use it like this

    print_image(Eigen::VectorXf image);
I dont remember if this can come with the label but probably not.

# Instalation
You need cmake, a c++ compiler and git which again is pretty self explanatory.

    git clone git@github.com:Txrmed/TermedNet.git
    cd TermedNet
    mkdir build
    cd build
    cmake ..
    make
    ./TermedNet
Currently this will only work on linux because of how the CMake file is configured, but some time soon it might work on Windows.

## Notes
This is in no way an efficient implementation, it comes closer to a somewhat bruteforce approach due to how the options are processed. However i am working on fixing this.

The long term goal of this project is to implement an efficient convolutional net similar to LeNet. 

