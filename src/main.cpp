#include "../include/MLP.h"

#include<Eigen/Dense>
#include<iostream>
#include <csignal>

using namespace std;

int main() {

    // Load training dataset and custom data.
    Dataset dataset(true);
    dataset.load_dataset();


    // Get all the data from the dataset flattened to a 784d vector arranged to a 785x60k matrix (label + 784 pixels).
    Eigen::MatrixXf data = dataset.get_learning_data(60000);

    // Since the dataset comes arranged by classes we shuffle to eliminate any bias.
    Dataset::shuffle(data);

    // Divide the data between learning data and validation data.
    Eigen::MatrixXf learning_data = data.block(0, 0, 785, 50000);
    Eigen::MatrixXf validation_data = data.block(0, 50000, 785, 10000);

    // Get the testing data.
        Eigen::MatrixXf testing_data = dataset.get_testing_data();

    Eigen::MatrixXf custom_images = dataset.get_custom_data();

    MLP::NetworkParameters parameters{
        .regularization = MLP::Regularization::L2,
        .regularization_parameter = 0.01,
        .initializer = MLP::Initializer::HE,
        .dropout_regularization = true,
        .lambda = 0.1
    };



    // Create a network.
    MLP network(MLP::InputLayer(784),
    {MLP::DenseLayer(30, MLP::ActivationFunction::RELU)},
                MLP::OutputLayer(10, MLP::ActivationFunction::SIGMOID, MLP::LossFunction::BINARY_CROSS_ENTROPY),
                parameters);

    // network.set_minibatch_log_point(100);
    // network.set_testing_log_point(1000);
    //

    MLP::Hyperparameters hyper_parameters{.mini_batch_size = 50, .learning_rate = 0.1, .epochs = 100};
    // Start training the network.
    network.train(MLP::Optimizer::SGD, learning_data, validation_data, testing_data, hyper_parameters);



    // for (int i = 0; i < custom_images.cols(); i++) {
    //     const int out = network.classify(custom_images.block(0, i, 784, 1));
    //     cout << "Prediction of the image below: " << out << endl;
    //     Dataset::print_image(custom_images.block(0, i, 784, 1));
    //     cout << endl;
    // }


}