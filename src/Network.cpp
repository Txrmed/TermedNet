//
// Created by termed on 2.05.2026.
//

#include "../include/Network.h"
#include "../include/Dataset.h"
#include "../include/Hyperparams.h"
#include "../include/NetworkState.h"
#include "../include/Deltas.h"

#include<Eigen/Dense>
#include<vector>
#include<iostream>
#include<algorithm>

// change this value to change the range of the random weights and biases.
#define MAX 0.1

using namespace std;


Network::Network(const vector<unsigned int>& neurons, const Hyperparams& parameters, const Dataset& dataset) : dataset_(dataset),
    parameters_(parameters), layers_(neurons.size()), neurons_(neurons) {

    cout << "Loading data" << endl;
    dataset_.load_dataset();
    cout << "Finished loading data." << endl;

    // Initialize biases and weights vectors to random values.
    for (int i = 1; i < layers_; i++) {
        Eigen::VectorXf bias_vector = Eigen::VectorXf::Random(neurons_[i]);
        bias_vector = bias_vector * MAX;
        biases_.push_back(bias_vector);
    }

    for (int i = 1; i < layers_; i++) {
        Eigen::MatrixXf weights_matrix = Eigen::MatrixXf::Random(neurons_[i], neurons_[i - 1]);
        weights_matrix = weights_matrix * MAX;
        weights_.push_back(weights_matrix);
    }
};


Deltas Network::backpropagation(const NetworkState& state) {

    Deltas d;
    vector<Eigen::MatrixXf> d_weights;
    vector<Eigen::VectorXf> d_biases;

    d_weights.reserve(layers_ - 1);
    d_biases.reserve(layers_ - 1);

    // Calculate error in the output layer.
    Eigen::MatrixXf error_output_layer = (state.output - state.label).cwiseProduct(state.output.unaryExpr([this](const float x){ return this->sigmoid_prime(x);} ));

    vector<Eigen::MatrixXf> errors;
    errors.push_back(error_output_layer);

    Eigen::MatrixXf error_next_layer = std::move(error_output_layer);

    for (int i = layers_ - 2; i > 0; i--) {
        Eigen::MatrixXf error_l = (weights_[i].transpose() * error_next_layer).cwiseProduct(state.zs[i - 1].unaryExpr([this](const float x){ return this->sigmoid_prime(x);}));
        errors.push_back(error_l);
        error_next_layer = std::move(error_l);
    }

    reverse(errors.begin(), errors.end());

    for (int i = 0; i < layers_ - 1; i++) {
        Eigen::VectorXf db = errors[i];
        Eigen::MatrixXf dw = errors[i] * state.activations[i].transpose();
        d_weights.push_back(dw);
        d_biases.push_back(db);
    }

    d.d_weights = std::move(d_weights);
    d.d_biases = std::move(d_biases);

    return d;
}



NetworkState Network::feedforward(Eigen::Ref<const Eigen::VectorXf> input, unsigned int label) const {

    vector<Eigen::VectorXf> zs;
    zs.reserve(layers_);

    vector<Eigen::VectorXf> activations;
    activations.reserve(layers_);

    activations.push_back(input);

    Eigen::VectorXf prev = input;

    for (int i = 0; i < layers_ - 1; i++) {

        // Activation before applying sigmoid function.
        Eigen::VectorXf z = (weights_[i] * prev) + biases_[i];
        zs.push_back(z);

        // Activations after applying sigmoid function.
        Eigen::VectorXf a = z.unaryExpr([this](const float x){return 1 / (1 + pow(e_, -x)); });
        activations.push_back(a);

        // Next iteration assignment.
        prev = std::move(a);
    }

    // Create a label vector with 1 placed at the index of the digit.
    Eigen::VectorXf label_vector(10);
    label_vector.setZero();
    label_vector(label, 0) = 1.0;

    // State
    NetworkState state{std::move(zs), std::move(activations), std::move(prev), std::move(label_vector)};

    return state;
}


void Network::train() {
    // Calculate how many mini batches fit inside one epoch.
    const unsigned int num_mini_batches = parameters_.num_learning_img / parameters_.mini_batch_size;

    // Iterate parameters.epochs times.
    for (int epoch_i = 0; epoch_i < parameters_.epochs; epoch_i++) {
        float total_cost = 0;

        // Iterate over the number of mini batches
        for (int minibatch_i = 0; minibatch_i < num_mini_batches; minibatch_i++) {

            // Load minibatch from dataset
            Eigen::MatrixXf mini_batch = dataset_.get_mini_batch(parameters_.mini_batch_size);

            // Vectors for storing changes to weights and biases.
            vector<Eigen::MatrixXf> dw;
            vector<Eigen::VectorXf> db;

            for (int populate_i = 0; populate_i < layers_ - 1; populate_i++) {
                Eigen::MatrixXf w(weights_[populate_i].rows(), weights_[populate_i].cols());
                w.setZero();

                Eigen::VectorXf b(biases_[populate_i].rows());
                b.setZero();

                dw.push_back(w);
                db.push_back(b);
            }

            // Iterate over images in one mini batch
            for (int data_i = 0; data_i < parameters_.mini_batch_size; data_i++) {

                NetworkState state = feedforward(mini_batch.col(data_i)(Eigen::seq(1, PIXEL_COUNT + 1)),
                                                                               static_cast<unsigned int>(mini_batch(0, data_i)));
                Deltas deltas = backpropagation(state);

                total_cost += cost_function(state.output, state.label);

                for (int l = 0; l < layers_ - 1; l++) {
                    dw[l] = dw[l] + deltas.d_weights[l];
                    db[l] = db[l] + deltas.d_biases[l];
                }

            }
            for (int l = 0; l < layers_ - 1; l++) {
                weights_[l] = weights_[l] - (parameters_.learning_rate / parameters_.mini_batch_size) * dw[l];
                biases_[l] = biases_[l] - (parameters_.learning_rate / parameters_.mini_batch_size) * db[l];
            }

            if (minibatch_i % 500 == 0 && minibatch_i != 0) {
                cout << "Minibatch " << minibatch_i << " complete! " << "Average Cost function is: " << total_cost / 500 << "." << endl;
                total_cost = 0;
            }
        }
        // Shuffle the entire matrix.
        dataset_.shuffle();

        // Reset index to 0.
        dataset_.set_return();

        cout << "Epoch " << epoch_i + 1 << " finished." << endl;

        // Finally - test the model against testing data.
        int n_tests = 500;
        unsigned int correct = test_network(n_tests);
        float percentage = static_cast<float>(correct) / n_tests * 100;
        cout << "Correctly classified " << correct << "/" << n_tests << "( " << percentage << "% )" << " digits" << endl;
    }
}

int Network::classify(Eigen::Ref<const Eigen::VectorXf> input) const {
    const NetworkState s = feedforward(input(Eigen::seq(1, PIXEL_COUNT + 1)), static_cast<int>(input(0)));

    Eigen::Index predicted_label;
    s.output.maxCoeff(&predicted_label);

    return static_cast<int>(predicted_label);
}

unsigned int Network::test_network(unsigned int n) const {
    Eigen::MatrixXf tests = dataset_.get_testing_data(n);

    unsigned int correct = 0;

    for (int image_i = 0; image_i < n; image_i++) {
        // 1. Extract label and image data
        int label = static_cast<int>(tests(0, image_i));

        // Feedforward data and get the state of the network.
        Eigen::VectorXf image_data = tests.block(1, image_i, PIXEL_COUNT, 1);
        NetworkState state = feedforward(image_data, label);

        // Get index of the highest activation.
        Eigen::Index predicted_label;
        state.output.maxCoeff(&predicted_label);

        // Compare prediciton to label.
        if (static_cast<unsigned int>(predicted_label) == label) {
            correct++;
        }
    }
    return correct;
}
