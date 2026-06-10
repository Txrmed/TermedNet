//
// Created by termed on 2.05.2026.
//

#include "MLP.h"
#include "../include/MLP.h"
#include "../include/Logger.h"

#include<Eigen/Dense>
#include<vector>
#include<algorithm>
#include<cmath>
#include <functional>



using namespace std;

MLP::MLP(InputLayer input_layer, std::initializer_list<DenseLayer> dense_layers, OutputLayer output_layer,
         NetworkParameters p, bool verbose) : output_layer_(output_layer), input_layer_(input_layer), network_parameters_(p), logger(verbose) {
    if (network_parameters_.regularization != Regularization::NONE && network_parameters_.regularization_parameter == 0) {
        throw invalid_argument("Cant use regularization parameter set to 0");
    }
    if (network_parameters_.dropout_regularization && network_parameters_.lambda <= 0 || network_parameters_.lambda >= 1) {
        throw invalid_argument("Cant use lambda < 0 or lambda > 1 for dropout.");
    }

    // Add the amount of neurons in the input layer.
    neurons_.push_back(input_layer.neuron_amount_);

    // General case for adding the amount of neurons in the dense layers.
    for (auto layer : dense_layers) {
        neurons_.push_back(layer.neuron_amount_);
        dense_layers_.push_back(layer);
    }

    // Add the amount of neurons in the output layer.
    neurons_.push_back(output_layer.neuron_amount_);

    layers_ = neurons_.size();

    if (network_parameters_.dropout_regularization) {
        dropout_mask_.reserve(layers_ - 2);
    }

    // Initialize the weights
    for (size_t i = 1; i < neurons_.size(); i++) {
        Eigen::MatrixXf weights;
        switch (network_parameters_.initializer) {
            case Initializer::RANDOM:
                weights = Eigen::MatrixXf::Random(neurons_[i], neurons_[i - 1]);
                break;
            case Initializer::HE:
                weights = Eigen::MatrixXf::Random(neurons_[i], neurons_[i - 1]);
                weights *= std::sqrt(6.0f / neurons_[i - 1]);
                break;
            default:
                throw std::invalid_argument("Invalid initializer used.");
        }
        weights_.push_back(weights);
    }

    // Initialize the biases.
    for (size_t i = 1; i < neurons_.size(); i++) {
        biases_.push_back(Eigen::VectorXf::Random(neurons_[i]));
    }

    logger.log(Logger::TRAINING) << "Initialized weights and biases for each layer." << endl;

    // Initialize the activation functions vector.
    for (size_t i = 0; i < neurons_.size() - 2; i++) {
            switch (dense_layers_[i].activation_function_) {
                case ActivationFunction::SIGMOID:
                    activation_.push_back(&MLP::sigmoid);
                    activation_prime_.push_back(&MLP::sigmoid_prime);
                    break;
                case ActivationFunction::RELU:
                    activation_.push_back(&MLP::relu);
                    activation_prime_.push_back(&MLP::relu_prime);
                    break;
                case ActivationFunction::TANH:
                    activation_.push_back(&MLP::tanh);
                    activation_prime_.push_back(&MLP::tanh_prime);
                    break;
                case ActivationFunction::SOFTMAX:
                    throw std::invalid_argument("Dont use softmax in the hidden layers ( I dont want to write the derivatives )");
                default:
                    throw std::invalid_argument("dense layers");
        }
    }
    switch (output_layer_.activation_function_) {
        case ActivationFunction::SIGMOID:
            activation_.push_back(&MLP::sigmoid);
            activation_prime_.push_back(&MLP::sigmoid_prime);
            break;
        case ActivationFunction::RELU:
            activation_.push_back(&MLP::relu);
            activation_prime_.push_back(&MLP::relu_prime);
            break;
        case ActivationFunction::TANH:
            activation_.push_back(&MLP::tanh);
            activation_prime_.push_back(&MLP::tanh_prime);
            break;
        case ActivationFunction::SOFTMAX:
            break;
        default:
            throw std::invalid_argument("final layers");
    }

    switch (output_layer_.loss_function_) {
        case LossFunction::QUADRATIC:
            loss_function = &MLP::quadratic_cost_numerical;
            break;
        case LossFunction::BINARY_CROSS_ENTROPY:
            loss_function = &MLP::binary_cross_entropy_cost_numerical;
            break;
        default:
            throw std::invalid_argument("Invalid cost function used.");
    }


}

[[nodiscard]] inline float MLP::quadratic_cost_numerical(const Eigen::VectorXf& a, const Eigen::VectorXf& label) {
    return 0.5f * (a - label).array().square().sum();
}

[[nodiscard]] inline float MLP::binary_cross_entropy_cost_numerical(const Eigen::VectorXf& a, const Eigen::VectorXf& label) {
    float epsilon = 1e-5f;

    // Clamp 'a' cleanly between [epsilon] and [1 - epsilon]
    auto clamped_a = a.array().cwiseMax(epsilon).cwiseMin(1.0f - epsilon);

    // One-shot SIMD vector calculation
    float c = ((label.array() * clamped_a.log()) + ((1.0f - label.array()) * (1.0f - clamped_a).log())).sum();

    return c * -1.0f;
}


MLP::NetworkState MLP::feedforward(Eigen::Ref<const Eigen::VectorXf> input, size_t label, bool learning_mode) const {

    // Vector storing an Eigen::Vector of every z.
    vector<Eigen::VectorXf> zs;
    zs.reserve(layers_);

    // Vector storing an Eigen::Vector of every activation.
    vector<Eigen::VectorXf> activations;
    activations.reserve(layers_);


    activations.push_back(input);

    Eigen::VectorXf prev = input;

    Eigen::VectorXf a, z;

    // Calculate activations and z's for dense layers.
    for (size_t i = 0; i < layers_ - 2; i++) {

        // Activation before applying the activation function.
        z = (weights_[i] * prev) + biases_[i];
        zs.push_back(z);

        // Activations  after applying activation function.

        a = z.unaryExpr([this, i](const float x){return (*activation_[i])(x); });
        if (network_parameters_.dropout_regularization && learning_mode) {
            a = a.array() * dropout_mask_[i].array();
        }
        activations.push_back(a);

        // Next iteration assignment.
        prev = a;
    }

    // Calculate z in the last layer.
    z = (weights_.back() * prev) + biases_.back();
    zs.push_back(z);

    // Apply the specified activation function for the last layer.
    if (output_layer_.activation_function_ == ActivationFunction::SOFTMAX) {
        float max = z.maxCoeff();
        Eigen::VectorXf exp_z = (z.array() - max).exp();
        a = exp_z / exp_z.sum();
    }
    else {
        a = z.unaryExpr([this](const float x){return (*activation_.back())(x); });
    }

    activations.push_back(a);

    prev = a;

    // Create a label vector with 1 placed at the index of the digit.
    Eigen::VectorXf label_vector(10);
    label_vector.setZero();
    label_vector(label, 0) = 1.0;

    // State
    NetworkState state{std::move(zs), std::move(activations), std::move(prev), std::move(label_vector)};

    return state;
}


MLP::Deltas MLP::backpropagation(const NetworkState& state) {

    Deltas d;
    d.d_weights.reserve(layers_ - 1);
    d.d_biases.reserve(layers_ - 1);

    // Calculate error in the output layer.
    Eigen::MatrixXf error_next_layer;
    switch (output_layer_.loss_function_) {
        case LossFunction::QUADRATIC:
            error_next_layer = (state.output - state.label).cwiseProduct(
                state.zs.back().unaryExpr([this](const float z) { return (*activation_.back())(z); }));
            break;
        case LossFunction::BINARY_CROSS_ENTROPY:
            error_next_layer = state.output - state.label;
            break;
        default:
            throw std::invalid_argument("No such loss function.");
    }

    vector<Eigen::MatrixXf> errors;
    errors.push_back(error_next_layer);

    for (int i = layers_ - 2; i > 0; i--) {
        Eigen::MatrixXf error_l = (weights_[i].transpose() * error_next_layer)
                                    .cwiseProduct(state.zs[i - 1].unaryExpr([this, i](const float x){ return (*activation_prime_[i - 1])(x);}));
        errors.push_back(error_l);
        error_next_layer = error_l;
    }

    reverse(errors.begin(), errors.end());

    if (network_parameters_.dropout_regularization) {
        for (int i = 0; i < layers_ - 2; i++) {
            errors[i] = errors[i].array() * dropout_mask_[i].array();
        }
    }

    for (size_t i = 0; i < layers_ - 1; i++) {
        Eigen::VectorXf db = errors[i];
        Eigen::MatrixXf dw = errors[i] * state.activations[i].transpose();
        d.d_weights.push_back(dw);
        d.d_biases.push_back(db);
    }

    return d;
}


void MLP::train(Optimizer optimizer, const Eigen::MatrixXf& learning_matrix, const Eigen::MatrixXf& validation_matrix ,const Eigen::MatrixXf& testing_matrix,const Hyperparameters& hyperparameters) {

    Eigen::MatrixXf learning_data = learning_matrix;
    Eigen::Matrix validation_data = validation_matrix;
    Eigen::MatrixXf testing_data = testing_matrix;


    // Calculate how many mini batches fit inside one epoch.
    const size_t num_mini_batches = static_cast<int>( learning_data.cols() / hyperparameters.mini_batch_size);



    // Vectors for storing changes to weights and biases.
    vector<Eigen::MatrixXf> dw;
    vector<Eigen::VectorXf> db;

    dw.reserve(layers_ - 1);
    db.reserve(layers_ - 1);

    for (size_t populate_i = 0; populate_i < layers_ - 1; populate_i++) {
        Eigen::MatrixXf w(weights_[populate_i].rows(), weights_[populate_i].cols());
        w.setZero();

        Eigen::VectorXf b(biases_[populate_i].rows());
        b.setZero();

        dw.push_back(w);
        db.push_back(b);
    }

    // Iterate parameters.epochs times.
    for (size_t epoch_i = 0; epoch_i < hyperparameters.epochs; epoch_i++) {

        // Index used to track the currect column in the learning_data.
        size_t index = 0;

        // Iterate over the number of mini batches
        for (size_t minibatch_i = 0; minibatch_i < num_mini_batches; minibatch_i++) {
            float total_cost = 0;

            for (size_t i = 0; i < layers_ - 1; i++) {
                dw[i].setZero();
                db[i].setZero();
            }

            // Load minibatch from dataset
            Eigen::MatrixXf mini_batch = learning_data.block(0, index, PIXEL_COUNT + 1, hyperparameters.mini_batch_size);
            index += hyperparameters.mini_batch_size;

            // Iterate over images in one mini batch
            for (size_t data_i = 0; data_i < hyperparameters.mini_batch_size; data_i++) {


                if (network_parameters_.dropout_regularization) {
                    dropout_mask_.clear();
                    for (int i = 0; i < layers_ - 2; i++) {
                        dropout_mask_.push_back(generate_dropout_mask(neurons_[i + 1], network_parameters_.lambda, i));
                    }
                }

                NetworkState state = feedforward(mini_batch.block(1, data_i, PIXEL_COUNT, 1),
                                                 static_cast<unsigned int>(mini_batch(0, data_i)), true);

                auto [d_biases, d_weights] = backpropagation(state);

                // Keep track of the total loss function to see if the model is learning anything.
                total_cost += loss_function(state.output, state.label);

                for (size_t l = 0; l < layers_ - 1; l++) {
                    dw[l] = dw[l] + d_weights[l];
                    db[l] = db[l] + d_biases[l];
                }

            }
            for (size_t l = 0; l < layers_ - 1; l++) {
                switch (network_parameters_.regularization){
                    case Regularization::NONE:
                        weights_[l] = weights_[l] - (hyperparameters.learning_rate / hyperparameters.mini_batch_size) * dw[l];
                        biases_[l] = biases_[l] - (hyperparameters.learning_rate / hyperparameters.mini_batch_size) * db[l];
                        break;
                    case Regularization::L1:
                        weights_[l] = weights_[l] - (hyperparameters.learning_rate / hyperparameters.mini_batch_size) * dw[l];
                        biases_[l] = biases_[l] - (hyperparameters.learning_rate / hyperparameters.mini_batch_size) * db[l];
                        break;
                    case Regularization::L2:
                        weights_[l] = ((1 - hyperparameters.learning_rate * network_parameters_.regularization_parameter / learning_data.cols()) * weights_[l])
                                                                    -
                                          (hyperparameters.learning_rate / hyperparameters.mini_batch_size) * dw[l];
                        biases_[l] = biases_[l] - (hyperparameters.learning_rate / hyperparameters.mini_batch_size) * db[l];
                }
            }

            int a = 100;
            if (minibatch_i % a == 0 && minibatch_i != 0) {
                logger.log(Logger::Prefix::TRAINING) << "Minibatch " << minibatch_i << " complete! " <<
                        "Average Cost function is: " << total_cost / (a * hyperparameters.mini_batch_size) <<  " over 100 minibatches." << endl;
                total_cost = 0;
            }
        }

        // Test the network using the validation data.
        int n_tests = 10000;
        size_t correct_vdata = test_network(validation_data);
        float percentage_vdata = static_cast<float>(correct_vdata) / n_tests * 100.0f;

        // Test the network using the testing data.
        size_t correct_tdata = test_network(testing_data);
        float percentage_tdata = static_cast<float>(correct_tdata) / n_tests * 100.0f;


        // Shuffle the entire matrix.
        Dataset::shuffle(learning_data);

        logger.log(Logger::Prefix::TRAINING) << "Epoch " << epoch_i + 1 << "/" << hyperparameters.epochs << " finished." << endl;
        logger.log(Logger::TESTING) << "Correctly classified " << correct_tdata << "(" << percentage_tdata << "%) testing images." << endl;
        logger.log(Logger::TESTING) << "Correctly classified " << correct_vdata << "(" << percentage_vdata << "%) validation images." << endl;

    }
    // cout << "Costs : " << endl;
    // cout << "[";
    // for (auto i : costs) {
    //     cout << i << ", " ;
    // }
    // cout << "]" << endl;
    //
    // cout << "Accuracies : " << endl;
    // cout << "[";
    // for (auto i : accuracy) {
    //     cout << i << ", " ;
    // }
    // cout << "]" << endl;

}

int MLP::classify(Eigen::Ref<const Eigen::VectorXf> input) const {
    const NetworkState s = feedforward(input.block(0, 0, PIXEL_COUNT, 1), 0, false);

    Eigen::Index predicted_label;
    s.output.maxCoeff(&predicted_label);

    return static_cast<int>(predicted_label);
}

size_t MLP::test_network(Eigen::MatrixXf testing_data) const {

    size_t correct = 0;

    for (size_t image_i = 0; image_i < testing_data.cols(); image_i++) {
        // 1. Extract label and image data
        int label = static_cast<int>(testing_data(0, image_i));

        // Feedforward data and get the state of the network.
        Eigen::VectorXf image_data = testing_data.block(1, image_i, PIXEL_COUNT, 1);
        NetworkState state = feedforward(image_data, label, false);

        // Get index of the highest activation.
        Eigen::Index predicted_label;
        state.output.maxCoeff(&predicted_label);

        // Compare prediciton to label.
        if (static_cast<size_t>(predicted_label) == label) {
            correct++;
        }
    }
    return correct;
}

Eigen::VectorXf MLP::generate_dropout_mask(int rows, float p, int layer_n) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::bernoulli_distribution dist(1 - p);

    Eigen::VectorXf dropout_mask(rows);

    if (layer_n == 0) {
        dropout_mask = Eigen::VectorXf::NullaryExpr(rows, [&](){
            return dist(gen) ? 1.0f : 0.0f;});
    }
    else {
        dropout_mask = Eigen::VectorXf::NullaryExpr(rows, [&](){
            return dist(gen) ? 1 + 1 / 1 - p : 0.0f;});
    }

    return dropout_mask;

}
