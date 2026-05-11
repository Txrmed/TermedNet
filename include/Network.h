//
// Created by termed on 2.05.2026.
//

#ifndef DIGITRECOGNITION_NETWORK_H
#define DIGITRECOGNITION_NETWORK_H

#include "../include/Dataset.h"
#include "../include/Hyperparams.h"
#include "../include/NetworkState.h"
#include "../include/Deltas.h"

#include<vector>
#include<Eigen/Dense>


class Network {
    float e_ = 2.71828;

    Dataset dataset_;

    Hyperparams parameters_;

    // Amount of layers in the network.
    const unsigned int layers_;

    // Amount of neurons at each layer.
    std::vector<unsigned int> neurons_;

    // weights and biases vectors.
    std::vector<Eigen::MatrixXf> weights_;
    std::vector<Eigen::VectorXf> biases_;

    /**
     *@brief Sigmoid function,
     *
     *@param z Input
     *
     *@returns Sigmoid function of z.
     */
    [[nodiscard]] float sigmoid(const float z) const {
        return static_cast<float>(1 / (1 + pow(e_, z)));
    }

    /**
     *@brief Derivative of the sigmoid function
     *
     *@param z Input
     *
     *@returns Derivative of the sigmoid function.
     */
    [[nodiscard]] float sigmoid_prime(const float z) const {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    /**
     * @brief Return the cost function for an input 'a' and a label 'label'
     *
     * @param a Vector representing the output of the network.
     * @param label Vector representing the label.
     *
     * @returns Square Cost function.
     */
    [[nodiscard]] float cost_function(const Eigen::VectorXf& a, const Eigen::VectorXf& label) const {
        return 0.5f * (a - label).array().square().sum();
    }

    /**
     * @brief Input an image into the network.
     *
     * @param input A Eigen::Vector of floats representing the pixels.
     * @param label The correct clasification.
     *
     * @returns NetworkState struct with activations and z's for each layer, the output of the network, and the label (correct clasification).
     *
     */
    [[nodiscard]] NetworkState feedforward(Eigen::Ref<const Eigen::VectorXf> input, unsigned int label) const;

    /**
    * @brief Use SGD to calculate delta_weights an delta_biases for each bias and each weight in the network.
    *
    * @param state NetworkState struct representing the state of the network after feedforwarding an input.
    *
    * @returns Deltas struct containing a vector of delta_weights and delta_biases to be applied to each weight matrix and bias matrix respectively.
    */
    [[nodiscard]] Deltas backpropagation(const NetworkState& state);


public:

    /**
     * @brief Network Constructor
     *
     * @param neurons Vector of unsigned ints representing the amount of neurons in the i-th layer.
     * @param parameters Hyperparams struct with hyperparameters of the network.
     * @param dataset Dataset object with MNIST data.
     */
    explicit Network(const std::vector<unsigned int>& neurons, const Hyperparams& parameters, const Dataset& dataset);

    /**
     * @brief Train the neural network based using SGD, on the dataset and hyperparameters.
     */
    void train();

    /**
     *@brief Classify an input.
     *
     *@param input Vector of pixel inputs.
     *
     *@returns The model's prediction.
     */
    [[nodiscard]] int classify(Eigen::Ref<const Eigen::VectorXf> input) const;

    /**
     * @brief Tests the model against n randomly selected test images from the dataset.
     *
     * @param n Amount of testing images to be used.
     *
     * @returns The amount of correctly classified images.
     */
    [[nodiscard]] unsigned int test_network(unsigned int n) const ;

};

#endif //DIGITRECOGNITION_NETWORK_H