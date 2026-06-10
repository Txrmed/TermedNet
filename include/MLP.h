//
// Created by termed on 2.05.2026.
//

#ifndef DIGITRECOGNITION_NETWORK_H
#define DIGITRECOGNITION_NETWORK_H

#include <random>

#include "../include/Dataset.h"
#include "../include/Logger.h"

#include<vector>
#include<Eigen/Dense>

/**
 * @class MLP
 * @brief A Multi-Layer Perceptron neural network with backpropagation.
 * This class creates a Multi-Layer perceptron neural network.
 *
 *
 */
class MLP {

public:

    /**
     * @enum LossFunction
     * @brief Available loss functions to be used for the MLP.
     */
    enum class LossFunction {
        QUADRATIC, ///< Quadratic function given by the equation \f$ L = (a(x) - y)^2 \f$
        BINARY_CROSS_ENTROPY, ///< Binary Cross-Entropy function given by the equation \f$ L = -(yln(a) + (1 - y)ln(1 - y)) \f$
    };

    /**
     * @enum ActivationFunction
     * @brief Available activation functions to be used for the MLP.
     */
    enum class ActivationFunction {
        SIGMOID, ///< Sigmoid function given by the equation \f$ \sigma = \frac{1}{1 + e^{-x}}\f$
        RELU, ///< ReLU function.
        TANH, ///< Tanh function.
        SOFTMAX ///< Softmax
    };

    /**
     * @enum Regularization
     * Different regularization methods to prevent overfitting
     * None signals for no regularization to be applied.
     */
    enum class Regularization {
        NONE, ///< No regularization
        L1, ///< L1 regularization #TODO
        L2, ///< L2 regularization
    };

    /**
     * @enum Initializer
     * Initialization method for the weights in each layer.
     *
     */
    enum class Initializer {
        RANDOM, ///< Randomly initialize weights to values between -1 and 1. Not recommended for ReLu.
        HE, ///< He initialization bounds random values between -sqrt(6 / n) and sqrt(6 / n) where n is the amount of input neurons.
        GLOROT ///< Glorot initialization. #TODO
    };

    enum class Optimizer {
        SGD, ///< Stochastic gradient descent, uses a default mini batch of 10
        ADAM ///< Adaptive Moment Estimation (G.O.A.T.)
    };

    /**
     * @struct Hyperparameters
     * @brief Struct containing hyperparameters for training the network.
     * Contains the mini batch size, learning rate and the amount of epochs.
     *
     */
    struct Hyperparameters {
        std::size_t mini_batch_size = 10; ///< Mini batch size (default is 10).

        float learning_rate; ///< Learning rate used when updating weights and biases.

        int epochs; ///< Amount of epochs.

    };

    struct NetworkParameters {
        // Optimizer to use
        Optimizer optimizer = Optimizer::SGD;

        // Which regularization technique apart from dropout to use.
        Regularization regularization = Regularization::NONE;

        // Parameter for regularization.
        float regularization_parameter = 0;

        // Weight initializer
        Initializer initializer = Initializer::RANDOM;

        // Use dropout to minimize overfitting.
        bool dropout_regularization = true;

        // Lambda for dropout regularization. Lambda is equal to the probability of a neuron dying.
        float lambda = 0.5;
    };

    /** @class InputLayer
     *  @brief Create an input layer to the MLP network.
     *
     * Class to represent the input layer to the network. Contains only the amount of neurons.
     *
     */
    class InputLayer {
        friend MLP;

        size_t neuron_amount_;
    public:
        InputLayer(size_t neurons) : neuron_amount_(neurons) {}

    };

    /** @class DenseLayer
     *  @brief Class representing a hidden layer in the MLP network.
     *
     * Represents a hidden layer in an MLP network. Contains the amount of neurons and the activation function to be used
     * in this layer.
     *
     */
    class DenseLayer {
        friend MLP;

        size_t neuron_amount_; ///< Amount of neurons in the layer.
        ActivationFunction activation_function_; ///< Activation function to be used.

    public:
        DenseLayer(size_t neurons, ActivationFunction activation) : neuron_amount_(neurons), activation_function_(activation) {}
    };

    /** @class OutputLayer
     * @brief OutputLayer class
     * Contains the amount of neurons, activation function and loss function to be used.
     *
     * When using the Cross Entropy loss function, you can't use activation functions which generate a random distribution
     * like softmax or sigmoid.
     *
     */
    class OutputLayer {
        friend MLP;

        size_t neuron_amount_; ///< Amount of neurons in the layer.
        ActivationFunction activation_function_; ///< Activation function to be used.
        LossFunction loss_function_; ///< Loss function to be used.

    public:
        OutputLayer(size_t neurons, ActivationFunction activation, LossFunction loss) : neuron_amount_(neurons), activation_function_(activation), loss_function_(loss) {
            if (loss_function_ == LossFunction::BINARY_CROSS_ENTROPY) {
                if (activation_function_ != ActivationFunction::SIGMOID && activation_function_ != ActivationFunction::SOFTMAX) {
                    throw std::invalid_argument("Cant use other activation function than softmax or sigmoid for cross entropy cost.");
                }
            }
        }

    };


    /**
     * @brief Network Constructor
     *
     * Every MLP network has to have an input layer and output layer, however hidden layers are optional.
     * If you want to create a network with no dense layers:
     * MLP(input_layer, {}, output_layer)
     *
     * Regularization defaults to NONE and initializer defaults to RANDOM.
     *
     */
    explicit MLP(InputLayer input_layer, std::initializer_list<DenseLayer> dense_layers, OutputLayer output_layer,
                 NetworkParameters p, bool verbose = true);

    /**
     * @brief Train the neural network using the learning data given in the constructor.
     *  Use the hyperparameters provided in the constructor to train the network.
     *  Loops the amount of set epochs, with each epoch containing \f$ \frac{learning_data.size}{mini_batch.size} \f$ minbatches,
     *  and runs the backpropagation algorithm on each image in the mini_batch.
     */
    void train(Optimizer optimizer, const Eigen::MatrixXf &learning_matrix, const Eigen::MatrixXf& validation_matrix, const Eigen::MatrixXf &testing_matrix, const
               Hyperparameters &hyperparameters);

    /**
     *@brief Classify an input.
     *
     *@param input Vector of pixel inputs.
     *
     *@returns The model's prediction.
     */
    [[nodiscard]] int classify(Eigen::Ref<const Eigen::VectorXf> input) const;


    /**
     * @brief Tests the model against the first $n$ test images from the dataset.
     * This method does not run backpropagation on the test data, only feedforwards the testing data.
     *
     * @param n Amount of testing images to be used.
     *
     * @returns The amount of correctly classified images.
     */
    [[nodiscard]] size_t test_network(Eigen::MatrixXf testing_data) const ;

private:


    struct NetworkState {

        std::vector<Eigen::VectorXf> zs;
        std::vector<Eigen::VectorXf> activations;

        Eigen::VectorXf output;
        Eigen::VectorXf label;
    };

    struct Deltas {
        // Changes for biases at i-th layer;
        std::vector<Eigen::VectorXf> d_biases;

        // Changes for weights.
        std::vector<Eigen::MatrixXf> d_weights;
    };

    Logger logger;

    // Amount of layers in the network.
    std::size_t layers_;

    // Amount of neurons at each layer.
    std::vector<std::size_t> neurons_;

    // Layers of the network.
    OutputLayer output_layer_;
    std::vector<DenseLayer> dense_layers_;
    InputLayer input_layer_;

    // Vector containing the weights for each layer. Matrix at position i corresponds to the `i + 2` layer.
    std::vector<Eigen::MatrixXf> weights_;

    // Vector containing the biases for each layer. Bias at position i corresponds to the `i + 2` layer.
    std::vector<Eigen::VectorXf> biases_;

    // Vector with dropout mask containing 1's and 0's to apply for the dropout regularization method.
    std::vector<Eigen::MatrixXf> dropout_mask_;

    // Parameters for the structure of the network.
    NetworkParameters network_parameters_;

    // Vector containing the activation function pointers for each layer. Activation function at position i corresponds to the `i + 2` layer.
    std::vector<float(*)(float)> activation_;

    // Vector containing pointers to the derivative of the activation function for the i + 1 layer.
    std::vector<float (*) (float)> activation_prime_;

    float (*loss_function) (const Eigen::VectorXf&, const Eigen::VectorXf&);


    /**
     *@brief Calculate sigmoid function.
     *
     *@param z Input
     *
     *@returns Sigmoid function of z.
     */
    [[nodiscard]] static float sigmoid(const float z){
        return static_cast<float>(1 / (1 + pow(2.71828, -z)));
    }

    /**
     *@brief Sigmoid function.
     *
     *@param z Input
     *
     *@returns Derivative of the sigmoid function.
     */
    [[nodiscard]] static float sigmoid_prime(const float z){
        return sigmoid(z) * (1 - sigmoid(z));
    }


    /**
     *@brief ReLU function.
     *
     *@param z Input.
     *
     *@returns ReLU function of z.
     */
    [[nodiscard]] static float relu(const float z) {
        return (z > 0.0) ? z : 0.01f * z;
    }

    /**
     *@brief Derivative of the ReLU function
     *
     *@param z Input
     *
     *@returns Derivative of the ReLU function.
     */
    [[nodiscard]] static float relu_prime(const float z) {
        return (z > 0.0f) ? 1.0f : 0.01f;
    }

    [[nodiscard]] static float tanh(const float z) {
        return std::tanh(z);
    }

    [[nodiscard]] static float tanh_prime(const float z) {
        return 1 - tanh(z) * tanh(z);
    }

    [[nodiscard]] static inline float quadratic_cost_numerical(const Eigen::VectorXf& a, const Eigen::VectorXf& label);

    [[nodiscard]] static inline float binary_cross_entropy_cost_numerical(const Eigen::VectorXf& a, const Eigen::VectorXf& label);

public:
    [[nodiscard]] static Eigen::VectorXf generate_dropout_mask(int rows, float p, int layer_n);
private:
    /**
     * @brief Input an image into the network.
     *
     * @param input An Eigen::Vector of floats representing the pixels.
     * @param label The correct clasification.
     * @param learning Essentialy if the feedforward function should apply the dropout regularization technique.
     *
     * @returns NetworkState struct with activations and z's for each layer, the output of the network, and the label (correct clasification).
     *
     */
    [[nodiscard]] NetworkState feedforward(Eigen::Ref<const Eigen::VectorXf> input, std::size_t label, bool learning) const;

    /**
    * @brief Use SGD to calculate delta_weights an delta_biases for each bias and each weight in the network.
    *
    * @param state NetworkState struct representing the state of the network after feedforwarding an input.
    *
    * @returns Deltas struct containing a vector of delta_weights and delta_biases to be applied to each weight matrix and bias matrix respectively.
    */
    [[nodiscard]] Deltas backpropagation(const NetworkState& state);


};

#endif //DIGITRECOGNITION_NETWORK_H