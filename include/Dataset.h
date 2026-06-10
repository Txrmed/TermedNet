//
// Created by termed on 8.05.2026.
//

#ifndef DIGITRECOGNITION_DATALOADER_H
#define DIGITRECOGNITION_DATALOADER_H

#include "../include/Logger.h"

#include<Eigen/Dense>
#include<fstream>
#include<iostream>
#include <filesystem>

#define PIXEL_COUNT 784

// struct DatasetParameters {
//     unsigned int num_training_img;
//     unsigned int num_testing_img;
// };


/**
 *@class Dataset
 * @brief A class to load and access learning data and testing data by the MLP or CNN.
 *  Loads learning data and testing data from the data/mnist/ directory.
 *  The directory should contains 4 files, one for learning image data, one for the learning labels and the same respectively
 *  for the testing data.
 *
 *  Also loads the custom data from /data/custom. Files should come in PGM P5 or P2 format.
 *
 *  Provides a shuffle() method to shuffle a matrix, usually the learning_data_ matrix.
 *
 */
class Dataset {

public:
    /**
     * @brief Dataset constructor
     * Creates a logger object to log information to standard output. Logs information if verbose is true.
     *
     * @param verbose Verbose mode
     */
    Dataset(bool verbose) : logger_(verbose) {}

    /**
     * @struct DataFiles
     * @brief Contains references to 4 streams : learning_images, learning_labels, testing_images, testing_labels
     *
     */
    struct DataFiles {

        // Training Data
        std::ifstream& train_images;
        std::ifstream& train_labels;

        // Testing data
        std::ifstream& test_images;
        std::ifstream& test_labels;

    };


    /**
     * @brief Load entire MNIST dataset into respective Matrices.
     */
    void load_dataset();


    /**
     * @brief Print an image to standard output.
     */
    static void print_image(Eigen::Ref<const Eigen::VectorXf> image);

    /**
     * @brief Shuffle the given matrix randomly.
     * @param matrix The matrix to be shuffled
     * @param format Format according to which the matrix will be shuffle. Using "MLP" the function will shuffle columns
     * @param format randomly. Using "CNN" the function will shuffle entire blocks 28x28 blocks.
     */
    static void shuffle(Eigen::Ref<Eigen::MatrixXf> matrix, std::string_view format = "MLP");


    /**
     * @brief Get the amount of learning_images.
     *
     * @return Amount of learning_images.
     */
    [[nodiscard]] int amount_training_images() const {
        return this->n_training_img;
    }

    /**
     * @brief Get the amount of testing images from the loaded dataset.
     *
     * @return amount of testing images.
     */
    [[nodiscard]] int amount_testing_images() const {
        return this->n_testing_img;
    }

    // Reshape a matrix to cnn format
    [[nodiscard]] static Eigen::MatrixXf reshape(Eigen::MatrixXf matrix);

    /**
     * @brief Get n images from the training data matrix.
     * Get n images from the training data matrix. The images come in a 785 x n matrix.
     * The first (0th) row in every column is the label of the image.
     * To use this data for training the CNN, ensure it is properly reshaped by calling the `reshape()` function.
     * @see reshape()
     *
     * @param n Amount of images in the training data.
     *
     * @return An Eigen::Ref to a Matrix of flattened pixel values with the label in the 0th row.
     */

    Eigen::MatrixXf get_learning_data(size_t n = 60000) const {
        return learning_data_.block(0, 0,  PIXEL_COUNT + 1, n);
    }

    /**
     * @brief Get n testing images.
     *
     * @param n Amount of testing images.
     *
     * @return An Eigen::Ref to a matrix with testing data.
     */
    [[nodiscard]] Eigen::MatrixXf get_testing_data(size_t n = 10000) const{
        return testing_data_.block(0, 0, PIXEL_COUNT + 1, n);
    }

    /**
     * @brief Get custom user-provided images.
     * Get all available custom images from the dataset. Unlike the training and testing datasets,
     * this matrix does not include image labels in the 0th row.
     * To use this data for a CNN, ensure it is properly reshaped or preprocessed using `reshape()`.
     * @see reshape()
     *
     * @return An Eigen::Ref to a matrix of flattened pixel values without labels.
     */
    [[nodiscard]] Eigen::MatrixXf get_custom_data() const {
        return custom_data_;
    }


private:

    Logger logger_;

    std::filesystem::path data_path_ = "/home/termed/trelemorele/sidehoes/TermedNet/data/";

    // Amount of training images.
    const unsigned int n_training_img = 60000;

    // Amount of validation images
    const unsigned int n_validation_img = 10000;

    // Amount of testing images.
    const unsigned int n_testing_img = 10000;

    // Data from the MNIST dataset
    Eigen::MatrixXf learning_data_;
    Eigen::MatrixXf validation_data_;
    Eigen::MatrixXf testing_data_;

    // Custom data drawn in paint :)
    Eigen::MatrixXf custom_data_;

    unsigned int current_index = 0;

    /**
     * @brief Load an image of the P2 PGM format and return it as a Eigen::VectorXf
     *
     * @param file filestream reference
     * @return An Eigen::VectorXf with the pixels
     */
    Eigen::Ref<const Eigen::VectorXf> load_p2_image(std::ifstream& file);

    /**
     * @brief Load an image of the P5 PGM format and return it as a Eigen::VectorXf
     *
     * @param file filestream reference
     * @return An Eigen::VectorXf with the pixels
     */
    Eigen::VectorXf load_p5_image(std::ifstream& file);


};


#endif //DIGITRECOGNITION_DATALOADER_H