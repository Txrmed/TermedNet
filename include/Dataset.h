//
// Created by termed on 8.05.2026.
//

#ifndef DIGITRECOGNITION_DATALOADER_H
#define DIGITRECOGNITION_DATALOADER_H

#include<Eigen/Dense>
#include<fstream>
#include<iostream>
#include <filesystem>

#define PIXEL_COUNT 784

// struct DatasetParameters {
//     unsigned int num_training_img;
//     unsigned int num_testing_img;
// };

struct DataFiles {

    // Training Data
    std::ifstream& train_images;
    std::ifstream& train_labels;

    // Testing data
    std::ifstream& test_images;
    std::ifstream& test_labels;

};

// Handle loading the MNIST Dataset into a cpp readable format.
class Dataset {

    std::string dataset_name_;

    std::filesystem::path data_path_ = "/home/termed/trelemorele/sidehoes/neuralnetworks/data/";

    // Amount of neurons in the input layer.
    const unsigned int input_layer_n = 0;

    // Amount of training images.
    const unsigned int n_training_img = 60000;

    // Amount of testing images.
    const unsigned int n_testing_img = 10000;

    // Data from the MNIST dataset
    Eigen::MatrixXf learning_data_;
    Eigen::MatrixXf testing_data_;

    // Custom data drawn in paint :)
    Eigen::MatrixXf custom_data_;

    unsigned int current_index = 0;

    //void validate_file_headers(std::ifstream& train_images, std::ifstream& train_labels, std::ifstream& test_images, ifstream& test_labels);

    // Some function to find files in /data/mnist/ dir
    // DataFiles find_files();

public:

    // Dataset(const std::string& dataset_name);

    /**
     * @brief Load entire MNIST dataset into respective Matrix
     */
    void load_dataset();

    /**
     * @brief Load every image in the /data/custom/ catalog.
     */
    void load_custom();

    /**
     * @brief Shuffle the learning_data_ matrix randomly.
     */
    void shuffle();

    /**
     * @brief Get a mini batch of learning images with labels.
     *
     * @param amount Amount of images in the minibatch.
     *
     * @returns A Eigen::Ref to a Matrix of images.
     */
    Eigen::Ref<const Eigen::MatrixXf> get_mini_batch(const unsigned int& amount);

    /**
     * @brief Get n testing images.
     *
     * @param n Amount of testing images.
     *
     * @returns An Eigen::Ref to a matrix with testing data.
     */
    [[nodiscard]] Eigen::Ref<const Eigen::MatrixXf> get_testing_data(unsigned int n) const;

    /**
     * @brief Return the index of the column of learning_data. Call this function after exhausting all training images.
     */
    void set_return(){ current_index = 0; }

};


#endif //DIGITRECOGNITION_DATALOADER_H