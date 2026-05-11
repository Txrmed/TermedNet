//
// Created by termed on 8.05.2026.
//

#include "../include/Dataset.h"

#include<bit>
#include<Eigen/Dense>
#include<fstream>
#include<iostream>
#include <random>
#include <filesystem>

#define PIXEL_COUNT 784

using namespace std;


/*
 *#TODO:
 * Create a dataset loader which automatically detects files in /data/mnist
 * and based on the magic numbers and size of the files determines which
 * are the label files and which are the image files.
 */
// void Dataset::validate_file_headers(ifstream& train_images, ifstream& train_labels, ifstream& test_images, ifstream& test_labels) {
//     int training_images_magical;
//     int testing_images_magical;
//
//     train_images.read(reinterpret_cast<char*>(&training_images_magical), )
// }

// Dataset::Dataset(const std::string& dataset_name) : dataset_name_(dataset_name) {
//     filesystem::path path = filesystem::read_symlink("/proc/self/exe").remove_filename().parent_path();
//     path /= "data";
//     path /= dataset_name_;
//     if (!filesystem::exists(path)) {
//         throw filesystem::filesystem_error("Dataset does not exist in /data/ catalog.",
//             path, std::make_error_code(std::errc::no_such_file_or_directory));
//     }
// };
//
// DataFiles Dataset::find_files() {
//     for (auto file : data_path_) {
//
//     }

//}

void Dataset::load_dataset() {

    //find_files();


    // Open training data.
    ifstream train_images("../data/mnist/train_images.idx3-ubyte", std::ios::binary);
    ifstream train_labels("../data/mnist/train_labels.idx1-ubyte", std::ios::binary);

    // Open testing data.
    ifstream test_images("../data/mnist/test_images.idx3-ubyte", std::ios::binary);
    ifstream test_labels("../data/mnist/test_labels.idx1-ubyte", std::ios::binary);

    if (!train_images.is_open() || !train_labels.is_open() || !test_images.is_open() || !test_labels.is_open()) {
        throw std::runtime_error("Could not open image or label file!");
    }

    //validate_file_headers(train_images, train_labels, test_images, test_labels);


    // Skip 16 byte MNIST header.
    train_images.seekg(16);
    test_images.seekg(16);

    // Skip 8 byte headers.
    train_labels.seekg(8);
    test_labels.seekg(8);

    learning_data_ = Eigen::MatrixXf(PIXEL_COUNT + 1, n_training_img);

    unsigned char buffer[PIXEL_COUNT + 1];
    unsigned char label[1];

    for (int col = 0; col < n_training_img; col++) {
        train_images.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT);
        train_labels.read(reinterpret_cast<char*>(label), 1);

        if (col % 5000 == 0) std::cout << col << " training images loaded." << endl;

        // Add pixels into learning_data_ matrix.
        learning_data_(0, col) = label[0];
        for (int row = 0; row < PIXEL_COUNT; row++) {
            learning_data_(row + 1, col) = static_cast<float>(buffer[row]) / 255;
        }
    }

    testing_data_ = Eigen::MatrixXf(PIXEL_COUNT + 1, n_testing_img);

    for (int col = 0; col < n_testing_img; col++) {
        test_images.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT);
        test_labels.read(reinterpret_cast<char*>(label), 1);

        if (col % 5000 == 0) std::cout << col << " testing images loaded." << endl;

        testing_data_(0, col) = label[0];
        for (int row = 0; row < PIXEL_COUNT; row++) {
            testing_data_(row + 1, col) = static_cast<float>(buffer[row]) / 255;
        }

    }
}

void Dataset::load_custom() {
    filesystem::path custom_path = data_path_;
    custom_path /= "custom/";

    for (const auto& entry : filesystem::directory_iterator(custom_path)) {
        ifstream file(entry.path(), std::ios::binary);

        int magic, width, height, max;
        file >> magic;
        file >> width;
        file >> height;
        file >> max;

        file.ignore(1);

        if (max > 255) {

        }
        }
}

void Dataset::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());

    int n = static_cast<int>(learning_data_.cols());

    // 2. Fisher-Yates Shuffle
    for (int i = n - 1; i > 0; --i) {
        // Pick a random index from 0 to i
        std::uniform_int_distribution dist(0, i);
        int j = dist(g);

        // 3. Swap column i with column j
        // Eigen's .swap() is highly optimized for this
        learning_data_.col(i).swap(learning_data_.col(j));
    }

    n = static_cast<int>(testing_data_.cols());
    for (int i = n - 1; i > 0; --i) {
        // Pick a random index from 0 to i
        std::uniform_int_distribution dist(0, i);
        int j = dist(g);

        // 3. Swap column i with column j
        // Eigen's .swap() is highly optimized for this
        testing_data_.col(i).swap(testing_data_.col(j));
    }
}

Eigen::Ref<const Eigen::MatrixXf> Dataset::get_mini_batch(const unsigned int& amount) {
    current_index += amount;
    return learning_data_.block(0, current_index - amount, PIXEL_COUNT + 1, amount);
}

[[nodiscard]] Eigen::Ref<const Eigen::MatrixXf> Dataset::get_testing_data(unsigned int n) const{
    return testing_data_.block(0, 0, PIXEL_COUNT + 1, n);
}
