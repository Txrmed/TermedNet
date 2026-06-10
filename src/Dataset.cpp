//
// Created by termed on 8.05.2026.
//

#include "../include/Dataset.h"

#include<Eigen/Dense>
#include<fstream>
#include<iostream>
#include <random>
#include <filesystem>

#define PIXEL_COUNT 784

using namespace std;

void Dataset::load_dataset() {

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

    for (size_t col = 0; col < n_training_img; col++) {
        train_images.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT);
        train_labels.read(reinterpret_cast<char*>(label), 1);

        if (col % 5000 == 0 && col != 0) logger_.log(Logger::Prefix::DATASET) << col << " training images loaded." << endl;

        // Add pixels into learning_data_ matrix.
        learning_data_(0, col) = label[0];
        for (int row = 0; row < PIXEL_COUNT; row++) {
            learning_data_(row + 1, col) = static_cast<float>(buffer[row]) / 255;
        }
    }

    logger_.log(Logger::Prefix::DATASET) << "60000 training images loaded." << endl;

    testing_data_ = Eigen::MatrixXf(PIXEL_COUNT + 1, n_testing_img);

    for (size_t col = 0; col < n_testing_img; col++) {
        test_images.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT);
        test_labels.read(reinterpret_cast<char*>(label), 1);

        if (col % 5000 == 0 && col != 0) logger_.log(Logger::Prefix::DATASET) << col << " testing images loaded." << endl;

        testing_data_(0, col) = label[0];
        for (int row = 0; row < PIXEL_COUNT; row++) {
            testing_data_(row + 1, col) = static_cast<float>(buffer[row]) / 255;
        }

    }

    ////////////////////////////////////////////////////////////////
    // Load custom images from the data/custom/ catalog

    filesystem::path custom_path = data_path_;
    custom_path /= "custom/";

    // Get amount of custom input files in the /data/custom/ catalog.
    size_t count = 0;
    for (const auto& entry : filesystem::directory_iterator(custom_path)) {
        if (entry.is_regular_file()) {
            count++;
        }
    }

    Eigen::MatrixXf data(PIXEL_COUNT, count);
    int col = 0;
    for (const auto& entry : filesystem::directory_iterator(custom_path)) {
        ifstream file(entry.path(), std::ios::binary);

        char header[2];
        file.read(header, 2);

        logger_.log(Logger::Prefix::DATASET) << "Loading image: " << entry << endl;

        Eigen::VectorXf image;
        if (header[0] == 'P' && header[1] == '5') {
            image = load_p5_image(file);
            data.col(col++) = image;
        }
        else if (header[0] == 'P' && header[1] == '2') {
            // some function to load P2 ascii image
        }
        else {
            throw std::invalid_argument("Invalid file type. File type has to be P5 PGM or P2 PGM.");
        }
    }
    custom_data_ = std::move(data);

}

Eigen::VectorXf Dataset::load_p5_image(ifstream& file) {
    file.seekg(0, ios::end);
    streamsize size = file.tellg();

    // Process the file if it uses 2 bytes per pixel.
    if (size > PIXEL_COUNT * 2) {
        int header_jump = size - PIXEL_COUNT * 2;
        file.seekg(ios::beg);
        file.seekg(header_jump);

        unsigned char buffer[PIXEL_COUNT * 2];
        file.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT * 2);

        Eigen::VectorXf pixels(PIXEL_COUNT, 1);

        int row = 0;
        for (int i = 0; i < PIXEL_COUNT * 2; i += 2) {
            pixels(row++, 0) = 1 - ((buffer[i + 1] << 8 | buffer[i]) / 65535.0);
        }

        return pixels;
    }
    // Process the file if it uses 1 byte per pixel.
    if (size > PIXEL_COUNT) {
        int header_jump = size - PIXEL_COUNT;
        file.seekg(ios::beg);
        file.seekg(header_jump);

        unsigned char buffer[PIXEL_COUNT];
        file.read(reinterpret_cast<char*>(buffer), PIXEL_COUNT );

        Eigen::VectorXf pixels(PIXEL_COUNT, 1);

        int row = 0;
        for (unsigned char byte : buffer) {
            pixels(row++, 0) = 1 - (byte / 255.0);
        }
        return pixels;
    }
    throw std::invalid_argument("Invalid file type. File type has to be P5 PGM or P2 PGM.");
}


void Dataset::print_image(Eigen::Ref<const Eigen::VectorXf> image) {
    for (int i = 0; i < 58; i++) {
        cout << "=";
    }
    cout << endl;
    for (int cols = 0; cols < 28; cols++) {
        cout << "|" << flush;
        for (int rows = 0; rows < 28; rows++) {
            float val = image(rows + cols * 28, 0);
            if (val > 0.1) {
                cout << "# ";
            }
            else cout << "  ";
        }
        cout << "|" << endl;
    }
    for (int i = 0; i < 58; i++) {
        cout << "=";
    }
    cout << endl;

}

Eigen::MatrixXf Dataset::reshape(Eigen::MatrixXf matrix) {
    int num_images = matrix.cols();
    int total_cols = 28 * num_images;

    // 1. Initialize the final output matrix with zeros (30 rows)
    Eigen::MatrixXf reshaped_matrix = Eigen::MatrixXf::Zero(29, total_cols);

    for (int i = 0; i < num_images; ++i) {
        int target_col_start = i * 28;

        // 2. Put the label in the top-left corner (Row 0, Col start of this block)
        float label = matrix(0, i);
        reshaped_matrix(0, target_col_start) = label;



        // 3. Extract the 784 image pixels for this column
        // We grab rows 1 to 784 from the current column 'i'
        Eigen::VectorXf flattened_image = matrix.col(i).bottomRows(784);

        // 4. Map the flat vector into a 28x28 Row-Major image layout
        Eigen::Map<Eigen::Matrix<float, 28, 28, Eigen::RowMajor>> image_view(flattened_image.data());



        // 5. Paste the 28x28 image into the target block starting at Row 2
        // (Row 0 is the label, Row 1 stays 0 as padding, Rows 2-29 hold the image)
        reshaped_matrix.block<28, 28>(1, target_col_start) = image_view;
    }

    return reshaped_matrix;
}

void Dataset::shuffle(Eigen::Ref<Eigen::MatrixXf> matrix, std::string_view format) {
    if (format == "MLP") {
        std::random_device rd;
        std::mt19937 g(rd());

        int n = static_cast<int>(matrix.cols());

        for (int i = n - 1; i > 0; --i) {
            // Pick a random index from 0 to i
            std::uniform_int_distribution dist(0, i);
            int j = dist(g);

            matrix.col(i).swap(matrix.col(j));
        }
    }
    else if (format == "CNN") {

    }
}



