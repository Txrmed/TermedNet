#include "../include/Network.h"
#include "../include/Hyperparams.h"

#include<Eigen/Dense>
#include<iostream>
#include <filesystem>

using namespace std;

int main() {

    const vector<unsigned int> neurons = {784, 30, 10};

    constexpr Hyperparams parameters{10, 2, 60000, 10000, 2};

    Dataset dataset;//("mnist");

    dataset.load_custom();
    // Network network(neurons, parameters, dataset);
    // network.train();

}

// cout << "Label: " << label << endl;
// cout << "Prediction: " << predicted_label << endl;
//
// for (int i = 0; i < n; i++) {
//     cout << tests(0, i) << endl;
//
//     for (int col = 0; col < 28; col++) {
//         for (int row = 0; row < 28; row++) {
//             float val = tests(row + col * 28, i);
//             if (val > 0.25) cout << "# ";
//             else cout << ". ";
//         }
//         cout << endl;
//     }
// }
//
// cout << endl;