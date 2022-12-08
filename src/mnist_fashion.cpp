#include "activation_functions.hpp"
#include "math.hpp"
#include "misc.hpp"
#include "neural_network.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

auto start = std::chrono::steady_clock::now();

int main()
{
    // load training data
    vec X_train;
    X_train.reserve(60000 * 784);
    vec y_train;
    y_train.reserve(60000);
    load_csv("data/fashion_mnist_train_vectors.csv", X_train);
    load_csv("data/fashion_mnist_train_labels.csv", y_train);

    // load testing data
    vec X_test;
    X_test.reserve(10000 * 784);
    vec y_test;
    y_test.reserve(10000);
    load_csv("data/fashion_mnist_test_vectors.csv", X_test);
    load_csv("data/fashion_mnist_test_labels.csv", y_test);

    // run network
    NeuralNetwork nn = NeuralNetwork(X_train, y_train, 0.05, 42, 100);
    nn.train();
    vec test_labels = nn.predict(X_test, y_test, 10000, 100, true);
    vec train_labels = nn.predict(X_train, y_train, 60000, 100, false);

    output_csv(train_labels, 10000, "train_predictions.csv");
    output_csv(test_labels, 10000, "test_predictions.csv");

    measuretime(start, "Finished");

    return 0;
}
