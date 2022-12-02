#include "neural_network.hpp"
#include "activation_functions.hpp"
#include "misc.hpp"

#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
/**
 * @brief Construct a new Neural Network object
 *
 *
 * @param X The training data - expects rows to be training samples and columns to be data of the sample
 * @param Y The training labels - expects rows to be labels
 * @param learning_Rate learning rate or training rate defines how big a step does the network take after
 * backpropagation (the lower, the smaller)
 * @param iterations number of iterations (will be removed later for stopping condition)
 */

NeuralNetwork::NeuralNetwork(const vec &X, const vec &y, double learning_Rate, int epochs, int batch_size)
{
    this->learning_rate = learning_Rate;
    this->epochs = epochs;
    this->batch_size = batch_size;
    examples = 60000;

    this->X = X;
    multiply_by_scalar((1.0 / 255.0), this->X, examples, 784);
    // transpose(this->X, examples, 784);
    this->y = y;

    one_hot_Y = one_hot_encode(y);
    // transpose(one_hot_Y, examples, 10);

    // initialize parameters for layers
    input_size = 784;
    layer1_size = 150;
    layer2_size = 100;
    output_size = 10;

    init_params(W1, b1, layer1_size, input_size);
    init_params(W2, b2, layer2_size, layer1_size);
    init_params(W3, b3, output_size, layer2_size);
}

void NeuralNetwork::init_params(vec &W, vec &b, int rows, int cols)
{
    // Kaiming He initialization

    // fixed seed
    std::mt19937 mt{0};
    double deviation = 2.0 / double(cols);
    std::normal_distribution<double> dist(0, deviation);

    // generate initial values for weights
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            W.push_back(dist(mt));
        }
    }

    // generate initial values for bias
    for (int i = 0; i < rows; i++)
    {
        b.push_back(0.0);
    }
}

void NeuralNetwork::forward_prop(const vec &batch)
{
    // first layer pass
    Z1 = mul(W1, batch, layer1_size, input_size, input_size, batch_size);
    add_vec_to_columns(Z1, b1, layer1_size, batch_size);
    A1 = ReLU(Z1);

    // second layer pass
    Z2 = mul(W2, A1, layer2_size, layer1_size, layer1_size, batch_size);
    add_vec_to_columns(Z2, b2, layer2_size, batch_size);
    A2 = ReLU(Z2);

    // output
    Z3 = mul(W3, A2, output_size, layer2_size, layer2_size, batch_size);
    add_vec_to_columns(Z3, b3, output_size, batch_size);
    A3 = transpose(softmax(transpose(Z3, output_size, batch_size), batch_size, output_size), batch_size, output_size);
}

void NeuralNetwork::back_prop(const vec &batch, const vec &one_hot_Y)
{
    // categorical cross entropy loss and softmax derivative gives us output error
    vec output_error = A3;
    matrix_subtract(output_error, one_hot_Y, output_size, batch_size);

    // calculate derivatives for output
    d_W3 = mul(output_error, transpose(A2, layer2_size, batch_size), output_size, batch_size, batch_size, layer2_size);
    multiply_by_scalar((1.0 / batch_size), d_W3, output_size, layer2_size);
    d_b3 = element_sum(output_error, output_size, batch_size);
    multiply_by_scalar((1.0 / batch_size), d_b3, output_size, 1);

    // layer 2 error
    vec z2_error = element_mul(
        mul(transpose(W3, output_size, layer2_size), output_error, layer2_size, output_size, output_size, batch_size),
        ReLU_derivative(Z2), layer2_size, batch_size);

    // calculate derivatives for layer 2
    d_W2 = mul(z2_error, transpose(A1, layer1_size, batch_size), layer2_size, batch_size, batch_size, layer1_size);
    multiply_by_scalar((1.0 / batch_size), d_W2, layer2_size, layer1_size);
    d_b2 = element_sum(z2_error, layer2_size, batch_size);
    multiply_by_scalar((1.0 / batch_size), d_b2, layer2_size, 1);

    // layer 1 error
    vec z1_error = element_mul(
        mul(transpose(W2, layer2_size, layer1_size), z2_error, layer1_size, layer2_size, layer2_size, batch_size),
        ReLU_derivative(Z1), layer1_size, batch_size);

    // calculate derivatives for layer 1
    d_W1 = mul(z1_error, transpose(batch, input_size, batch_size), layer1_size, batch_size, batch_size, input_size);
    multiply_by_scalar((1.0 / batch_size), d_W1, layer1_size, input_size);
    d_b1 = element_sum(z1_error, layer1_size, batch_size);
    multiply_by_scalar((1.0 / batch_size), d_b1, layer1_size, 1);
}

void NeuralNetwork::update()
{
    multiply_by_scalar(learning_rate, d_W1, layer1_size, input_size);
    matrix_subtract(W1, d_W1, layer1_size, input_size);

    multiply_by_scalar(learning_rate, d_b1, layer1_size, 1);
    matrix_subtract(b1, d_b1, layer1_size, 1);

    multiply_by_scalar(learning_rate, d_W2, layer2_size, layer1_size);
    matrix_subtract(W2, d_W2, layer2_size, layer1_size);

    multiply_by_scalar(learning_rate, d_b2, layer2_size, 1);
    matrix_subtract(b2, d_b2, layer2_size, 1);

    multiply_by_scalar(learning_rate, d_W3, output_size, layer2_size);
    matrix_subtract(W3, d_W3, output_size, layer2_size);

    multiply_by_scalar(learning_rate, d_b3, output_size, 1);
    matrix_subtract(b3, d_b3, output_size, 1);
}

void NeuralNetwork::train()
{
    auto start = std::chrono::steady_clock::now();

    // training loop using mini-batch sgd
    for (int i = 0; i < epochs; i++)
    {
        for (int j = 0; j < examples / batch_size; j++)
        {
            // load one batch of training vectors
            vec X_batch = vec(X.begin() + j * batch_size * input_size,
                              X.begin() + j * batch_size * input_size + batch_size * input_size);
            X_batch = transpose(X_batch, batch_size, input_size);

            // load one batch of encoded labels
            vec Y_batch =
                vec(one_hot_Y.begin() + j * batch_size * 10, one_hot_Y.begin() + j * batch_size * 10 + batch_size * 10);
            Y_batch = transpose(Y_batch, batch_size, 10);

            // train
            forward_prop(X_batch);
            back_prop(X_batch, Y_batch);
            update();
        }

        // training print
        if (i % 10 == 0)
        {
            int correct = 0;
            for (int j = 0; j < examples / batch_size; j++)
            {
                vec X_batch = vec(X.begin() + j * batch_size * input_size,
                                  X.begin() + j * batch_size * input_size + batch_size * input_size);
                X_batch = transpose(X_batch, batch_size, input_size);

                vec Y_batch = vec(one_hot_Y.begin() + j * batch_size * 10,
                                  one_hot_Y.begin() + j * batch_size * 10 + batch_size * 10);
                transpose(Y_batch, batch_size, 10);

                forward_prop(X_batch);
                vec pred = one_hot_decode(transpose(A3, output_size, batch_size));
                correct += check_accuracy(pred, one_hot_decode(Y_batch));
            }
            double acc = double(correct) / double(examples);

            std::cout << "Epoch no.:" << i << "; ";
            std::cout << "Training accuracy:" << acc << "; ";
            std::cout << "Learning rate:" << learning_rate << "\n";
            measuretime(start, "Elapsed time:");
        }
    }
}

vec NeuralNetwork::predict(const vec &X, const vec &y, int examples, int batch_size, bool test)
{
    vec X_pred = X;
    multiply_by_scalar((1.0 / 255.0), X_pred, examples, 784);

    vec one_hot_Y = one_hot_encode(y);

    vec res;
    int correct = 0;
    for (int j = 0; j < examples / batch_size; j++)
    {
        vec X_batch = vec(X_pred.begin() + j * batch_size * input_size,
                          X_pred.begin() + j * batch_size * input_size + batch_size * input_size);
        X_batch = transpose(X_batch, batch_size, input_size);

        vec Y_batch =
            vec(one_hot_Y.begin() + j * batch_size * 10, one_hot_Y.begin() + j * batch_size * 10 + batch_size * 10);
        transpose(Y_batch, batch_size, 10);

        forward_prop(X_batch);
        vec pred = one_hot_decode(transpose(A3, output_size, batch_size));
        res.insert(res.end(), pred.begin(), pred.end());
        correct += check_accuracy(pred, one_hot_decode(Y_batch));
    }

    double acc = double(correct) / double(examples);

    if (test)
    {
        std::cout << "Test accuracy:" << acc << "\n";
    }
    else
    {
        std::cout << "Train accuracy:" << acc << "\n";
    }

    return res;
}