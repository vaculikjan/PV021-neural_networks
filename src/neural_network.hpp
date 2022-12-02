#pragma once
#include "math.hpp"

#include <algorithm>
#include <random>
#include <vector>

class NeuralNetwork
{

  private:
    // training data
    vec X;
    vec y;

    vec one_hot_Y;
    int examples;

    // network variables
    int input_size;
    int layer1_size;
    int layer2_size;
    int output_size;

    vec W1, W2, W3; // weights
    vec b1, b2, b3; // biases

    vec Z1, Z2, Z3; // multiplication products
    vec A1, A2, A3; // activated multiplication prudcts

    vec d_W1, d_W2, d_W3; // calculated weight deltas
    vec d_b1, d_b2, d_b3; // calculated bias deltas

    // hyper params
    double learning_rate;
    int epochs;
    int batch_size;

    /**
     * @brief Initialize weights and biases.
     *
     * Uses Kaiming He initialization suitable for networks using ReLU.
     *
     * The result is passed by reference in W and b.
     *
     * @param W weight matrix to be initialized
     * @param b bias vector to be initialized
     * @param rows number of rows
     * @param cols number of columns
     */
    void init_params(vec &W, vec &b, int rows, int cols);

    /**
     * @brief Propagete forward through the network.
     *
     * Takes the input and propagates it through the network. In each layer, the input from the previous layer is
     * multiplied by the weights and passed through the activation function.
     *
     * @param batch train data
     */
    void forward_prop(const vec &batch);
    /**
     * @brief Propagate backward through the network.
     *
     * Minimizes the loss function by computing the partial derivatives in each layer. Creates the deltas for adjusting
     * the weights and biases of the network.
     *
     * @param batch train data
     * @param one_hot_Y train labels
     */
    void back_prop(const vec &batch, const vec &one_hot_Y);

    /**
     * @brief Update the weights and biases.
     *
     * Updates the weights and biases of the network based on the output of the last pass of the backward propagation.
     * The calculated deltas are multiplied by the chosen learning rate.
     *
     */
    void update();

  public:
    /**
     * @brief Construct a new Neural Network object.
     *
     * @param X train data
     * @param y train labels
     * @param learning_rate learning rate
     * @param epochs number of training epochs
     * @param batch_size number of training examples in one batch
     */
    NeuralNetwork(const vec &X, const vec &y, double learning_rate, int epochs, int batch_size);

    /**
     * @brief Train the network.
     *
     * Uses the minibatch gradient descent algorithm to train the neural network.
     *
     */
    void train();

    /**
     * @brief Predict labels.
     *
     * Infers the labels based on the internal configuration achieved by previous training.
     *
     * @param X input data
     * @param y actual labels
     * @param examples number of items to be labeled
     * @param batch_size number of items in one batch
     * @param test test or train set
     *
     * @return vec precited labels
     */
    vec predict(const vec &X, const vec &y, int examples, int batch_size, bool test);
};