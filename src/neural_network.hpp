#pragma once
#include "math.hpp"

#include <algorithm>
#include <random>

class NeuralNetwork
{

  private:
    // training data
    vec2d X;
    vec2d Y;
    std::vector<vec2d> X_batch;
    std::vector<vec2d> Y_batch;
    vec2d one_hot_Y;
    int examples;

    // layer variables
    vec2d W1, W2, W3;
    vec2d b1, b2, b3;

    // forward prop variables
    vec2d Z1, Z2, Z3;
    vec2d A1, A2, A3;

    // backprop variables
    vec2d d_W1, d_W2, d_W3;
    vec2d d_b1, d_b2, d_b3;

    // hyper params
    double alpha;
    int epochs;
    int batch_size;

    // functions
    void init_params(vec2d &W, vec2d &b, int rows, int cols);
    vec2d one_hot_encode(const vec2d &Y, int classes);
    vec2d one_hot_decode(const vec2d &one_hot_Y);
    double check_accuracy(const vec2d &pred, const vec2d &Y);
    void update();
    void forward_prop(const vec2d &batch);
    void back_prop(const vec2d &batch);

  public:
    NeuralNetwork(const vec2d &X, const vec2d &Y, double training_rate, int epochs, int batch_size);
    void train();
};