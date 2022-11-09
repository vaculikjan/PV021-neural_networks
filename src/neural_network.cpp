#include "neural_network.hpp"
#include "activation_functions.hpp"
#include "misc.hpp"

#include <chrono>
#include <iostream>
#include <random>

/**
 * @brief Construct a new Neural Network object
 *
 *
 * @param X The training data - expects rows to be training samples and columns to be data of the sample
 * @param Y The training labels - expects rows to be labels
 * @param training_rate alpha or training rate defines how big a step does the network take after backpropagation (the
 * lower, the slower)
 * @param iterations number of iterations (will be removed later for stopping condition)
 */

NeuralNetwork::NeuralNetwork(const vec2d &X, const vec2d &Y, double training_rate, int epochs, int batch_size)
{
    alpha = training_rate;
    this->epochs = epochs;
    this->batch_size = batch_size;
    examples = Y.size();

    this->X = (1.0 / 255.0) * X;
    this->Y = Y;

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

    // create two random engines with the same state
    std::mt19937 eng1(seed);
    auto eng2 = eng1;

    // std::shuffle(std::begin(this->X), std::end(this->X), eng1);
    // std::shuffle(std::begin(this->Y), std::end(this->Y), eng2);

    for (int i = 0; i < X.size() - batch_size; i += batch_size)
    {
        vec2d vec_x(this->X.begin() + i, this->X.begin() + i + batch_size);
        X_batch.push_back(transpose(vec_x));
        vec2d vec_y(this->Y.begin() + i, this->Y.begin() + i + batch_size);
        Y_batch.push_back(vec_y);
    }

    one_hot_Y = transpose(one_hot_encode(Y, 10));

    // initialize parameters for layers

    init_params(W1, b1, 150, 784);
    init_params(W2, b2, 100, 150);
    init_params(W3, b3, 10, 100);
}

void NeuralNetwork::init_params(vec2d &W, vec2d &b, int rows, int cols)
{
    double deviation = 2.0 / double(cols);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(0, deviation);

    for (int i = 0; i < rows; i++)
    {
        std::vector<double> vec;
        for (int j = 0; j < cols; j++)
        {
            vec.push_back(dist(mt));
        }
        W.push_back(vec);
    }

    for (int i = 0; i < rows; i++)
    {
        b.push_back(std::vector<double>{0.0});
    }

    return;
}

vec2d NeuralNetwork::one_hot_encode(const vec2d &Y, int classes)
{
    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d one_hot_Y;

    for (row = Y.begin(); row != Y.end(); row++)
    {
        for (col = row->begin(); col != row->end(); col++)
        {
            std::vector<double> vec(classes, 0);
            vec[*col] = 1;
            one_hot_Y.push_back(vec);
        }
    }

    return one_hot_Y;
}

vec2d NeuralNetwork::one_hot_decode(const vec2d &one_hot_Y)
{
    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d Y;

    for (row = one_hot_Y.begin(); row != one_hot_Y.end(); row++)
    {
        int i = 0;
        double max = 0;
        for (col = row->begin(); col != row->end(); col++)
        {
            if (*col > max)
            {
                max = *col;
                i = col - row->begin();
            }
        }
        Y.push_back(std::vector<double>{double(i)});
    }

    return Y;
}

double NeuralNetwork::check_accuracy(const vec2d &pred, const vec2d &Y)
{

    int number_of_examples = Y.size();
    int correct_predictions = 0;

    int pred_size = pred.size();

    for (int i = 0; i < Y.size(); i++)
    {
        if (Y[i][0] == pred[i][0])
        {
            correct_predictions++;
        }
    }

    return (double(correct_predictions) / double(number_of_examples));
}

void NeuralNetwork::forward_prop(const vec2d &batch)
{
    Z1 = mul(W1, batch) + b1;
    A1 = ReLU(Z1);

    Z2 = mul(W2, A1) + b2;
    A2 = ReLU(Z2);

    Z3 = mul(W3, A2) + b3;
    A3 = transpose(softmax(transpose(Z3)));
}

void NeuralNetwork::back_prop(const vec2d &batch)
{
    // categorical cross entropy loss and softmax derivative
    vec2d output_error = A3 - one_hot_Y;

    d_W3 = (1.0 / examples) * (mul(output_error, transpose(A2)));

    // should be one number
    d_b3 = (1.0 / examples) * element_sum(output_error);

    vec2d z2_error = element_mul(mul(transpose(W3), output_error), ReLU_derivative(Z2));

    d_W2 = (1.0 / examples) * (mul(z2_error, transpose(A1)));
    // should be one number
    d_b2 = (1.0 / examples) * element_sum(z2_error);

    vec2d z1_error = element_mul(mul(transpose(W2), z2_error), ReLU_derivative(Z1));

    d_W1 = (1.0 / examples) * (mul(z1_error, transpose(batch)));
    d_b1 = (1.0 / examples) * element_sum(z1_error);
}

void NeuralNetwork::update()
{
    W1 = W1 - (alpha * d_W1);
    b1 = b1 - (alpha * d_b1);

    W2 = W2 - (alpha * d_W2);
    b2 = b2 - (alpha * d_b2);

    W3 = W3 - (alpha * d_W3);
    b3 = b3 - (alpha * d_b3);
}

void NeuralNetwork::train()
{
    double x_size = X.size();
    vec2d x_trans = transpose(X);
    auto start = std::chrono::steady_clock::now();
    float last_acc = -1;
    for (int i = 0; i < epochs; i++)
    {
        /*
        for (int j = 0; j < X_batch.size(); j++)
        {
            forward_prop(X_batch[j]);
            back_prop(X_batch[j]);
            update();

            if (i % 10 == 0 && j == X_batch.size() - 1)
            {
                forward_prop(x_trans);
                vec2d pred = one_hot_decode(transpose(A3));
                double acc = check_accuracy(pred, Y);
                if (acc - last_acc < 0)
                {
                    alpha = (std::max(alpha * 0.75, 0.001));
                }
                last_acc = acc;

                std::cout << "Iteration:" << i << "; ";
                std::cout << "Accuracy:" << acc << "; ";
                std::cout << "Training rate:" << alpha << "\n";
            }
        }
        */

        forward_prop(x_trans);
        back_prop(x_trans);
        update();

        if (i % 5 == 0)
        {
            vec2d pred = one_hot_decode(transpose(A3));
            double acc = check_accuracy(pred, Y);
            std::cout << "Iteration:" << i << "\n";
            std::cout << "Accuracy:" << acc << "\n";
            if (acc - last_acc < 0)
            {
                alpha = (std::max(alpha * 0.75, 0.05));
            }
            last_acc = acc;
        }
    }
}