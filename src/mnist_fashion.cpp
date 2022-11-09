#include "activation_functions.hpp"
#include "math.hpp"
#include "misc.hpp"
#include "neural_network.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

auto start = std::chrono::steady_clock::now();

void init_params(vec2d &W, vec2d &b, int rows, int cols);
void forward_prop(const vec2d &W1, const vec2d &b1, const vec2d &W2, const vec2d &b2, const vec2d X, vec2d &Z1,
                  vec2d &A1, vec2d &Z2, vec2d &A2);
void back_prop(const vec2d &Z1, const vec2d &A1, const vec2d &A2, const vec2d &W2, const vec2d &X, const vec2d &Y,
               const vec2d &one_hot_Y, vec2d &d_W1, vec2d &d_b1, vec2d &d_W2, vec2d &d_b2);
void update(double alpha, const vec2d &d_W1, const vec2d &d_b1, const vec2d &d_W2, const vec2d &d_b2, vec2d &W1,
            vec2d &b1, vec2d &W2, vec2d &b2);
void train(const vec2d &X, const vec2d &Y, double training_rate, int iterations, int neurons);

int main()
{
    // Honza paths
///*
    vec2d Y_train = load_csv("../data/fashion_mnist_train_labels.csv");
    vec2d train_data = load_csv("../data/fashion_mnist_train_vectors.csv");
    
//*/
    // Martin paths
/*
    vec2d train_data = load_csv("C:\\Users\\Martin\\Desktop\\Neurals_new\\data\\fashion_mnist_train_vectors.csv");
    vec2d Y_train = load_csv("C:\\Users\\Martin\\Desktop\\Neurals_new\\data\\fashion_mnist_train_labels.csv");
*/

    // Standard transpose
///*
    vec2d X_train = (1.0 / 255.0) * transpose(train_data);
//*/
    
    // Efficient transpose
/*
    measuretime(start, "PRED");
    TwoDPivotWrapper<vec2d> wrapper(train_data);
    measuretime(start, "PO");
    vec2d X_train = (1.0 / 255.0) * wrapper;
    std::cout << X_train.size() << std::endl;
*/

    
    

    train(X_train, Y_train, 0.1, 500, 50);
    NeuralNetwork nn = NeuralNetwork(X_train, Y_train, 0.2, 100, 100);
    nn.train();

    return 0;
}

void init_params(vec2d &W, vec2d &b, int rows, int cols)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

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
        b.push_back(std::vector<double>{dist(mt)});
    }

    return;
}

vec2d one_hot_encode(const vec2d &Y, int classes)
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

// send in A2 transposed
vec2d one_hot_decode(const vec2d &one_hot_Y)
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

double check_accuracy(const vec2d &pred, const vec2d &Y)
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

void forward_prop(const vec2d &W1, const vec2d &b1, const vec2d &W2, const vec2d &b2, const vec2d X, vec2d &Z1,
                  vec2d &A1, vec2d &Z2, vec2d &A2)
{
    Z1 = mul(W1, X) + b1;
    A1 = ReLU(Z1);

    Z2 = mul(W2, A1) + b2;
    A2 = transpose(softmax(transpose(Z2)));
}

void back_prop(const vec2d &Z1, const vec2d &A1, const vec2d &A2, const vec2d &W2, const vec2d &X, const vec2d &Y,
               const vec2d &one_hot_Y, vec2d &d_W1, vec2d &d_b1, vec2d &d_W2, vec2d &d_b2)
{
    int number_of_examples = Y.size();
    // categorical cross entropy loss and softmax derivative
    vec2d output_error = A2 - one_hot_Y;

    d_W2 = (1.0 / number_of_examples) * (mul(output_error, transpose(A1)));

    // should be one number
    d_b2 = (1.0 / number_of_examples) * element_sum(output_error);

    vec2d z1_error = element_mul(mul(transpose(W2), output_error), ReLU_derivative(Z1));

    d_W1 = (1.0 / number_of_examples) * (mul(z1_error, transpose(X)));
    d_b1 = (1.0 / number_of_examples) * element_sum(z1_error);
}

void update(double alpha, const vec2d &d_W1, const vec2d &d_b1, const vec2d &d_W2, const vec2d &d_b2, vec2d &W1,
            vec2d &b1, vec2d &W2, vec2d &b2)
{
    W1 = W1 - (alpha * d_W1);
    b1 = b1 - (alpha * d_b1);

    W2 = W2 - (alpha * d_W2);
    b2 = b2 - (alpha * d_b2);
}

void train(const vec2d &X, const vec2d &Y, double training_rate, int iterations, int neurons)
{

    // network variables
    vec2d W1;
    vec2d b1;
    vec2d W2;
    vec2d b2;

    init_params(W1, b1, neurons, 784);
    init_params(W2, b2, 10, neurons);

    // forward prop variables
    vec2d Z1;
    vec2d A1;
    vec2d Z2;
    vec2d A2;

    // backprop variables
    vec2d d_W1;
    vec2d d_b1;
    vec2d d_W2;
    vec2d d_b2;

    vec2d one_hot_Y = transpose(one_hot_encode(Y, 10));
    for (int i = 0; i < iterations; i++)
    {
        for (int j = 0; j < 60000 / 100; j++)
        {

            forward_prop(W1, b1, W2, b2, X, Z1, A1, Z2, A2);
            back_prop(Z1, A1, A2, W2, X, Y, one_hot_Y, d_W1, d_b1, d_W2, d_b2);
            update(training_rate, d_W1, d_b1, d_W2, d_b2, W1, b1, W2, b2);

            if (i % 10 == 0)
            {
                vec2d pred = one_hot_decode(transpose(A2));
                double acc = check_accuracy(pred, Y);

                std::cout << "Iteration:" << i << "\n";
                std::cout << "Accuracy:" << acc << "\n";
                measuretime(start, "Time");
            }
        }
    }
}