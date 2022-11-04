#include "csv_loader.hpp"
#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <random>

void init_params(vec2d &W, vec2d &b, int rows, int cols);
void forward_prop(const vec2d &W1, const vec2d &b1, const vec2d &W2, const vec2d &b2, const vec2d X, vec2d &Z1,
                  vec2d &A1, vec2d &Z2, vec2d &A2);
vec2d ReLU(const vec2d &Z);
vec2d softmax(const vec2d &Z);
vec2d one_hot_encode(const vec2d &Y, int classes);
void back_prop(const vec2d &Z1, const vec2d &A1, const vec2d &Z2, const vec2d &A2, const vec2d &W2, const vec2d X,
               const vec2d Y, vec2d &d_W1, vec2d &d_b1, vec2d &d_W2, vec2d &d_b2);

int main()
{

    vec2d v1{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
    };

    vec2d v2{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    };

    mul(v1, v2);
    transpose(v1);
    vec2d X_train = (1.0 / 255.0) * transpose(load_csv("../data/fashion_mnist_train_vectors.csv"));
    vec2d Y_train = load_csv("../data/fashion_mnist_train_labels.csv");

    vec2d W1;
    vec2d b1;
    vec2d W2;
    vec2d b2;
    init_params(W1, b1, 10, 784);
    init_params(W2, b2, 10, 10);

    vec2d Z1;
    vec2d A1;
    vec2d Z2;
    vec2d A2;

    forward_prop(W1, b1, W2, b2, X_train, Z1, A1, Z2, A2);

    Y_train = transpose(one_hot_encode(Y_train, 10));
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

void forward_prop(const vec2d &W1, const vec2d &b1, const vec2d &W2, const vec2d &b2, const vec2d X, vec2d &Z1,
                  vec2d &A1, vec2d &Z2, vec2d &A2)
{
    Z1 = mul(W1, X) + b1;
    A1 = ReLU(Z1);

    Z2 = mul(W2, A1) + b2;
    A2 = softmax(Z2);
}

vec2d ReLU(const vec2d &Z)
{
    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d A;

    for (row = Z.begin(); row != Z.end(); row++)
    {
        std::vector<double> vec;

        for (col = row->begin(); col != row->end(); col++)
        {
            vec.push_back(std::max(*col, 0.0));
        }

        A.push_back(vec);
    }

    return A;
}

vec2d ReLU_derivative(const vec2d &Z)
{
    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d A;

    for (row = Z.begin(); row != Z.end(); row++)
    {
        std::vector<double> vec;

        for (col = row->begin(); col != row->end(); col++)
        {
            if (*col > 0)
            {
                vec.push_back(1);
            }
            else
            {
                vec.push_back(0);
            }
        }

        A.push_back(vec);
    }

    return A;
}

vec2d softmax(const vec2d &Z)
{
    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d A;

    double sum = 0;
    int count = 0;

    for (row = Z.begin(); row != Z.end(); row++)
    {
        std::vector<double> vec;

        for (col = row->begin(); col != row->end(); col++)
        {
            double res = std::exp(*col);
            count++;
            sum += res;
            vec.push_back(res);
        }

        A.push_back(vec);
    }

    vec2d::iterator row_A;
    std::vector<double>::iterator col_A;

    for (row_A = A.begin(); row_A != A.end(); row_A++)
    {
        std::vector<double> vec;

        for (col_A = row_A->begin(); col_A != row_A->end(); col_A++)
        {
            *col_A = *col_A / sum;
        }
    }

    return A;
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

void back_prop(const vec2d &Z1, const vec2d &A1, const vec2d &Z2, const vec2d &A2, const vec2d &W2, const vec2d X,
               const vec2d Y, vec2d &d_W1, vec2d &d_b1, vec2d &d_W2, vec2d &d_b2)
{
    int m = Y.size();
    vec2d output_error = A2 - transpose(one_hot_encode(Y, 10));

    vec2d d_W = (1.0 / m) * (mul(output_error, transpose(A1)));
    vec2d d_b = (1.0 / m) * element_sum(output_error);

    // vec2d z1_error = mul(transpose(W2), output_error) * ReLU_derivative(Z1);
    // implement element-wise multiplication
}