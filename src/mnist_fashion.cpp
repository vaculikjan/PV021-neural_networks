#include "csv_loader.hpp"
#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <random>

void init_params(vec2d &W, vec2d &b, int rows, int cols);
vec2d ReLU(const vec2d &Z);
vec2d softmax(const vec2d &Z);

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
    // vec2d X_train = transpose(load_csv("../data/fashion_mnist_train_vectors.csv"));
    // vec2d Y_train = transpose(load_csv("../data/fashion_mnist_train_labels.csv"));

    vec2d W;
    vec2d b;

    init_params(W, b, 10, 784);

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

void forward_prop(vec2d &W, vec2d &b, vec2d X)
{
    vec2d Z = mul(W, X) + b;
    vec2d A = ReLU(Z);
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