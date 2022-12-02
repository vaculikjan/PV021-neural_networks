#include "activation_functions.hpp"
#include <cmath>

vec ReLU(const vec &Z)
{
    vec res(Z.size());

    int i;
#pragma omp parallel for private(i) shared(Z, res)
    for (i = 0; i < Z.size(); i++)
    {
        res[i] = std::max(Z[i], 0.0);
    }

    return res;
}

vec ReLU_derivative(const vec &Z)
{
    vec res(Z.size());

    int i;
#pragma omp parallel for private(i) shared(Z, res)
    for (i = 0; i < Z.size(); i++)
    {
        if (Z[i] > 0)
        {
            res[i] = 1;
        }
        else
        {
            res[i] = 0;
        }
    }

    return res;
}

vec softmax(const vec &Z, int row, int col)
{
    vec soft_vec(row * col);

    for (int i = 0; i < row; i++)
    {
        double sum = 0;
        for (int j = 0; j < col; j++)
        {
            double res = std::exp(Z[i * col + j]);
            sum += res;
            soft_vec[i * col + j] = res;
        }

        for (int j = 0; j < col; j++)
        {
            soft_vec[i * col + j] = soft_vec[i * col + j] / sum;
        }
    }

    return soft_vec;
}
