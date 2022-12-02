#include "math.hpp"

#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>

// matrix multiplication
vec mul(const vec &v1, const vec &v2, int r1, int c1, int r2, int c2)
{
    int size = r1 * c2;
    vec mul(size);

    int i, j, k;
#pragma omp parallel for private(i, j, k) shared(v1, v2)
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            double sum = 0;
            for (int k = 0; k < c1; k++)
                sum = sum + v1[i * c1 + k] * v2[k * c2 + j];
            mul[i * c2 + j] = sum;
        }
    }

    return mul;
}

// matrix transposition
vec transpose(const vec &v, int row, int col)
{
    vec trans(row * col);
    int k = 0;
    int i, j;
    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < row; j++)
        {
            trans[k++] = v[j * col + i];
        }
    }

    return trans;
}

// adds a vector to a matrix
void add_vec_to_columns(vec &v1, const vec &v2, int row, int col)
{

    int i, j;
#pragma omp parallel for private(i, j) shared(v1, v2)
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            v1[i * col + j] += v2[i];
        }

    return;
}

// matrix subtraction
void matrix_subtract(vec &v1, const vec &v2, int row, int col)
{
    int i, j;
#pragma omp parallel for private(i, j) shared(v1, v2)
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            v1[i * col + j] -= v2[i * col + j];
        }

    return;
}

// multiply matrix by scalar
void multiply_by_scalar(double s, vec &v, int row, int col)
{
    int i, j;
#pragma omp parallel for private(i, j) shared(v, s)
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            v[i * col + j] *= s;
        }
    return;
}

// sum elements into one column
vec element_sum(const vec &v, int row, int col)
{

    vec res(row);

    for (int i = 0; i < row; i++)
    {
        double sum = 0;
        for (int j = 0; j < col; j++)
        {
            sum += v[i * col + j];
        }
        res[i] = sum;
    }

    return res;
}

// elementwise matrix multiplication
vec element_mul(const vec &v1, const vec &v2, int row, int col)
{
    vec res(row * col);
    int i, j;
#pragma omp parallel for private(i, j) shared(v1, v2)
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            res[i * col + j] = v1[i * col + j] * v2[i * col + j];
        }
    return res;
}
