#include "math.hpp"

#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>

vec2d mul(const vec2d &v1, const vec2d &v2)
{

    // get dimensions of matrices
    int r1 = v1.size();
    int c1 = v1[0].size();

    int r2 = v2.size();
    int c2 = v2[0].size();

    // set up product matrix
    vec2d mul;

    for (int i = 0; i < r1; i++)
    {
        /*if (r2 != v1[i].size())
        {
            throw std::invalid_argument("Matrices have incompatible dimensions.");
        }*/

        mul.push_back(std::vector<double>(c2, 0));
    }

    // multiplication calculation

    int i, j, k;
#pragma omp parallel for private(i, j, k) shared(v1, v2)
    for (i = 0; i < r1; i++)
        for (k = 0; k < c1; k++)
            for (j = 0; j < c2; j++)
            {
                mul[i][j] += v1[i][k] * v2[k][j];
            }

    return mul;
}

vec2d transpose(const vec2d &v)
{

    // get dimensions of the matrix
    int r = v.size();
    int c = v[0].size();

    // set up product matrix
    vec2d transpose(c, std::vector<double>(r));
    int i, j;

    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
        {
            transpose[j][i] = v[i][j];
        }

    return transpose;
}

// adds a vector to a matrix
vec2d operator+(const vec2d &v1, const vec2d &v2)
{

    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d res;

    int i = 0;
    for (row = v1.begin(); row != v1.end(); row++)
    {
        std::vector<double> vec;

        for (col = row->begin(); col != row->end(); col++)
        {
            vec.push_back(*col + v2[i][0]);
        }

        i++;
        res.push_back(vec);
    }

    return res;
}

vec2d operator-(const vec2d &v1, const vec2d &v2)
{

    vec2d res;

    for (int i = 0; i < v1.size(); i++)
    {
        std::vector<double> vec;

        for (int j = 0; j < v1[0].size(); j++)
        {
            vec.push_back(v1[i][j] - v2[i][j]);
        }

        res.push_back(vec);
    }

    return res;
}

vec2d operator*(double s, const vec2d &v)
{

    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d res;

    for (row = v.begin(); row != v.end(); row++)
    {
        std::vector<double> vec;

        for (col = row->begin(); col != row->end(); col++)
        {
            vec.push_back(*col * s);
        }
        res.push_back(vec);
    }

    return res;
}
/*
vec2d operator*(double s, const TwoDPivotWrapper<vec2d> &v)
{
    std::vector<std::vector<double> > table;
    int R = v.object.size();
    for (int i = 0; i < R; i++)
    {
        // construct a vector of int
        std::vector<double> vec;
        int C = v.object[0].size();
        for (int j = 0; j < C; j++)
        {
            vec.push_back(v.object[i][j] * s);
        }
    // push back above one-dimensional vector
    table.push_back(vec);
    }

    return table;
}
*/

vec2d element_sum(const vec2d &v)
{

    vec2d::const_iterator row;
    std::vector<double>::const_iterator col;

    vec2d res;

    for (row = v.begin(); row != v.end(); row++)
    {
        double sum = 0;
        for (col = row->begin(); col != row->end(); col++)
        {
            sum += *col;
        }
        std::vector<double> vec{sum};
        res.push_back(vec);
    }

    return res;
}

vec2d element_mul(const vec2d &v1, const vec2d &v2)
{

    vec2d res;

    for (int i = 0; i < v1.size(); i++)
    {
        std::vector<double> vec;

        for (int j = 0; j < v1[0].size(); j++)
        {
            vec.push_back(v1[i][j] * v2[i][j]);
        }

        res.push_back(vec);
    }

    return res;
}
