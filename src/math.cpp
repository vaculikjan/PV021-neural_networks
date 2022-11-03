#include "math.hpp"

#include <iostream>
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
        if (r2 != v1[i].size())
        {
            throw std::invalid_argument("Matrices have incompatible dimensions.");
        }

        mul.push_back(std::vector<double>(c2, 0));
    }

    // multiplication calculation
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c2; j++)
            for (int k = 0; k < c1; k++)
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
    vec2d transpose;

    for (int i = 0; i < c; i++)
    {
        transpose.push_back(std::vector<double>(r, 0));
    }

    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
        {
            transpose[j][i] = v[i][j];
        }

    return transpose;
}

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
