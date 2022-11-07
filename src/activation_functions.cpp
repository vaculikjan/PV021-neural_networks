#include "activation_functions.hpp"
#include <cmath>

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
    std::vector<double>::iterator sum_it;

    vec2d A;

    for (row = Z.begin(); row != Z.end(); row++)
    {
        std::vector<double> vec;
        double sum = 0;

        for (col = row->begin(); col != row->end(); col++)
        {
            double res = std::exp(*col);
            sum += res;
            vec.push_back(res);
        }

        for (sum_it = vec.begin(); sum_it != vec.end(); sum_it++)
        {
            *sum_it = *sum_it / sum;
        }

        A.push_back(vec);
    }

    return A;
}
