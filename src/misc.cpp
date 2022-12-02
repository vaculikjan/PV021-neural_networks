#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

void load_csv(std::string file_path, std::vector<double> &vec)
{
    ifstream file;
    string line;

    file.open(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file");
    }

    while (getline(file, line))
    {
        string tmp;
        stringstream input(line);

        while (getline(input, tmp, ','))
        {
            vec.push_back(stod(tmp));
        }
    }

    return;
}

void measuretime(std::chrono::time_point<std::chrono::steady_clock> start, std::string message)
{
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << message << ": " << std::chrono::duration<double, std::milli>(diff).count() / 1000 << "s" << std::endl;
}

int check_accuracy(const vec &pred, const vec &Y)
{
    int number_of_examples = Y.size();
    int correct_predictions = 0;

    for (int i = 0; i < number_of_examples; i++)
    {
        if (Y[i] == pred[i])
        {
            correct_predictions++;
        }
    }

    return correct_predictions;
}

vec one_hot_encode(const vec &y)
{
    vec one_hot_Y(y.size() * 10);

    int i;
#pragma omp parallel for private(i) shared(y, one_hot_Y)
    for (i = 0; i < y.size(); i++)
    {
        one_hot_Y[10 * i + y[i]] = 1;
    }

    return one_hot_Y;
}

vec one_hot_decode(const vec &one_hot_Y)
{
    vec y(one_hot_Y.size() / 10);

    for (int i = 0; i < one_hot_Y.size() / 10; i++)
    {
        int index = 0;
        double max = 0;

        for (int j = 0; j < 10; j++)
        {
            if (one_hot_Y[i * 10 + j] > max)
            {
                max = one_hot_Y[i * 10 + j];
                index = j;
            }
        }
        y[i] = index;
    }

    return y;
}