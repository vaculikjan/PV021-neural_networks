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

vec2d load_csv(string file_path)
{
    ifstream file;
    string line;

    file.open(file_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file");
    }

    vec2d data;

    while (getline(file, line))
    {
        string tmp;
        stringstream input(line);
        vector<double> vec;
        while (getline(input, tmp, ','))
        {
            vec.push_back(stof(tmp));
        }
        data.push_back(vec);
    }

    return data;
}

void measuretime(std::chrono::time_point<std::chrono::steady_clock> start, std::string message)
{
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << message << ": " << std::chrono::duration<double, std::milli>(diff).count() / 1000 << "s" << std::endl;
}