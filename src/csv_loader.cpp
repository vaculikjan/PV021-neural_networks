#include "csv_loader.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vec2d load_csv(string file_path)
{
    ifstream file;
    string line;

    file.open(file_path);
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