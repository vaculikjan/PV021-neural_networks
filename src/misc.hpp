#pragma once

#include "math.hpp"

#include <chrono>
#include <string>

/**
 * @brief Loads csv file.
 *
 * Takes the argument and loads the data as 2 dimensional vector of values with dimensions of LxE where L is the number
 * of lines and E is the number of elements on one line, separated by the <,> delimenter.
 *
 * @param file_path Path to the csv file (e.g. [../data/my_training_set.csv])
 * @return vec2d
 */
vec2d load_csv(std::string file_path);

/**
 * @brief Print runtime since program started.
 *
 * Calculates and outputs program runtime duration since the execution started.
 *
 * @param start Starting point for time measurement.
 * @param message String containing message identifying the method call.
 */
void measuretime(std::chrono::time_point<std::chrono::steady_clock> start, std::string message);