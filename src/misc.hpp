#pragma once

#include "math.hpp"

#include <array>
#include <chrono>
#include <string>

/**
 * @brief Load csv file.
 *
 * Takes the argument and loads the data as 2 dimensional vector of values with dimensions of LxE where L is the number
 * of lines and E is the number of elements on one line, separated by the <,> delimenter.
 *
 * @param file_path path to the csv file (e.g. [../data/my_training_set.csv])
 * @param vec vector for writing the file content
 */
void load_csv(std::string file_path, std::vector<double> &vec);

/**
 * @brief Print runtime since program started.
 *
 * Calculates and outputs program runtime duration since the execution started.
 *
 * @param start starting point for time measurement
 * @param message string containing message identifying the method call
 */
void measuretime(std::chrono::time_point<std::chrono::steady_clock> start, std::string message);

/**
 * @brief Compares predicted with actual labels.
 *
 * @param pred predicted labels
 * @param Y actual labels
 * @return int number of correct prediction
 */
int check_accuracy(const vec &pred, const vec &Y);

/**
 * @brief Perform one-hot encoding on a vector.
 *
 * One-hot encodes the vector containg the labels. Takes a vector Mx1 and returns a matrix Mx10. Each row contains all
 * zeroes except a single one, on a position of the original label value.
 *
 * @param y labels
 * @return vec one-hot encoded matrix
 */
vec one_hot_encode(const vec &y);

/**
 * @brief Transforms the one-hot encoded matrix to a label vector.
 *
 * Takes a one-hot encoded matrix with dimensions Mx10 and retunrs a vector Mx1 with values corresponding to the
 * position of 1 in each row.
 *
 * @param one_hot_Y one-hot encoded matrix
 * @return vec labels
 */
vec one_hot_decode(const vec &one_hot_Y);
