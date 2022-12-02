#pragma once

#include "math.hpp"

/**
 * @brief Apply the ReLU activation.
 *
 * Applies the ReLU activation function the input. If the input > 0 then returns input, else returns 0.
 *
 * @param Z input matrix
 * @return vec
 */
vec ReLU(const vec &Z);

/**
 * @brief Get ReLU derivative.
 *
 * Calculates the ReLU derivative. If the input > 0 then returns 1, else returns 0. Undefined for 0, this implementation
 * chooses 0.
 *
 * @param Z input matrix
 * @return vec
 */
vec ReLU_derivative(const vec &Z);

/**
 * @brief Apply the Softmax activation.
 *
 * Applies the Softmax activation function the the input vector. Converts the vector into a probability distribution of
 * vector_size outcomes.
 *
 * @param Z input matrix
 * @param row number of rows
 * @param col nuimber of columns
 * @return vec
 */
vec softmax(const vec &Z, int row, int col);