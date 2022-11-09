#pragma once

#include <vector>

typedef std::vector<std::vector<double>> vec2d;

/**
 * @brief Multiply two 2D matrices.
 *
 * Takes two matrices of dimensions MxN and NxB (i.e. the number of columns in first matrix has to be
 * equal to the number of rows in the second) and returns a new matrix that is the multiplication of the previous
 * matrices. The product has dimensions MxB.
 *
 * @param v1 First matrix in the form of std::vector<std::vector<double>>.
 * @param v2 Second matrix in the form of std::vector<std::vector<double>>.
 * @return vec2d Product of given matrices.
 */
vec2d mul(const vec2d &v1, const vec2d &v2);

/**
 * @brief Transpose a matrix.
 *
 * Takes a matrix of dimensions MxN and changes the rows to be columns and vice versa.
 * In other words it does a reflection over the main diagonal.
 * The transposed matrix has dimensions NxM.
 *
 * @param v Input matrix in the form of std::vector<std::vector<double>>
 * @return vec2d Transposed matrix.
 */
vec2d transpose(const vec2d &v);

vec2d operator+(const vec2d &v1, const vec2d &v2);
vec2d operator-(const vec2d &v1, const vec2d &v2);
vec2d operator*(double s, const vec2d &v);
// vec2d operator*(double s, const TwoDPivotWrapper<vec2d> &v);
vec2d element_sum(const vec2d &v);
vec2d element_mul(const vec2d &v1, const vec2d &v2);