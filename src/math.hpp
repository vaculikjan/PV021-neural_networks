#pragma once

#include <vector>

typedef std::vector<std::vector<double>> vec2d;
typedef std::vector<double> vec;

/**
 * @brief Multiply two 2D matrices.
 *
 * Takes two matrices of dimensions MxN and NxB (i.e. the number of columns in first matrix has to be
 * equal to the number of rows in the second) and returns a new matrix that is the multiplication of the previous
 * matrices. The product has dimensions MxB.
 *
 * @param v1 first matrix
 * @param v2 second matrix
 * @param r1 number of rows for the first matrix
 * @param c1 number of columns for the first matrix
 * @param r2 number of rows for the second matrix
 * @param c2 number of columns for the second matrix
 * @return vec product of given matrices
 */
vec mul(const vec &v1, const vec &v2, int r1, int c1, int r2, int c2);

/**
 * @brief Transpose a matrix.
 *
 * Takes a matrix of dimensions MxN and changes the rows to be columns and vice versa.
 * In other words it does a reflection over the main diagonal.
 * The transposed matrix has dimensions NxM.
 *
 * @param v input matrix
 * @param row number of rows
 * @param col number of columns
 * @return vec transposed matrix
 */
vec transpose(const vec &v, int row, int col);

/**
 * @brief Add a vector to each row of a matrix.
 *
 * Adds a vector of dimensions Mx1 toa matrix of dimensions MxN.
 *
 * The result is passed by reference in v1.
 *
 * @param v1 input matrix
 * @param v2 input vector
 * @param row number of rows of both the matrix and the vector
 * @param col number of columns of the input matrix
 */
void add_vec_to_columns(vec &v1, const vec &v2, int row, int col);

/**
 * @brief Subtract 2 matrices.
 *
 * Subtracts the second matrix from the first matrix (v1 - v2). The 2 matrices need to have the same dimensions. And the
 * difference keeps them.
 *
 * The result is passed by reference in v1.
 *
 * @param v1 first matrix
 * @param v2 second matrix
 * @param row number of rows for both matrices
 * @param col number of columns for both matrices
 */
void matrix_subtract(vec &v1, const vec &v2, int row, int col);

/**
 * @brief Multiply a matrix by scalar value.
 *
 * The result is passed by reference in v1.
 *
 * @param s scalar
 * @param v input matrix
 * @param row number of rows
 * @param col number of columns
 */
void multiply_by_scalar(double s, vec &v, int row, int col);

/**
 * @brief Sum columns of a matrix.
 *
 * Sums columns of the input matrix with dimensions of MxN into a matrix with dimensions Mx1.
 *
 * @param v input matrix
 * @param row number of rows
 * @param col number of columns
 * @return vec summed vector
 */
vec element_sum(const vec &v, int row, int col);

/**
 * @brief Multiples 2 matrices element-wise.
 *
 * Multiples the elements of 2 matrices with dimensions of MxN each with the corresponding counterpart.
 *
 * @param v1 first matrix
 * @param v2 second matrix
 * @param row number of rows for both matrices
 * @param col number of columns for both matrices
 * @return vec element-multiplied matrix
 */
vec element_mul(const vec &v1, const vec &v2, int row, int col);