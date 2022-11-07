#pragma once

#include "math.hpp"

vec2d ReLU(const vec2d &Z);
vec2d ReLU_derivative(const vec2d &Z);
vec2d softmax(const vec2d &Z);