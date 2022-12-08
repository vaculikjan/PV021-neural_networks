#!/bin/bash

module add gcc

echo "#################"
echo "    COMPILING    "
echo "#################"

g++ -g src/activation_functions.cpp src/math.cpp src/misc.cpp src/mnist_fashion.cpp src/neural_network.cpp -o mnist_fashion -fopenmp -ffast-math -O3

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 ./mnist_fashion