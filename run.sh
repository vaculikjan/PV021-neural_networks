#!/bin/bash
## change this file to your needs

module add gcc

echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network
g++ -g src/activation_functions.cpp src/math.cpp src/misc.cpp src/mnist_fashion.cpp src/neural_network.cpp -o mnist_fashion -fopenmp -ffast-math -O3

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network
./mnist_fashion
