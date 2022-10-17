#include "csv_loader.hpp"
#include "math.hpp"

int main()
{

    vec2d v1{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
    };

    vec2d v2{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    };

    mul(v1, v2);
    transpose(v1);
    vec2d labels = load_csv("../data/fashion_mnist_train_vectors.csv");

    return 0;
}