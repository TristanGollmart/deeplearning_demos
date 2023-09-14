#include <iostream>
#include <vector>
#include <cmath>

double function(const std::vector<double>& x){
    double sum = 0.0;
    for (double xi: x){
        sum += xi * xi;
    }
    return sum;
}

void gradient(const std::vector<double>&  x0, std::vector<double>& grad){
    // pass grad by reference to update it in place
    // pass x by reference to avoid making a copy and thus make function more memory efficient for large vectors
    for (int i=0;i<x0.size(); ++i){
        grad[i] = 2 * x[i];
    }
}

void gradient_descent(std::vector<double>& x0, double learning_rate, int iterations){
    std::vector<double> grad(x0.size);

    for (int i; i< iterations; i++){
        gradient(x0, grad);
        for(size_t j=0;j<x0.size(),++j){
            x0[j] -= learning_rate * grad[j];
        }
    }
}

int main(){
    int dimension = 3;
    std::vector<double> x_init(dimension, 3.0);
    double learning_rate = 0.1;
    int iterations = 100;

    gradient_descent(x_init, learning_rate, iterations);

    std::cout << "minimum found at (";
    for (double xi: x_init){
        std::cout << xi;
    }
    std::cout << ")" << std::endl;
    std::cout << "minimum value: " << function(x_init) << std::endl;
    return 0;
}