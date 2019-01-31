#include <iostream>
#include <cmath>

#include "NeuralNetwork.hpp"

int main()
{

    NeuralNetwork network;
    network.create({2,12,1});

    std::vector< std::vector<double> > inputs = { {0,2}, {3,4}, {5,1}, {2,1} };
    network.normalizeSet(inputs);

    const int trainingEpoches = 200;
    const double learningRate = 0.01;

    for(int a = 0; a < trainingEpoches; a++)
    {
        double error = 0;
        for(std::size_t i = 0; i < inputs.size(); i++)
        {
            const std::vector<double> output = network.forward(inputs[i]);
            error += pow(output[0]-i%2, 2);

            network.backward({i%2});
            network.applyLearning(learningRate);
        }

        std::cout << a << " Error: " << error << std::endl;
    }

    return 0;
}
