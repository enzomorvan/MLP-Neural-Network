#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <string>

typedef std::vector< std::vector< std::vector<double> > > NeuralData;

class NeuralNetwork
{
public:
    NeuralNetwork(void);

    ///create a network using a layer structure, ie: {2, 12, 1} for a network with a 2 nodes input layer, 8 nodes hidden layer and 1 node output layer
    void create(const std::vector<int>& structure);

    ///create the network from already existing structure and weights.
    void create(const NeuralData& data);


    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& wanted);
    void applyLearning(const double learningRate);


    ///sets value and gradient of every nodes to 0
    void reinitialize(void);


    ///load structure and weights from previously saved binary file and create the network with it, return false on failure
    bool loadAndCreate(std::string filename);

    ///load structure and weights from previously saved binary file
    static NeuralData load(std::string filname);


    ///save internal structure and weights to a binary file, return false on failure
    bool saveData(std::string filename) const;

    ///save external structure and weights to a binary file, return false on failure
    static bool saveData(std::string filename, const NeuralData& data);



    ///Normalize the values in the range [0;1], return false on failure
    static bool normalize(std::vector<double>&);
    static bool normalizeSet(std::vector< std::vector<double> >&);


    ///get Neural network internal layer structure and weights
    NeuralData getData(void) const;


    int getInputSize(void) const;
    int getOutputSize(void) const;

private:

    ///activation function and its derivative
    double sigma(double) const;
    double sigma_derivative(double) const;

    struct Node
    {
        Node(void) : value(0), gradient(0) {}
        double value;
        double gradient;
        std::vector<double> weights;
        std::vector<double> vars;
    };

    std::vector< std::vector<Node> > mLayers;
};


#endif // NEURALNETWORK_HPP
