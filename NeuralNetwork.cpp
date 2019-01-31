#include "NeuralNetwork.hpp"

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>

#define E 2.71828182846

//helper functions
namespace NUti
{
    int generateInt(int min, int max)
    {
        return ( rand()% (abs(min-max)+1) )+std::min(min, max);
    }

    double generateDouble(double min,double max, const int nbrDeci)
    {
        double p = pow(10,nbrDeci);

        return generateInt(min*p,max*p)/p;
    }

    template <class T>
    void write(std::ofstream& ofs,const T& data)
    {
        ofs.write( (char*)&data , sizeof(T));
    }

    template <class T>
    T read(std::ifstream& ifs)
    {
        char* data = new char[sizeof(T)];

        ifs.read(data,sizeof(T));

        const T result = *(T*)data;

        delete data;

        return result;
    }
};


NeuralNetwork::NeuralNetwork(void)
{}

void NeuralNetwork::create(const std::vector<int>& structure)
{
    mLayers.clear();

    for(std::size_t i = 0; i < structure.size(); i++)
    {
        mLayers.push_back( std::vector<Node>(structure[i]) );
    }

    for(std::size_t a = 0; a+1 < mLayers.size(); a++)
    {
        for(std::size_t b = 0; b < mLayers[a].size(); b++)
        {
            mLayers[a][b].vars = std::vector<double>( mLayers[a+1].size() );
            for(std::size_t c = 0; c < mLayers[a+1].size(); c++)
            {
                //weights are in the range [-2;2]
                mLayers[a][b].weights.push_back( NUti::generateDouble(-2,2,6) );
            }
        }
    }

}

void NeuralNetwork::create(const std::vector< std::vector< std::vector<double> > >& data)
{
    mLayers.clear();

    mLayers = std::vector< std::vector<Node> >( data.size() );
    for(std::size_t a = 0; a < data.size(); a++)
    {
        mLayers[a] = std::vector<Node>( data[a].size() );
        for(std::size_t b = 0; b < data[a].size(); b++)
        {
            mLayers[a][b].weights = data[a][b];
            mLayers[a][b].vars = std::vector<double>(data[a][b].size(),0);
        }
    }
}




std::vector<double> NeuralNetwork::forward(const std::vector<double>& input)
{
    if(mLayers.empty())
    {
        std::cout << "Error: std::vector<double> NeuralNetwork::forward(std::vector<double>&) mLayer is empty." << std::endl;
        return {};
    }

    std::vector<Node>& inputLayer = mLayers.at(0);

    if(inputLayer.size() != input.size())
    {
        std::cout << "Error: std::vector<double> NeuralNetwork::forward(std::vector<double>&) input size not matching." << std::endl;
        return {};
    }

    for(std::size_t a = 0; a < inputLayer.size(); a++)
    {
        inputLayer[a].value = input[a];
    }


    for(std::size_t a = 1; a < mLayers.size(); a++)
    {
        for(std::size_t b = 0; b < mLayers[a].size(); b++)
        {
            double value = 0;
            for(std::size_t c = 0; c < mLayers[a-1].size(); c++)
            {
                value += mLayers[a-1][c].value*mLayers[a-1][c].weights[b];
            }

            mLayers[a][b].value = sigma(value);
        }
    }

    std::vector<double> output;
    for(std::size_t i = 0; i < mLayers.back().size(); i++)
    {
        output.push_back( mLayers.back()[i].value );
    }

    return output;

}

void NeuralNetwork::reinitialize(void)
{
    for(std::size_t a = 0; a < mLayers.size(); a++)
    {
        for(std::size_t b = 0; b < mLayers[a].size(); b++)
        {
            mLayers[a][b].value = 0;
            mLayers[a][b].gradient = 0;
        }
    }
}

void NeuralNetwork::backward(const std::vector<double>& wanted)
{

    if(mLayers.empty() or wanted.size() != mLayers.back().size())
    {
        std::cout << "Error: void NeuralNetwork::backward(const std::vector<double>&) size not matching " << wanted.size() << " " << mLayers.back().size() << std::endl;
        return;
    }

    for(std::size_t i = 0; i < mLayers.back().size(); i++)
    {
        mLayers.back()[i].gradient = (wanted[i]-mLayers.back()[i].value)*sigma_derivative( mLayers.back()[i].value );
    }

    for(auto at = mLayers.rbegin()+1; at != mLayers.rend(); at++)
    {
        const std::vector<Node>& nextLayer = *(at-1);
        for(std::size_t b = 0; b < at->size(); b++)
        {
            double error = 0;
            for(std::size_t c = 0; c < nextLayer.size(); c++)
            {
                error += (*at)[b].weights[c]*nextLayer[c].gradient;
            }

            (*at)[b].gradient = error*sigma_derivative( (*at)[b].value );

            for(std::size_t c = 0; c < nextLayer.size(); c++)
            {
                (*at)[b].vars[c] += nextLayer[c].gradient*(*at)[b].value;
            }
        }
    }
}

void NeuralNetwork::applyLearning(const double learningRate)
{
    for(std::size_t a = 0; a < mLayers.size(); a++)
    {
        for(std::size_t b = 0; b < mLayers[a].size(); b++)
        {
            Node& node = mLayers[a][b];
            for(std::size_t c = 0; c < node.vars.size(); c++)
            {
                node.weights[c] += node.vars[c]*learningRate;
                node.vars[c] = 0;
            }

            //node.bias += node.biasVar*learningRate;
            //node.biasVar = 0;
        }
    }
}


int NeuralNetwork::getInputSize(void) const
{
    if(mLayers.empty())
    {
        return 0;
    }

    return mLayers.front().size();
}

int NeuralNetwork::getOutputSize(void) const
{
    if(mLayers.empty())
    {
        return 0;
    }

    return mLayers.back().size();
}


std::vector< std::vector< std::vector<double> > > NeuralNetwork::getData(void) const
{
    std::vector< std::vector< std::vector<double> > > data(mLayers.size());
    for(std::size_t a = 0; a < mLayers.size(); a++)
    {
        for(std::size_t b = 0; b < mLayers[a].size(); b++)
        {
            data[a].push_back( mLayers[a][b].weights );
        }
    }

    return data;
}

double NeuralNetwork::sigma(const double x) const
{
    return 1 / (1 + (exp((double)-x)) );
}

double NeuralNetwork::sigma_derivative(const double x) const
{
    return x*(1-x);
}


bool NeuralNetwork::loadAndCreate(const std::string filename)
{
    NeuralData data = load(filename);
    if(data.empty())
    {
        return false;
    }

    create(data);
    return true;
}

NeuralData NeuralNetwork::load(const std::string filename)
{
    std::ifstream ifs;
    ifs.exceptions( std::ifstream::failbit | std::ifstream::badbit);

    std::vector< std::vector< std::vector<double> > > data;

    try
    {
        ifs.open(filename);
    }
    catch(std::exception& e)
    {
        std::cout << "Error: NeuralData NeuralNetwork::load(std::string) Exception " << std::strerror(errno) << " failed to open file |" << filename << "|." << std::endl;
        return data;
    }

    const std::uint64_t layerNumber = NUti::read<std::uint64_t>(ifs);

    for(std::uint64_t a = 0; a < layerNumber; a++)
    {
        const std::uint64_t nodeNumber = NUti::read<std::uint64_t>(ifs);
        data.push_back( std::vector< std::vector<double> >(nodeNumber) );

        for(std::uint64_t b = 0; b < nodeNumber; b++)
        {
            const std::uint64_t weightNumber = NUti::read<std::uint64_t>(ifs);
            data[a][b] = std::vector<double>(weightNumber,0);

            for(std::uint64_t c = 0; c < weightNumber; c++)
            {
                data[a][b][c] = NUti::read<double>(ifs);
            }
        }
    }

    ifs.close();

    return data;
}

bool NeuralNetwork::saveData(const std::string filename) const
{
    return saveData(filename, getData());
}

bool NeuralNetwork::saveData(const std::string filename, const NeuralData& data)
{
    if(sizeof(double) != 8)
    {
        std::cout << "ERROR: bool NeuralNetwork::saveData(std::vector< std::vector< std::vector<double> > >) sizeof(double) isn't 8." << std::endl;
        return false;
    }

    std::ofstream ofs(filename, std::ofstream::out);

    NUti::write(ofs, std::uint64_t(data.size()) );

    for(std::size_t a = 0; a < data.size(); a++)
    {
        NUti::write(ofs, std::uint64_t(data[a].size()) ); ///layer size
        for(std::size_t b = 0; b < data[a].size(); b++)
        {
            NUti::write(ofs, std::uint64_t(data[a][b].size()) ); ///weights size

            for(std::size_t c = 0; c < data[a][b].size(); c++)
            {
                NUti::write(ofs, double(data[a][b][c]) );
            }
        }
    }

    ofs.close();

    return true;
}

bool NeuralNetwork::normalize(std::vector<double>& vec)
{
    if(vec.empty())
    {
        return true;
    }

    double max = vec.at(0);
    double min = vec.at(0);

    for(std::size_t i = 0; i < vec.size(); i++)
    {
        max = std::max(max,vec[i]);
        min = std::min(min,vec[i]);
    }

    const double dif = max-min;

    if(dif == 0)
    {
        return false;
    }

    for(std::size_t i = 0; i < vec.size(); i++)
    {
        vec[i] = (vec[i]-min)/dif;
    }

    return true;
}


bool NeuralNetwork::normalizeSet(std::vector< std::vector<double> >& set)
{
    if(set.empty())
    {
        return true;
    }

    const int itemSize = set.at(0).size();
    for(std::size_t i = 0; i < set.size(); i++)
    {
        if(set[0].size() != itemSize)
        {
            std::cout << "bool NeuralNetwork::normalizeSet not all set items have the same size." << std::endl;
            return false;
        }
    }

    if(itemSize == 0)
    {
        return true;
    }


    double max = set.at(0).at(0);
    double min = set.at(0).at(0);

    for(std::size_t a = 0; a < set.size(); a++)
    {
        for(std::size_t b = 0; b < set[a].size(); b++)
        {
            max = std::max(max,set[a][b]);
            min = std::min(min,set[a][b]);
        }
    }

    const double dif = max-min;

    if(dif == 0)
    {
        return false;
    }

    for(std::size_t a = 0; a < set.size(); a++)
    {
        for(std::size_t b = 0; b < set[a].size(); b++)
        {
            set[a][b] = (set[a][b]-min)/dif;
        }
    }

    return true;
}


