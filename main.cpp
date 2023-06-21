#include <iostream>
#include <cmath>
#include "mnistReader.h"
#include "Eigen/Eigen/Dense"
using namespace std;

Eigen::VectorXd convertStdVecToEigenVec(const vector<double>&);
int indOfMaxNum(Eigen::VectorXd output);

class Network
{
    int amountLayers;
    int* sizesLayers;   // 0th element - size of input layer
    Eigen::VectorXd* biases;    // 0th elem. - vector of biases for connections from first layer (input)
                                // to the second layer (first hidden layer)
    Eigen::MatrixXd* weights;   // 0th elem. - matrix of weights (1 row for 1 neuron in 2 layer ... )

public:
    Network(int aL, int* sL): amountLayers(aL)
    {
        sizesLayers = new int[amountLayers];
        memcpy(sizesLayers, sL, amountLayers*sizeof(int));

        biases = new Eigen::VectorXd[amountLayers - 1];
        weights = new Eigen::MatrixXd[amountLayers - 1];
        for (int i = 0; i < amountLayers - 1; ++i)
        {
            biases[i].resize(sizesLayers[i + 1]);
            biases[i].setRandom();
            weights[i].resize(sizesLayers[i + 1], sizesLayers[i]);
            weights[i].setRandom();
        }
    }

    void GDForInterval(const vector<vector<double>>& input, vector<int> labels, float step, int from, int to)
    {
        Eigen::VectorXd** errors = new Eigen::VectorXd*[to - from];
        Eigen::VectorXd** layersActivations = new Eigen::VectorXd*[to - from];
        for (int j = 0; j < to - from; ++j)   // j - number of input
        {
            Eigen::VectorXd* layersValues = new Eigen::VectorXd[amountLayers];
            layersValues[0] = convertStdVecToEigenVec(input[j + from]);
            layersValues[1] = weights[0]*layersValues[0] + biases[0];
            layersActivations[j] = new Eigen::VectorXd[amountLayers];
            layersActivations[j][0] = layersValues[0];
            layersActivations[j][1] = sigmoid(layersValues[1]);
            for (int i = 2; i < amountLayers; ++i)
            {
                layersValues[i] = weights[i - 1]*layersActivations[j][i - 1] + biases[i - 1];
                layersActivations[j][i] = sigmoid(layersValues[i]);
            }

            errors[j] = new Eigen::VectorXd[amountLayers];

            errors[j][amountLayers - 1] = (layersActivations[j][amountLayers - 1] - getResultVector(labels[j + from])).cwiseProduct(
                    sigmoidPrime(layersValues[amountLayers - 1]));

            // Backpropagate the error
            for (int i = amountLayers - 2; i >= 1; --i)
            {
                errors[j][i] = (weights[i].transpose()*errors[j][i + 1]).cwiseProduct(
                        sigmoidPrime(layersValues[i]));
            }

            delete[] layersValues;
        }

        for (int j = 0; j < to - from; ++j)
        {
            for (int i = 0; i < amountLayers - 1; ++i)
            {
                Eigen::MatrixXd temp = step*errors[j][i + 1]*layersActivations[j][i].transpose();
                weights[i] -= step*errors[j][i + 1]*layersActivations[j][i].transpose();
                for (int k = 0; k < temp.rows(); ++k) {
                    for (int l = 0; l < temp.cols(); ++l) {
                        if (temp(k,l) > 1.0f) cout << temp(k,l);
                    }
                }
                biases[i] -= step*errors[j][i + 1];
            }

            delete[] errors[j];
            delete[] layersActivations[j];
        }

        delete[] errors;
        delete[] layersActivations;
    }

    void SGD(const vector<vector<double>>& input, vector<int> labels, float step, int batchSize, int trainAmount, int epochs)
    {
        for (int l = 0; l < epochs; ++l)
        {
            for (int k = 0; k < trainAmount/batchSize; ++k) // k - number of a batch
            {
                GDForInterval(input, labels, step/batchSize, k*batchSize, (k + 1)*batchSize);
            }
        }
    }

    double testNetwork(const vector<vector<double>>& input, vector<int> labels, int testAmount)
    {
        int correctResults = 0, results = 0;
        for (int i = 0; i < testAmount; ++i)
        {
            Eigen::VectorXd output = feedForward(convertStdVecToEigenVec(input[i]));
            if (labels[i] == indOfMaxNum(output))
            {
                correctResults++;
            }
            results++;
        }
        return (double)correctResults/results;
    }

    Eigen::VectorXd feedForward(Eigen::VectorXd input)
    {
        for (int i = 0; i < amountLayers - 1; ++i)
        {
            input = sigmoid(weights[i]*input + biases[i]);
        }
        return input;
    }

    Eigen::VectorXd feedForward(vector<double> input)
    {
        return feedForward(convertStdVecToEigenVec(input));
    }

    float countCostFunctionValue(Eigen::VectorXd output, int target)
    {
        Eigen::VectorXd res(output.size());
        for (int i = 0; i < res.size(); ++i)
        {
            res[i] = output[i];
            if (i == target) res[i] -= 1.0f;
            res[i] *= res[i];
        }
        return res.sum()/2;
    }

    Eigen::VectorXd getResultVector(int desiredResult)
    {
        Eigen::VectorXd resultVec(sizesLayers[amountLayers - 1]);
        resultVec[desiredResult] = 1.0f;
        return resultVec;
    }

    void printLayerInfo(int num)
    {
        if (num <= 1 || num > amountLayers) {cout << "Wrong num of layer"; return;}
        cout << "Weights: " << endl << weights[num - 2] << endl;
        cout << "Biases: " << endl << biases[num - 2].transpose() << endl;
    }

    Eigen::VectorXd sigmoid(Eigen::VectorXd vec)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            vec[i] = 1/(1 + exp(-vec[i]));
        }
        return vec;
    }

    Eigen::VectorXd sigmoidPrime(Eigen::VectorXd vec)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            vec[i] = exp(-vec[i])/((1 + exp(-vec[i]))*(1 + exp(-vec[i])));
        }
        return vec;
    }

    ~Network()
    {
        delete[] biases;
        delete[] weights;
        delete[] sizesLayers;
    }
};

Eigen::VectorXd convertStdVecToEigenVec(const vector<double>& vec)
{
    Eigen::VectorXd res(vec.size());
    for (int i = 0; i < vec.size(); ++i)
    {
        res[i] = vec[i];
    }
    return res;
}

int indOfMaxNum(Eigen::VectorXd output)
{
    int maxInd = 0;
    for (int i = 1; i < output.size(); ++i)
    {
        if (output[i] > output[maxInd]) maxInd = i;
    }
    return maxInd;
}

void printInput(vector<double> input)
{
    for (int i = 0; i < 784; ++i)
    {
        if (i % 28 == 0) cout << endl;
        if (input[i] > 0) cout << 1;
        else cout << 0;
    }
    cout << endl;
}

int main()
{
    //srand(time(NULL));
    // 28 by 28 input picture, 2 hidden layers of 8 neurons, 10 output neurons
    int amountOfLayers = 4;
    int* sizesOfLayers = new int[amountOfLayers] { 784, 8, 8, 10 };

    Network network(amountOfLayers, sizesOfLayers);

    vector<vector<double>> trainingData;
    readMnistTrainingData("/Users/viacheslavpopov/CLionProjects/NeuralNetworkNumberRecognition/train-images-idx3-ubyte", trainingData);

    vector<int> trainingDataLabels;
    readMnistTrainingDataLabels("/Users/viacheslavpopov/CLionProjects/NeuralNetworkNumberRecognition/train-labels-idx1-ubyte", trainingDataLabels);

    const int trainAmount = 1000;
    float step = 20.0f;
    network.SGD(trainingData, trainingDataLabels, step, 100, trainAmount, 1);
    cout << network.testNetwork(trainingData, trainingDataLabels, trainAmount);

    delete[] sizesOfLayers;
    return 0;
}
