#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <stdexcept>

using namespace std;

namespace Activation {
    inline double relu (double x ) { return std::max(0.0 , x);}
    inline double reluDerivative (double x ) { return (x > 0) ? 1.0 : 0.0 ;}
    inline double sigmoid (double x) { return 1.0 / (1.0 + exp(-x));}
    inline double sigmoidDerivative (double x) {
        double t = sigmoid (x) ;
        return t * (1 - t);
    }
}

class Martrix 
{
private:
    std::vector<std::vector<double>> data;
    size_t rows , cols; 

public:
    Martrix (size_t r , size_t c) : rows(r) , cols(c)
    {
        data.resize (rows , std::vector<double>(cols , 0.0f));  
    }

    double &operator() (size_t i , size_t j) { return data[i][j];}

    const double &operator() (size_t i , size_t j) const {return data[i][j];}

    size_t getRows() const { return rows;}
    size_t getCols() const { return cols;}


};


class NetralNetwork 
{
private:
    vector<int> layerSize ;
    Martrix weight1 , weight2 , weight3 ;
    vector<double> bias1 , bias2 , bias3;  // 神经元偏执
    mt19937 gen ;   // 生成随机数
  
    void initalizeWeights()
    {
        normal_distribution<> dist(0.0 , 1.0);
        double scale1 = sqrt( 2.0 / layerSize[0] );
        double scale2 = sqrt( 2.0 / layerSize[1] );
        double scale3 = sqrt( 2.0 / layerSize[2] );

        for (int i = 0 ; i < weight1.getRows() ; i++)
        {
            for (int j = 0 ; j < weight1.getCols() ; j++)
            {
                weight1(i , j) = dist(gen) * scale1;
            }
        }

        for (int i = 0 ; i < weight2.getRows() ; i++)
        {
            for (int j = 0 ; j < weight2.getCols() ; j++)
            {
                weight2(i , j) = dist(gen) * scale2 ;
            }
        }
        
         for (int i = 0 ; i < weight3.getRows() ; i++)
        {
            for (int j = 0 ; j < weight3.getCols() ; j++)
            {
                weight3(i , j) = dist(gen) * scale3 ;
            }
        }
        
        fill(bias1.begin() , bias1.end() , 0.0);
        fill(bias2.begin() , bias2.end() , 0.0);
        fill(bias3.begin() , bias3.end() , 0.0);
    }

public: 
    NetralNetwork(int inputSize , int hiddenSize1 , int hiddenSize2 , int outputSize)
    : layerSize{inputSize , hiddenSize1 , hiddenSize2 , outputSize} , weight1(inputSize, hiddenSize1)
    , weight2(hiddenSize1 , hiddenSize2) , weight3(hiddenSize2 , outputSize) ,bias1(hiddenSize1)
    ,bias2(hiddenSize2), bias3(outputSize), gen(random_device{}())
    {
        if(inputSize < 0 || hiddenSize1 < 0 || hiddenSize2 < 0 || outputSize < 0)
        {
            throw invalid_argument("please input positive number");
        }
        initalizeWeights();   
    }
    vector<double> forward (const vector<double>& input)
    {
        if (input.size() != layerSize[0] )
        {
            throw runtime_error("input size mismatch");
        }

        vector<double> hidden1 (layerSize[1]);  //用于存储第一个隐藏值的激活值
        for (int j = 0; j < layerSize[1] ; j++)
        {
            double sum = bias1[j];
            for (int i = 0 ; i < layerSize[0] ; i++)
            {
                sum += input[i] * weight1(i , j);
            }
            hidden1[j] = Activation::relu(sum);
        }
        vector<double> hidden2 (layerSize[2]);
        for (int j = 0 ; j < layerSize[2] ; j++)
        {
            double sum = bias2[j];
            for (int i = 0; i < layerSize[1]; i++)
            {
                sum += hidden1[i] * weight2(i,j);
            }
            hidden2[j] = Activation::relu(sum);
            
        }
        vector <double> output (layerSize[3]);
        for (int j = 0; j < layerSize[3]; j++)
        {
            double sum = bias3[j];
            for (int i = 0; i < layerSize[2]; i++)
            {
                sum += hidden2[i] * weight3(i , j);
            }
            output[j] = Activation::sigmoid(sum);
            
        }
        
        return output;   
    }
    void train(const vector<vector<double>>& inputs , const vector <vector<double>>& targets , double learningRat , int epochs)
    {
        if (inputs.size() != targets.size())
        {
            throw runtime_error("Inputs ant Targets size don't match");
        }

        for (int epoch = 0 ; epoch < epochs ; epoch++)
        {
            double totalErrol = 0;
            for(size_t k = 0 ; k < inputs.size() ; k++)
            {
                vector<double> hidden1 (layerSize[1]);
                vector<double> hidden1Pre (layerSize[i]);
                for (int j = 0; j < layerSize[1]; j++)
                {
                    double sum = bias1[j];
                    for (int i = 0; i < layerSize[0]; i++)
                    {
                        sum += inputs[k][i] * weight1(i , j);
                    }
                    hidden1Pre[j] = sum;
                    hidden1[j] = Activation::relu(sum);
                       
                }

                vector<double> hidden2(layerSize[2]);
                vector<double> hidden2Pre(layerSize[2]);
                for (int j = 0 ; j < layerSize[2] ; j++)
                {
                    double sum = bias2[j];
                    for (int i = 0; i < layerSize[1]; i++)
                    {
                        sum += hidden1[i] * weight2(i , j);
                    }
                    hidden2Pre[j] = sum;
                    hidden2[j] = Activation::relu(sum);
                }

                vector<double> output(layerSize[3]);
                vector<double> outputPre(layerSize[3]);
                for (int j = 0 ; j < layerSize[3] ; j++)
                {
                    double sum = bias3[j];
                    for (int i = 0; i < layerSize[2]; i++)
                    {
                        sum += hidden2[i] * weight3(i,j);             
                    }
                    outputPre[j] = sum;
                    output[j] = Activation::sigmoid(sum);                    
                }
                
                for (int j = 0; j < layerSize[3]; j++)
                {
                    double error = targets[k][j] - output[j];
                    totalErrol = error * error;
                }

                vector<double> outputGradients(layerSize[3]);
                for (int j = 0 ; j < layerSize[3] ; j++)
                {
                    outputGradients[j] = (output[j] - targets[k][j]) * Activation::sigmoidDerivative(outputPre[j]);
                }
                vector<double> hidden2Gradients(layerSize[2]);
                for (int i = 0; i < layerSize[2]; i++)
                {
                    double error = 0;
                    for (int j = 0; j < layerSize[3]; j++)
                    {
                        error += outputGradients[j] * weight3(i , j);
                    }
                    hidden2Gradients[i] = error * Activation::reluDerivative(hidden2Pre[i]);
                }
                
                vector<double> hidden1Gradients(layerSize[1]);
                for (int i = 0 ; i < layerSize[1]; i++)
                {
                    double error = 0;
                    for (int j = 0 ; j < layerSize[2];j++)
                    {
                        error += hidden2Gradients[j] * weight2(i , j);
                    }
                    hidden1Gradients[i] = error * Activation::reluDerivative(hidden1Pre[i]);
                } 

                for (int i = 0 ; i < layerSize[2];i++)
                {
                    for (int j = 0 ; j < layerSize[3]; j++)
                    {
                        weight3(i , j) -= learningRat * outputGradients[j] * hidden2[i];
                    }
                }

                for (int i = 0 ; i < layerSize[3]; i++)
                {
                    bias3[i] -= learningRat * outputGradients[i];
                }

                for (int i = 0 ; i < layerSize[1] ; i++)
                {
                    for (int j = 0; j < layerSize[2]; j++)
                    {
                        weight2(i , j) -= learningRat * hidden2Gradients[j] * hidden1[i];
                    }
                }

                for (int i = 0 ; i < layerSize[2] ; i++)
                {
                    bias2[i] -= learningRat * hidden2Gradients[i];
                }

                for (int i = 0 ; i < layerSize[0] ; i++)
                {
                    for (int j = 0 ; j < layerSize[1];j++)
                    {
                        weight1(i , j) -= learningRat * hidden1Gradients[j] * inputs[k][i];
                    }
                }

                for (int j = 0 ; j < layerSize[1];j++)
                {
                    bias1[j] -= learningRat * hidden1Gradients[j];
                }
               
            }
             if(epoch % 100 == 0)
                {
                    std::cout << "EPOCH : " << epoch << "MSE : " << totalErrol / inputs.size() << "\n";
                }

        }

    }
};

int main ()
{
    try{
        NetralNetwork nn (2 , 2 , 4 , 1);
        mt19937 gen(random_device{}());
        uniform_real_distribution<> dist(-2.0 , 2.0);
        const int numSample = 5000 ;    // the number of Sample for training
        vector<vector<double>> inputs(numSample);
        vector<vector<double>> targets(numSample);

        for (int i = 0; i < numSample; i++)
        {
            // double x = dist(gen);
            // double y = dist(gen);
            // inputs[i] = {x , y};
            // double distance = sqrt(x*x + y*y);
            // targets[i] = {distance < 1.0 ? 1.0 : 0.0};

            double x, y, distance;
            // 强制一半在圆内，一半在圆外
            do {
                x = dist(gen);
                y = dist(gen);
                distance = sqrt(x*x + y*y);
            } while ((i < numSample / 2 && distance >= 1.0) || (i >= numSample / 2 && distance < 1.0));
            inputs[i] = {x, y};
            targets[i] = {distance < 1.0 ? 1.0 : 0.0};    
        }

        auto start = chrono::high_resolution_clock::now();
        nn.train(inputs , targets , 0.01 , 1000);
        auto end = chrono::high_resolution_clock::now();
        std::cout << "Traning time is " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms\n"; 


        vector<vector<double>> testpoints = {
            {0.0 , 0.0},
            {1.0 , 0.0},
            {0.5 , 0.5},
            {2.0 , 0.0}

        };
        std::cout << "\n test result (1 = inside , 0 = outside) :" << std::endl;

        for (const auto& points : testpoints )
        {
            auto output = nn.forward(points);
            double actual = sqrt(points[0]*points[0] + points[1]*points[1]) < 1.0 ? 1.0 : 0.0;
            cout << "point (" << points[0] << "," << points[1] << ") : "
            << output[0] << "( actual : " << actual << " . error : " << abs(output[0] - actual) << ") \n";
        }
    }
    catch (const exception& e)   // print error messages
    {
        cerr << "ERROR : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}