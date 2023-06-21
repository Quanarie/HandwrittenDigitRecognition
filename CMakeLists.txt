cmake_minimum_required(VERSION 3.24)
project(NeuralNetworkNumberRecognition)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetworkNumberRecognition main.cpp mnistReader.h)
