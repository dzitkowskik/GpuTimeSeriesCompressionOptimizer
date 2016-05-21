#!/bin/bash

#install required packages
sudo apt-get install cmake -y

mkdir benchmarkInstallTemp
cd benchmarkInstallTemp
git clone https://github.com/google/benchmark.git
cd benchmark
cmake .
sudo make install
cd ../..
rm -rf benchmarkInstallTemp/
