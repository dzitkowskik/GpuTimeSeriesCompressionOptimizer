#!/bin/bash

sudo apt-get install cmake
mkdir benchmarkInstallTemp
cd benchmarkInstallTemp
git clone https://github.com/google/benchmark.git
cd benchmark
cmake .
sudo make install
cd ../..
rm -rf benchmarkInstallTemp/
