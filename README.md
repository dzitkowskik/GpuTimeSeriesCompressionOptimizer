# OPTYMALIZATOR KOMPRESJI SZEREGÓW CZASOWYCH NA KARTACH GPU W TECHNOLOGII CUDA #
# TIME SERIES COMPRESSION OPTIMIZER ON GPU USING CUDA #

### Requirements ###

* CUDA 7
* Cuda device with compute capability greater than 2.0
* gtest, google benchmark, log4cplus, Thrust
* boost
  * to install boost on Ubuntu type:
  ```
  sudo apt-get install libboost-all-dev
  ```


### Install ###

To install application run following instructions:
```
git clone https://dzitkowskik@bitbucket.org/dzitkowskik/gpustore.git
cd gpuStore
cd install
./log4plus_install.sh
./gtest_install.sh
./benchmark_install.sh
cd ..
```
After that make sure that you have these env variables set:
```
export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```


### How do I get set up? ###

* make test - to build tests
* make benchmark - to build benchmarks
* make - to build an application
* make run - to run (tests/benchmarks/application)

### Licence ###

The MIT License (MIT)

Copyright (c) 2015 Karol Dzitkowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
