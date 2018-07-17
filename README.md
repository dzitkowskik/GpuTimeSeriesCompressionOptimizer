# TIME SERIES COMPRESSION OPTIMIZER ON GPU USING CUDA #

## OVERVIEW ##

Many devices such as sensors, measuring stations or even servers produce enormous
amounts of data in the form of time series, which are then processed and stored for
later analysis. A huge role in this process takes data processing on graphics cards
in order to accelerate calculations. To efficiently use the GPGPU (General Purpose
Graphical Processing Unit) a number of solutions has been presented, that use the
GPU as a coprocessor in a databases. There were also attempts to create a GPU-side
databases. It has been known that data compression plays here the crucial role. Time
series are special kind of data, for which choosing the right compression according
to the characteristics of the data series is essencial.
This paper is a research and presents a new approach to compression of time
series on the side of the GPU, using a planner to keep building the compression
scheme based on statistics of incoming data, in the incremental manner. The solution
compresses columnar data using lightweight and lossless compressions in CUDA
technology. The aim of the work is to create an optimizer with high performance in
terms of obtained compression ratio for data of variable characteristics.
The beginning of the document is a description of the problem, along with an
analysis of existing solutions and research, under the direction of compression using
SIMD (Single Instruction Multiple Data) architectures. Further it describes adopted
technology developed by NVIDIA and implemented algorithms of light compres-
sion for GPU. The following sections describe the implementation of the optimizer
algorithm, along with created environment and a program for parallel compression
of data columns. At the end are the results of experiments demonstrating the useful-
ness of such a solution and a description of further work that will be conducted in
the topic. This method is applicable in all types of columnar data warehouses, which
will be the subject further research.

### Requirements ###

* CUDA 7
* Cuda device with compute capability greater than 2.0
* gtest, google benchmark, log4cplus, Thrust
* boost
    * to install boost on Ubuntu type:

    ```bash
    sudo apt-get install libboost-all-dev
    ```


### Install ###

To install application run following instructions:
```bash
git clone https://dzitkowskik@bitbucket.org/dzitkowskik/gpustore.git
cd gpuStore
./install/install_all.sh
cp config.ini.example config.ini
cd sample_data
wget https://www.dropbox.com/s/3lea51f4jd2h2mz/openbookultraMM_N20130403_1_of_1
wget https://www.dropbox.com/s/neej12spsmx2fhv/info.log
cd ..
```
After that make sure that you have these env variables set:
```
export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### RUN TESTS VIA DOCKER ###

To run this project using docker, nvidia-docker is required. It can be acquired from their github repository: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
(see [installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation)).

TEMPORARILY:
To use a dockerfile you must have `id_rsa` file registered in bitbucket's repository
`git@bitbucket.org:dzitkowskik/gpustore.git` as well as bitbucket.org added to known_hosts.

AFTER THAT ONE CAN DO:
```bash
docker build . -t dzitkowskik/gpustore
nvidia-docker run dzitkowskik/gpustore /gpustore/gpuStore
```

### How do I get set up? ###

* `make test` - to build tests
* `make benchmark` - to build benchmarks
* `make release` - to build an application in release mode
* `make debug` - to build an application in debug mode
* `make run` - to run (tests/benchmarks/application)

### Program options ###

Example output program when compiled with `make release` compresses in parallel many columns of time series selected as input file.

* `--compress,-c`: perform compression
* `--decompress,-d`: perform decompression
* `--header,-h [path]`: sets the path to header file
* `--input,-i`: sets the path to input file
* `--output,-o`: sets the path to output file
* `--generate,-g [size]`: generate sample data of selected size
* `--padding,-p [value]`: sets the padding for binary file. When rows in binary format are aligned to some size, value is the difference between aligned size and real data size. For example when data consists of three 4 byte words (12 bytes) but are aligned to 16, then value should be equal to 4.


### Licence ###

The MIT License (MIT)

Copyright (c) 2015 Karol Dzitkowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
