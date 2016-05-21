#!/bin/bash

#install required packages
sudo apt-get install cmake -y

mkdir celeroInstallTemp
cd celeroInstallTemp
git clone https://github.com/DigitalInBlue/Celero.git
cd Celero
cmake .
sudo make install
cd ../..
rm -rf celeroInstallTemp/
