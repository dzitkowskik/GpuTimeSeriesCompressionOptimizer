//
// Created by Karol Dzitkowski on 11.10.15.
//

#include "file_comparer.h"
#include <iostream>
#include <fstream>
#include <cstring>

bool FileComparer::AreEqual(std::string pathOne, std::string pathTwo) {
    std::ifstream inputFileOne(pathOne, std::ios::binary);
    std::ifstream inputFileTwo(pathTwo, std::ios::binary);

    // GET SIZES OF FILES
    std::ifstream::pos_type sizeOfFileOne, sizeOfFileTwo;
    sizeOfFileOne = inputFileOne.seekg(0, std::ifstream::end).tellg();
    inputFileOne.seekg(0, std::ifstream::beg);
    sizeOfFileTwo = inputFileTwo.seekg(0, std::ifstream::end).tellg();
    inputFileTwo.seekg(0, std::ifstream::beg);

    // IF FILES WITH DIFFERENT SIZES THEN NOT EQUAL
    if(sizeOfFileOne != sizeOfFileTwo)
        return false;

    const size_t blockSize = 4096;
    size_t remaining = sizeOfFileOne;

    while(remaining)
    {
        char buffer1[blockSize], buffer2[blockSize];
        size_t size = std::min(blockSize, remaining);

        inputFileOne.read(buffer1, size);
        inputFileTwo.read(buffer2, size);

        if(0 != memcmp(buffer1, buffer2, size))
            return false;

        remaining -= size;
    }

    return true;
}
