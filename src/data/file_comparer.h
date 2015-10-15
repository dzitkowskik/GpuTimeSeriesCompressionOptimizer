//
// Created by Karol Dzitkowski on 11.10.15.
//

#ifndef TIME_SERIES_DATA_READER_FILECOMPARER_H
#define TIME_SERIES_DATA_READER_FILECOMPARER_H

#include <string>

class FileComparer
{
public:
    bool AreEqual(std::string pathOne, std::string pathTwo);
};


#endif //TIME_SERIES_DATA_READER_FILECOMPARER_H
