//
// Created by Karol Dzitkowski on 19.10.15.
//

#ifndef TIME_SERIES_DATA_READER_TS_FILE_DEFINITION_H
#define TIME_SERIES_DATA_READER_TS_FILE_DEFINITION_H

#include <vector>

struct FileDefinition
{
    std::vector<std::string> Header;
    std::vector<DataType> Columns;
    std::vector<int> Decimals;
};

struct CSVFileDefinition : FileDefinition
{
    bool HasHeader = true;
    std::string Separator = ",";

    CSVFileDefinition(){}
    ~CSVFileDefinition(){}
    CSVFileDefinition(const CSVFileDefinition&) = default;
    CSVFileDefinition(const FileDefinition& def) : FileDefinition(def) {};
};

struct BinaryFileDefinition : FileDefinition
{
    BinaryFileDefinition(){}
    ~BinaryFileDefinition(){}
    BinaryFileDefinition(const BinaryFileDefinition&) = default;
    BinaryFileDefinition(const FileDefinition& def) : FileDefinition(def) {};
};

#endif //TIME_SERIES_DATA_READER_TS_FILE_DEFINITION_H
