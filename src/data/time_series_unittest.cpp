#include "data/time_series_reader.hpp"
#include <gtest/gtest.h>

//#define DDJ_USE_BOOST 1

TimeSeries GenerateFakeTimeSeries(std::string name, int N)
{
    TimeSeries result(name);
    result.init(std::vector<DataType> {DataType::d_time, DataType::d_float});
    result.setColumnNames(std::vector<std::string> {"time", "value"});

    size_t size = result.getRecordSize();
    char* data = new char[size];

    for(time_t i = 0; i < N; i++)
    {
        float value = (float)i * (i % 3);
        memcpy(data, &i, sizeof(time_t));
        memcpy(data+sizeof(time_t), &value, sizeof(float));
        result.addRecord(data);
    }

    return result;
}

TEST(TimeSeries, ReadWrite_CSV_Data_ToFile)
{ 
    auto testFile = File::GetTempFile();
    TimeSeries fake = GenerateFakeTimeSeries("fake", 1000);
    CSVFileDefinition fileDefinition;
    fileDefinition.Header = std::vector<std::string> {"time", "value"};
    fileDefinition.Columns = std::vector<DataType> { DataType::d_time, DataType::d_float };
    TimeSeriesReaderCSV reader(fileDefinition);
    reader.Write(testFile, fake);
    auto result = reader.Read(testFile);

    EXPECT_TRUE(result->compare(fake));
    testFile.Delete();
}

TEST(TimeSeries, ReadWrite_CSV_Data_FromFile)
{
    File realDataFile("../../sample_data/info.log");
    File testFile = File::GetTempFile();
    File testFile2 = File::GetTempFile();
    CSVFileDefinition fileDefinition;
    fileDefinition.Columns = std::vector<DataType> {
            DataType::d_time,
            DataType::d_float,
            DataType::d_float,
            DataType::d_float
    };
    auto data = TimeSeriesReaderCSV(fileDefinition).Read(realDataFile);
    TimeSeriesReaderCSV(fileDefinition).Write(testFile, *data);
    auto data2 = TimeSeriesReaderCSV(fileDefinition).Read(testFile);
    TimeSeriesReaderCSV(fileDefinition).Write(testFile2, *data2);
    EXPECT_TRUE(testFile.Compare(testFile2));
    testFile.Delete();
    testFile2.Delete();
}

TEST(TimeSeries, ReadWrite_Binary_Data_ToFile)
{
    auto testFile = File::GetTempFile();

    TimeSeries fake = GenerateFakeTimeSeries("fake", 1000);

    BinaryFileDefinition fileDefinition;
    fileDefinition.Header = std::vector<std::string> {"time", "value"};
    fileDefinition.Columns = std::vector<DataType> { DataType::d_time, DataType::d_float };
    TimeSeriesReaderBinary(fileDefinition).Write(testFile, fake);
    auto result = TimeSeriesReaderBinary(fileDefinition).Read(testFile);
    EXPECT_TRUE(result->compare(fake));
    testFile.Delete();
}

TEST(TimeSeries, Read_CSV_Data_InMultipleParts_FromFile_CheckWithOnePartRead)
{
	File realDataFile("../test/data/info.log");
	CSVFileDefinition fileDefinition;
    fileDefinition.Columns = std::vector<DataType> {
            DataType::d_time,
            DataType::d_float,
            DataType::d_float,
            DataType::d_float
    };

	// read whole data 1000 rows
	int allRows = 1000;
	auto wholeData = TimeSeriesReaderCSV(fileDefinition).Read(realDataFile, allRows);

	// compare with reading 5 times 200 rows
	int partNo = 5;
	int partRowCnt = allRows / partNo;
    TimeSeriesReaderCSV reader(fileDefinition);
	for(int i = 0; i < partNo; i++)
	{
		auto partData = reader.Read(realDataFile, partRowCnt);
        EXPECT_EQ(partData->getRecordAsStrings(0)[0], wholeData->getRecordAsStrings(partRowCnt*i)[0]);
	}
}

TEST(TimeSeries, Read_Header_File)
{
	File headerFile("../test/data/info.header");
	auto fileDef = TimeSeriesReader::ReadFileDefinition(headerFile);

    // check
    EXPECT_EQ(DataType::d_time, fileDef.Columns[0]);
    EXPECT_EQ(DataType::d_float, fileDef.Columns[1]);
    EXPECT_EQ(DataType::d_float, fileDef.Columns[2]);
    EXPECT_EQ(DataType::d_float, fileDef.Columns[3]);
    EXPECT_EQ("timestamp", fileDef.Header[0]);
    EXPECT_EQ("GPU TEMP", fileDef.Header[3]);
}

TEST(TimeSeries, Read_Binary_With_Header_File)
{
    File headerFile("../test/data/info.header");
    auto fileDef = TimeSeriesReader::ReadFileDefinition(headerFile);

    File realDataFile("../test/data/info.inf");
    auto result = TimeSeriesReaderBinary(fileDef, 4).Read(realDataFile);

//    result->print(5);
}

TEST(TimeSeries, Read_CSV_With_Header_File_Write_And_Compare)
{
    File testFile = File::GetTempFile();
    File headerFile("../test/data/info.header");
    auto fileDef = TimeSeriesReader::ReadFileDefinition(headerFile);

    File realDataFile("../test/data/info.log");
    auto result = TimeSeriesReaderCSV(fileDef).Read(realDataFile);

    TimeSeriesReaderCSV(fileDef).Write(testFile, *result);

    EXPECT_TRUE( realDataFile.Compare(testFile) );
}

TEST(TimeSeries, Read_Binary_With_Header_File_Write_And_Compare)
{
    File testFile = File::GetTempFile();
    File headerFile("../test/data/info.header");
    auto fileDef = TimeSeriesReader::ReadFileDefinition(headerFile);

    File realDataFile("../test/data/info.inf");
    auto result = TimeSeriesReaderBinary(fileDef, 4).Read(realDataFile);

//    result->print(100000);

    TimeSeriesReaderBinary(fileDef, 4).Write(testFile, *result);

    auto another = TimeSeriesReaderBinary(fileDef, 4).Read(testFile);

    EXPECT_TRUE( another->compare(*result) );
}

TEST(TimeSeries, Read_NYSE_CSV_With_Header_File)
{
    File testFile = File::GetTempFile();
    File headerFile("../test/data/nyse.header");
    auto fileDef = TimeSeriesReader::ReadFileDefinition(headerFile);

    File realDataFile("../test/data/nyse.csv");
    auto result = TimeSeriesReaderCSV(fileDef).Read(realDataFile);

    TimeSeriesReaderCSV(fileDef).Write(testFile, *result);

    EXPECT_TRUE( realDataFile.Compare(testFile) );
}
