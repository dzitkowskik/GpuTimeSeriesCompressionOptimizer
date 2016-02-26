/*
 *  optimizer_unittest.cpp
 *
 *  Created on: 12/11/2015
 *      Author: Karol Dzitkowski
 */

#include "test/unittest_base.hpp"
#include "optimizer/path_generator.hpp"
#include "optimizer/compression_optimizer.hpp"
#include "core/cuda_array.hpp"
#include "util/copy/cuda_array_copy.hpp"

#include "core/macros.h"
#include <gtest/gtest.h>

namespace ddj
{

class OptimizerTest : public UnittestBase
{
public:
	void TestCompressionOptimizerOnNyseCol(int colIdx);
	template<typename T>
	void CompressDecompressNPartsData(SharedCudaPtrVector<T> data);
};

TEST_F(OptimizerTest, PathGenerator_GeneratePaths)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, PathGenerator_GeneratePaths");
	auto paths = PathGenerator().GeneratePaths();
	EXPECT_GT(paths.size(), 0);
}

TEST_F(OptimizerTest, PathGenerator_GenerateTree_TestCompression)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, PathGenerator_GenerateTree_TestCompression");
	auto paths = PathGenerator().GeneratePaths();

	auto tsData = GetTsIntDataFromTestFile();
	auto data = CudaArrayTransform().Cast<time_t, int>(tsData);

	for(auto& path : paths)
	{
		auto tree = PathGenerator().GenerateTree(path, DataType::d_int);
		LOG4CPLUS_INFO(_logger, "Optimal tree: \n" << tree.ToString());

		auto compressedData = tree.Compress(CastSharedCudaPtr<int, char>(data));
		auto decompressedData = tree.Decompress(compressedData);

		auto expected = data;
		auto actual = MoveSharedCudaPtr<char, int>(decompressedData);

		LOG4CPLUS_INFO(_logger, std::endl
			<< "Size before compression:\t" << data->size()*sizeof(int) << std::endl
			<< "Size after compression:\t" << compressedData->size()
		);

		ASSERT_EQ(expected->size(), actual->size());
		EXPECT_TRUE( CudaArray().Compare(expected, actual) );
	}
}

TEST_F(OptimizerTest, PathGenerator_GenerateTree_TestCompressedSizePrediction)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, PathGenerator_GenerateTree_TestCompressedSizePrediction");
	auto paths = PathGenerator().GeneratePaths();

	for(auto& path : paths)
	{
		auto tree = PathGenerator().GenerateTree(path, DataType::d_int);

		auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile());
		auto compressedData = tree.Compress(CastSharedCudaPtr<int, char>(data));

		auto expected = compressedData->size();
		auto actual = tree.GetPredictedSizeAfterCompression(
				CastSharedCudaPtr<int, char>(data), DataType::d_int);
		EXPECT_EQ(expected, actual);
	}
}

TEST_F(OptimizerTest, CompressionOptimizer_OptimizeTree_TimeDataFromFile)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_OptimizeTree_TimeDataFromFile");
	auto data =
		CastSharedCudaPtr<int, char>(
			CudaArrayTransform().Cast<time_t, int>(
					GetTsIntDataFromTestFile()));

	auto optimalTree = CompressionOptimizer().OptimizeTree(data, DataType::d_int);
	LOG4CPLUS_INFO(_logger, "Optimal tree: \n" << optimalTree.ToString());

	auto compressed = optimalTree.Compress(data);
	auto decompressed = optimalTree.Decompress(compressed);

	LOG4CPLUS_INFO(_logger, std::endl
		<< "Size before compression:\t" << data->size() << std::endl
		<< "Size after compression:\t" << compressed->size()
	);

	EXPECT_LE( compressed->size(), data->size() );
	EXPECT_TRUE( CudaArray().Compare(data, decompressed) );
}

TEST_F(OptimizerTest, CompressionOptimizer_OptimizeTree_RandomInt)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_OptimizeTree_RandomInt");
	auto data = CastSharedCudaPtr<int, char>(GetIntRandomData());

	auto optimalTree = CompressionOptimizer().OptimizeTree(data, DataType::d_int);
	LOG4CPLUS_INFO(_logger, "Optimal tree: \n" << optimalTree.ToString());

	auto compressed = optimalTree.Compress(data);
	auto decompressed = optimalTree.Decompress(compressed);

	LOG4CPLUS_INFO(_logger, std::endl
		<< "Size before compression:\t" << data->size() << std::endl
		<< "Size after compression:\t" << compressed->size()
	);

	EXPECT_LE( compressed->size(), data->size() );
	EXPECT_TRUE( CudaArray().Compare(data, decompressed) );
}

TEST_F(OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_RandomInt_CompressByBestTree)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_RandomInt_CompressByBestTree");
	CompressionOptimizer optimizer;

	auto randomInt = GetIntRandomData();
	auto data = CastSharedCudaPtr<int, char>(randomInt);
	auto results = optimizer.FullStatisticsUpdate(data, EncodingType::none, DataType::d_int, 0);
	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });
	results[0].first.Fix();
	auto compressed = results[0].first.Compress(data);
	CompressionTree t;
	auto decompressed = t.Decompress(compressed);
	EXPECT_TRUE( CudaArray().Compare(randomInt, CastSharedCudaPtr<char, int>(decompressed)) );
}

TEST_F(OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_Statistics)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_Statistics");
	CompressionOptimizer optimizer;

	auto randomInt = GetIntRandomData();
	auto data = CastSharedCudaPtr<int, char>(randomInt);
	auto results = optimizer.FullStatisticsUpdate(data, EncodingType::none, DataType::d_int, 0);
	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });

	auto comprStats = CompressionStatistics::make_shared(5);

	for(auto& tree : results)
	{
		tree.first.Fix();
		tree.first.UpdateStatistics(comprStats);
		tree.first.SetStatistics(comprStats);
	}

	auto compressed = results[0].first.Compress(data);
	CompressionTree t;
	auto decompressed = t.Decompress(compressed);

	EXPECT_TRUE( CudaArray().Compare(randomInt, CastSharedCudaPtr<char, int>(decompressed)) );
}

void OptimizerTest::TestCompressionOptimizerOnNyseCol(int colIdx)
{
	CompressionOptimizer optimizer;

	auto ts = Get1GBNyseTimeSeries();
	size_t size = GetSize()*sizeof(int);
	auto data = CudaPtr<char>::make_shared(size);
	auto sourceTimeColumnRawData = ts->getColumn(colIdx).getData();
	data->fillFromHost(sourceTimeColumnRawData, size);

	auto compressedData =
			optimizer.CompressData(data, ts->getColumn(colIdx).getType());
	auto decompressedData =
			optimizer.GetOptimalTree()->GetTree().Decompress(compressedData);

	LOG4CPLUS_INFO(_logger, std::endl
		<< "Size before compression:\t" << data->size() << std::endl
		<< "Size after compression:\t" << compressedData->size()
	);

	EXPECT_TRUE( CudaArray().Compare(data, decompressedData) );
	ts.reset();

}

TEST_F(OptimizerTest, CompressionOptimizer_Nyse_Int_Compress)
{
	TestCompressionOptimizerOnNyseCol(0);
}

TEST_F(OptimizerTest, CompressionOptimizer_Nyse_Short_Compress)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_Nyse_Short_Compress");
	TestCompressionOptimizerOnNyseCol(1);
}

TEST_F(OptimizerTest, CompressionOptimizer_Nyse_Char_Compress)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_Nyse_Char_Compress");
	TestCompressionOptimizerOnNyseCol(3);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressData_SourceTimeNyse_Compress)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressData_SourceTimeNyse_Compress");
	int colIdx = 16; 	// column with index 16 is source time
	TestCompressionOptimizerOnNyseCol(colIdx);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressAndDecompress_TsFloatData_Twice)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressAndDecompress_TsFloatData_Twice");
	CompressionOptimizer optimizer;

	auto floatData = GetTsFloatDataFromTestFile();
	auto data = CastSharedCudaPtr<float, char>(floatData);

	optimizer.CompressData(data, DataType::d_float);
	auto compressedData = optimizer.CompressData(data, DataType::d_float);

	auto decompressedData =
			optimizer.GetOptimalTree()->GetTree().Decompress(compressedData);

	LOG4CPLUS_INFO(_logger, std::endl
		<< "Size before compression:\t" << data->size() << std::endl
		<< "Size after compression:\t" << compressedData->size()
	);

	EXPECT_TRUE( CudaArray().Compare(data, decompressedData) );
}

template<typename T>
void OptimizerTest::CompressDecompressNPartsData(SharedCudaPtrVector<T> data)
{
	CompressionOptimizer optimizer;
	SharedCudaPtrVector<char> compressedDataParts;
	int N = data.size();

	size_t actualSizeSum = 0;
	size_t compressedSizeSum = 0;

	// Compress
	for(int i = 0; i < N; i++)
	{
		auto dataToCompress = CastSharedCudaPtr<T, char>(data[i]);
		actualSizeSum += dataToCompress->size();
		auto compressedPart = optimizer.CompressData(dataToCompress, GetDataType<T>());
		compressedSizeSum += compressedPart->size();
		compressedDataParts.push_back(compressedPart);

		CompressionTree& optimalTree = optimizer.GetOptimalTree()->GetTree();
		LOG4CPLUS_DEBUG_FMT(
			_logger,
			"%s",
			optimizer.GetStatistics()->ToStringShort().c_str()
		);

		LOG4CPLUS_DEBUG_FMT(_logger, "COMPRESSION RATIO = %f", (float)dataToCompress->size()/compressedPart->size());
	}

	// Check
	CompressionTree decompressionTree;
	for(int i = 0; i < N; i++)
	{
		// Decompress
		auto decompressedData = CastSharedCudaPtr<char, T>(
				decompressionTree.Decompress(compressedDataParts[i]));

		EXPECT_TRUE( CudaArray().Compare(data[i], decompressedData) );
	}

	LOG4CPLUS_INFO_FMT(_logger, "TOTAL COMPRESSION RATIO = %f", (float)actualSizeSum/compressedSizeSum);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeFloatData_PatternA_5Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeFloatData_PatternA_5Parts");
	int N = 5;
	SharedCudaPtrVector<float> dataParts;
	for(int i = 0; i < N; i++)
		dataParts.push_back(GetFakeDataWithPatternA<float>(i, 1e2, 1.5f, -10.0f, 1e6f, GetSize()));

	CompressDecompressNPartsData(dataParts);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeIntData_PatternA_25Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeIntData_PatternA_25Parts");
	int N = 25;
	SharedCudaPtrVector<int> dataParts;
	for(int i = 0; i < N; i++)
		dataParts.push_back(GetFakeDataWithPatternA<int>(i, 10, 1, -100, 1e6, GetSize()));

	CompressDecompressNPartsData(dataParts);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeFloatData_PatternB_9Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeFloatData_PatternB_9Parts");
	int N = 9;
	int SIZE = GetSize();
	SharedCudaPtrVector<float> dataParts;
	for(int i = 0; i < N; i++)
		dataParts.push_back(GetFakeDataWithPatternB<float>(i, 3*SIZE, -1e4f, 1e4f, SIZE/2));

	CompressDecompressNPartsData(dataParts);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressData_3_RandomDataParts_Int_Compress)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressData_3_RandomDataParts_Int_Compress");
	int N = 3;
	SharedCudaPtrVector<int> dataParts;
	for(int i = 0; i < N; i++)
		dataParts.push_back(GetIntRandomData(10, 10000));

	CompressDecompressNPartsData(dataParts);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressData_3_TimeRealData_Compress)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressData_3_TimeRealData_Compress");
	int N = 3;
	SharedCudaPtrVector<int> dataParts;
	for(int i = 0; i < N; i++)
		dataParts.push_back(
				CudaArrayTransform().Cast<time_t, int>(GetNextTsIntDataFromTestFile()));

	CompressDecompressNPartsData(dataParts);
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeIntData_PatternAB_25Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeIntData_PatternA_25Parts");
	int N = 25;
	int SIZE = GetSize();
	SharedCudaPtrVector<int> dataParts;
	for(int i = 0; i < N; i++)
	{
		dataParts.push_back(GetFakeDataWithPatternA<int>(i, 1e2, 1, -100, 1e6, GetSize()));
		dataParts.push_back(GetFakeDataWithPatternB<int>(i, 3*SIZE, -1e4, 1e4, SIZE/2));
	}

	CompressDecompressNPartsData(dataParts);
}

TEST_F(OptimizerTest, DISABLED_Results_FakeTimeAndPatternA_50Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_FakeTimeAndPatternA_50Parts");
	int N = 50;
	int SIZE = GetSize();
	SharedCudaPtrVector<time_t> dataParts;
	bool x = true;
	for(int i = 0; i < N; i++)
	{
		if(x)
			dataParts.push_back(GetFakeDataWithPatternA<time_t>(i, 1e2, 1, -100, 1e6, SIZE));
		else
			dataParts.push_back(GetFakeDataForTime());
		if(i%6 == 5) x = !x;
	}

	CompressDecompressNPartsData(dataParts);

	LOG4CPLUS_INFO(_logger, "compression with best tree");
	Path path
	{
		EncodingType::patch,
			EncodingType::rle,
				EncodingType::constData, EncodingType::none,
				EncodingType::delta,
					EncodingType::afl, EncodingType::none,
			EncodingType::constData, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_time);
	auto data = CudaArrayCopy().Concatenate(dataParts);
	auto compressed = tree.Compress(CastSharedCudaPtr<time_t, char>(data));
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE: %s", tree.ToString().c_str());
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE COMPRESSION RATIO = %f", (float)data->size()/compressed->size());
}

TEST_F(OptimizerTest, DISABLED_Results_PatternAB_50Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_PatternAB_50Parts");
	int N = 50;
	int SIZE = GetSize();
	SharedCudaPtrVector<int> dataParts;
	bool x = true;
	for(int i = 0; i < N; i++)
	{
		if(x)
			dataParts.push_back(GetFakeDataWithPatternA<int>(i, 10, 1, -100, 1e3, SIZE));
		else
			dataParts.push_back(GetFakeDataWithPatternB<int>(i, 3*SIZE, 1e4, 1e7, SIZE/2));
		if(i%6 == 5) x = !x;
	}

	CompressDecompressNPartsData(dataParts);

	LOG4CPLUS_INFO(_logger, "compression with best tree");
	Path path
	{
		EncodingType::dict,
			EncodingType::scale,
				EncodingType::dict, EncodingType::none, EncodingType::none,
			EncodingType::rle,
				EncodingType::constData, EncodingType::none,
				EncodingType::delta,
					EncodingType::afl, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_int);
	auto data = CudaArrayCopy().Concatenate(dataParts);
	auto compressed = tree.Compress(CastSharedCudaPtr<int, char>(data));
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE: %s", tree.ToString().c_str());
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE COMPRESSION RATIO = %f", (float)data->size()/compressed->size());
}

TEST_F(OptimizerTest, DISABLED_Results_PatternsABMaxPrec_Float_75Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, Results_Optimizer_vs_BestScheme");
	int N = 5;
	int SIZE = GetSize();
	SharedCudaPtrVector<float> dataParts;
	bool x = true;
	for(int k = 0; k < 5; k++)
	{
		for(int i = 0; i < N; i++)
			dataParts.push_back(GetFakeDataWithPatternA<float>(i, 1e2, 1, -100, 1e6, SIZE));
		for(int i = 0; i < N; i++)
			dataParts.push_back(GetFakeDataWithPatternB<float>(i, 3*SIZE, -1e4, 1e4, SIZE/2));
		for(int i = 0; i < N; i++)
			dataParts.push_back(GetFloatRandomDataWithMaxPrecision(2));
	}

	CompressDecompressNPartsData(dataParts);

	LOG4CPLUS_INFO(_logger, "compression with best tree");
	Path path
	{
		EncodingType::rle,
			EncodingType::patch,
				EncodingType::constData, EncodingType::none,
				EncodingType::constData, EncodingType::none,
			EncodingType::patch, EncodingType::none,
				EncodingType::constData, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_float);
	auto data = CudaArrayCopy().Concatenate(dataParts);
	auto compressed = tree.Compress(CastSharedCudaPtr<float, char>(data));
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE: %s", tree.ToString().c_str());
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE COMPRESSION RATIO = %f", (float)data->size()/compressed->size());
}

TEST_F(OptimizerTest, DISABLED_Results_PatternsTsIntConsecutiveRandom_Float_75Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, Results_Optimizer_vs_BestScheme");
	int N = 5;
	int SIZE = GetSize();
	SharedCudaPtrVector<int> dataParts;
	bool x = true;
	for(int k = 0; k < 5; k++)
	{
		for(int i = 0; i < N; i++)
			dataParts.push_back(CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile()));
		for(int i = 0; i < N; i++)
			dataParts.push_back(GetIntConsecutiveData());
		for(int i = 0; i < N; i++)
			dataParts.push_back(GetIntRandomData());
	}

	CompressDecompressNPartsData(dataParts);

	LOG4CPLUS_INFO(_logger, "compression with best tree");
	Path path
	{
		EncodingType::rle,
			EncodingType::scale,
				EncodingType::afl, EncodingType::none,
			EncodingType::delta,
				EncodingType::dict,
					EncodingType::afl, EncodingType::none,
					EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_int);
	auto data = CudaArrayCopy().Concatenate(dataParts);
	auto compressed = tree.Compress(CastSharedCudaPtr<int, char>(data));
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE: %s", tree.ToString().c_str());
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE COMPRESSION RATIO = %f", (float)data->size()/compressed->size());
}

TEST_F(OptimizerTest, Results_PatternB_Int_75Parts)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_CompressAndDecompress_FakeIntData_PatternB_25Parts");
	int N = 75;
	int SIZE = GetSize();
	SharedCudaPtrVector<int> dataParts;
	for(int i = 0; i < N; i++)
		dataParts.push_back(GetFakeDataWithPatternB<int>(i, 3*SIZE, -1e4, 1e4, SIZE/2));

	CompressDecompressNPartsData(dataParts);

	LOG4CPLUS_INFO(_logger, "compression with best tree");
	Path path
	{
		EncodingType::scale,
			EncodingType::patch,
				EncodingType::afl, EncodingType::none,
				EncodingType::scale,
					EncodingType::afl, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_int);
	auto data = CudaArrayCopy().Concatenate(dataParts);
	auto compressed = tree.Compress(CastSharedCudaPtr<int, char>(data));
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE: %s", tree.ToString().c_str());
	LOG4CPLUS_INFO_FMT(_logger, "BEST TREE COMPRESSION RATIO = %f", (float)data->size()/compressed->size());
}


TEST_F(OptimizerTest, DISABLED_Results_TreeCount)
{
	LOG4CPLUS_INFO(_logger, "OptimizerTest, CompressionOptimizer_TreeCount");

	LOG4CPLUS_INFO(_logger, "NYSE time");
	CompressionOptimizer().CompressData(CastSharedCudaPtr<time_t, char>(GetTsIntDataFromTestFile()), DataType::d_time);
	LOG4CPLUS_INFO(_logger, "Float max precision 3");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<float, char>(GetFloatRandomDataWithMaxPrecision(3)), DataType::d_float);
	LOG4CPLUS_INFO(_logger, "Random Int from 10, 10000");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<int, char>(GetIntRandomData(10, 10000)), DataType::d_int);
	LOG4CPLUS_INFO(_logger, "Int consecutive");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<int, char>(GetIntConsecutiveData()), DataType::d_int);
	LOG4CPLUS_INFO(_logger, "Int Pattern A");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<int, char>(GetFakeDataWithPatternA<int>()), DataType::d_int);
	LOG4CPLUS_INFO(_logger, "Float Pattern A");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<float, char>(GetFakeDataWithPatternA<float>()), DataType::d_float);
	LOG4CPLUS_INFO(_logger, "Int Pattern B");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<int, char>(GetFakeDataWithPatternB<int>()), DataType::d_int);
	LOG4CPLUS_INFO(_logger, "Float Pattern B");
	CompressionOptimizer().CompressData(
			CastSharedCudaPtr<float, char>(GetFakeDataWithPatternB<float>()), DataType::d_float);

}

class OptimizerResultsForOutliers : public OptimizerTest, public ::testing::WithParamInterface<double> {};

INSTANTIATE_TEST_CASE_P(
	OptimizerResultsForOutliers_Inst,
	OptimizerResultsForOutliers,
    ::testing::Range(0.0, 1.0, 0.01));

TEST_P(OptimizerResultsForOutliers, DISABLED_CompressDataWithOptimizer_20parts)
{
	LOG4CPLUS_INFO_FMT(_logger, "OptimizerTestForOutliers.CompressDataWithOptimizer_50parts(param=%f)", GetParam());
	int N = 20;
	size_t SIZE = 1e5;
	double outProb = GetParam();
	SharedCudaPtrVector<int> dataParts;
	for(int i = 0; i < N; i++)
	{
		auto data = _generator.GetFakeDataWithOutliers<int>(i, 20, 1, 0, 1e6, outProb, SIZE);
		dataParts.push_back(data);
	}

	CompressDecompressNPartsData(dataParts);
}

} /* namespace ddj */
