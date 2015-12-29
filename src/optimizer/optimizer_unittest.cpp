/*
 *  optimizer_unittest.cpp
 *
 *  Created on: 12/11/2015
 *      Author: Karol Dzitkowski
 */

#include "test/unittest_base.hpp"
#include "optimizer/path_generator.hpp"
#include "optimizer/compression_optimizer.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_print.hpp"
#include "core/macros.h"
#include <gtest/gtest.h>

namespace ddj
{

class OptimizerTest : public UnittestBase {};

TEST_F(OptimizerTest, PathGenerator_GeneratePaths)
{
	auto paths = PathGenerator().GeneratePaths();
	EXPECT_GT(paths.size(), 0);
}

TEST_F(OptimizerTest, PathGenerator_GenerateTree_TestCompression)
{
	auto paths = PathGenerator().GeneratePaths();

	auto tsData = GetTsIntDataFromTestFile();
	auto data = CudaArrayTransform().Cast<time_t, int>(tsData);
//	HelperPrint::PrintSharedCudaPtr(data, "data");

	for(auto& path : paths)
	{
		auto tree = PathGenerator().GenerateTree(path, DataType::d_int);
		tree.Print();

		auto compressedData = tree.Compress(CastSharedCudaPtr<int, char>(data));
		auto decompressedData = tree.Decompress(compressedData);

		auto expected = data;
		auto actual = MoveSharedCudaPtr<char, int>(decompressedData);

		//	printf("size before compression = %d\n", data->size()*sizeof(int));
		//	printf("size after comrpession = %d\n", compressed->size()*sizeof(char));

		ASSERT_EQ(expected->size(), actual->size());
		EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
		CUDA_ASSERT_RETURN(cudaGetLastError());
	}
}

TEST_F(OptimizerTest, PathGenerator_GenerateTree_TestCompressedSizePrediction)
{
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
	auto data =
		CastSharedCudaPtr<int, char>(
			CudaArrayTransform().Cast<time_t, int>(
					GetTsIntDataFromTestFile()));

	auto optimalTree = CompressionOptimizer().OptimizeTree(data, DataType::d_int);
	printf("Optimal tree: \n");
	optimalTree.Print();

	auto compressed = optimalTree.Compress(data);
	auto decompressed = optimalTree.Decompress(compressed);

	printf("Size before compression: %d\n", (int)data->size());
	printf("Size after compression: %d\n", (int)compressed->size());

	EXPECT_LE( compressed->size(), data->size() );
	EXPECT_TRUE( CompareDeviceArrays(data->get(), decompressed->get(), data->size()) );
}

TEST_F(OptimizerTest, CompressionOptimizer_OptimizeTree_RandomInt)
{
	auto data = CastSharedCudaPtr<int, char>(GetIntRandomData());

	auto optimalTree = CompressionOptimizer().OptimizeTree(data, DataType::d_int);
	printf("Optimal tree: \n");
	optimalTree.Print();

	auto compressed = optimalTree.Compress(data);
	auto decompressed = optimalTree.Decompress(compressed);

	printf("Size before compression: %d\n", (int)data->size());
	printf("Size after compression: %d\n", (int)compressed->size());

	EXPECT_LE( compressed->size(), data->size() );
	EXPECT_TRUE( CompareDeviceArrays(data->get(), decompressed->get(), data->size()) );
}

TEST_F(OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_RandomInt_CompressByBestTree)
{
	CompressionOptimizer optimizer;

	auto randomInt = GetIntRandomData();
	auto data = CastSharedCudaPtr<int, char>(randomInt);
	auto stats = CudaArrayStatistics().GenerateStatistics(data, DataType::d_int);
	auto results = optimizer.FullStatisticsUpdate(data, EncodingType::none, DataType::d_int, stats, 0);
	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });
	results[0].first.Fix();
//	results[0].first.Print();
	auto compressed = results[0].first.Compress(data);
	CompressionTree t;
	auto decompressed = t.Decompress(compressed);
	EXPECT_TRUE( CompareDeviceArrays(randomInt->get(), CastSharedCudaPtr<char, int>(decompressed)->get(), GetSize()));
//	printf("best tree statistic: %lu\n", results[0].second);
//	printf("compressed data size: %lu\n", compressed->size());
}

TEST_F(OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_Statistics)
{
	CompressionOptimizer optimizer;

	auto randomInt = GetIntRandomData();
	auto data = CastSharedCudaPtr<int, char>(randomInt);
	auto stats = CudaArrayStatistics().GenerateStatistics(data, DataType::d_int);
	auto results = optimizer.FullStatisticsUpdate(data, EncodingType::none, DataType::d_int, stats, 0);
//	printf("number of trees = %d\n", results.size());
	std::sort(results.begin(), results.end(), [&](PossibleTree A, PossibleTree B){ return A.second < B.second; });

	auto comprStats = CompressionStatistics::make_shared(5);

	for(auto& tree : results)
	{
		tree.first.Fix();
		tree.first.UpdateStatistics(comprStats);
		tree.first.SetStatistics(comprStats);
	}

//	printf("Best tree: \n");
//	results[0].first.Print(results[0].second);
//	printf("Before compression\n");
//	comprStats->PrintShort();

	auto compressed = results[0].first.Compress(data);
	CompressionTree t;
	auto decompressed = t.Decompress(compressed);

//	printf("After compression\n");
//	comprStats->PrintShort();

	EXPECT_TRUE( CompareDeviceArrays(randomInt->get(), CastSharedCudaPtr<char, int>(decompressed)->get(), GetSize()));
//	printf("size of data before compression: %lu\n", data->size());
//	printf("best tree statistic: %lu\n", results[0].second);
//	printf("compressed data size: %lu\n", compressed->size());
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressData_3_RandomDataParts_Int_Compress)
{
	int N = 3;
	CompressionOptimizer optimizer;
	SharedCudaPtrVector<int> randomIntDataParts;
	SharedCudaPtrVector<char> compressedDataParts;
	for(int i = 0; i < N; i++)
		randomIntDataParts.push_back(GetIntRandomData(10, 10000));

	// Compress
	for(int i = 0; i < N; i++)
	{
		auto compressedPart = optimizer.CompressData(
			CastSharedCudaPtr<int, char>(randomIntDataParts[i]), DataType::d_int);
		compressedDataParts.push_back(compressedPart);

//		optimizer.GetOptimalTree()->GetTree().Print(compressedPart->size());
//		printf("Compression ratio = %lf\n", optimizer.GetOptimalTree()->GetTree().GetCompressionRatio());
//		optimizer.GetStatistics()->PrintShort();
	}

	// Check
	CompressionTree decompressionTree;
	for(int i = 0; i < N; i++)
	{
		// Decompress
		auto decompressedData = CastSharedCudaPtr<char, int>(
				decompressionTree.Decompress(compressedDataParts[i]));
		EXPECT_TRUE( CompareDeviceArrays(
				randomIntDataParts[i]->get(),
				decompressedData->get(),
				randomIntDataParts[i]->size()) );
	}
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressData_3_TimeRealData_Compress)
{
	int N =3;
	CompressionOptimizer optimizer;
	SharedCudaPtrVector<int> timeDataParts;
	SharedCudaPtrVector<char> compressedDataParts;
	for(int i = 0; i < N; i++)
		timeDataParts.push_back(
				CudaArrayTransform().Cast<time_t, int>(GetNextTsIntDataFromTestFile()));

	// Compress
	for(int i = 0; i < N; i++)
	{
		auto compressedPart = optimizer.CompressData(
			CastSharedCudaPtr<int, char>(timeDataParts[i]), DataType::d_int);
		compressedDataParts.push_back(compressedPart);

		optimizer.GetOptimalTree()->GetTree().Print(compressedPart->size());
		printf("Compression ratio = %lf\n", optimizer.GetOptimalTree()->GetTree().GetCompressionRatio());
		optimizer.GetStatistics()->PrintShort();
	}

	// Check
	CompressionTree decompressionTree;
	for(int i = 0; i < N; i++)
	{
		// Decompress
		auto decompressedData = CastSharedCudaPtr<char, int>(
				decompressionTree.Decompress(compressedDataParts[i]));
		EXPECT_TRUE( CompareDeviceArrays(
				timeDataParts[i]->get(),
				decompressedData->get(),
				timeDataParts[i]->size()) );
	}
}

TEST_F(OptimizerTest, CompressionOptimizer_CompressData_SourceTimeNyse_Compress)
{
	CompressionOptimizer optimizer;

	auto ts = Get1GBNyseTimeSeries();
	int colIdx = 16; 	// column with index 16 is source time
	size_t size = GetSize()*sizeof(int);
	auto data = CudaPtr<char>::make_shared(size);
	auto sourceTimeColumnRawData = ts->getColumn(colIdx).getData();
	data->fillFromHost(sourceTimeColumnRawData, size);

	auto compressedData =
			optimizer.CompressData(data, ts->getColumn(colIdx).getType());
	auto decompressedData =
			optimizer.GetOptimalTree()->GetTree().Decompress(compressedData);

	printf("Size before compression: %lu\n", size);
	printf("Size after compression: %lu\n", compressedData->size());

	EXPECT_TRUE( CompareDeviceArrays(data->get(), decompressedData->get(), data->size()) );
	ts.reset();
}

}
