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

	for(auto& path : paths)
	{
		auto tree = PathGenerator().GenerateTree(path, DataType::d_int);

		auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile());
		auto compressedData = tree.Compress(CastSharedCudaPtr<int, char>(data));
		auto decompressedData = tree.Decompress(compressedData);

		auto expected = data;
		auto actual = MoveSharedCudaPtr<char, int>(decompressedData);

		//	printf("size before compression = %d\n", data->size()*sizeof(int));
		//	printf("size after comrpession = %d\n", compressed->size()*sizeof(char));

		ASSERT_EQ(expected->size(), actual->size());
		EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
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
	auto data =
		CastSharedCudaPtr<int, char>(
			CudaArrayGenerator().GenerateRandomIntDeviceArray(1<<20, 10, 1000));

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

}
