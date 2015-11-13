/*
 * optimizer_unittest.cpp
 *
 *  Created on: 12 lis 2015
 *      Author: Karol Dzitkowski
 */

#include "test/unittest_base.hpp"
#include "optimizer/path_generator.hpp"
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

}
