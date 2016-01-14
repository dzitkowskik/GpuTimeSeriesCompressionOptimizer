/*
 *  compression_tree_unittest.cpp
 *
 *  Created on: 15/10/2015
 *      Author: Karol Dzitkowski
 */

#include "test/unittest_base.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_print.hpp"
#include "tree/compression_tree.hpp"
#include "tree/compression_node.hpp"
#include "optimizer/path_generator.hpp"
#include <boost/make_shared.hpp>
#include <gtest/gtest.h>

#include "compression/default_encoding_factory.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "util/transform/cuda_array_transform.hpp"

namespace ddj {

class CompressionTreeTest : public UnittestBase, public ::testing::WithParamInterface<int>
{
protected:
	virtual void SetUp() { _size = GetParam(); }

public:
	void TreeCompressionTest_Compress_Decompress(CompressionTree& compressionTree);
};


INSTANTIATE_TEST_CASE_P(
    CompressionTree_Test_Inst,
    CompressionTreeTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(CompressionTreeTest, SimpleOneNodeTree_Delta_Int_Compress_NoException)
{
	CompressionTree tree;
	auto node = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	ASSERT_TRUE( tree.AddNode(node, 0) );
	tree.Compress(MoveSharedCudaPtr<int, char>(GetIntRandomData()));
}

void CompressionTreeTest::TreeCompressionTest_Compress_Decompress(CompressionTree& compressionTree)
{
	auto randomData = GetIntRandomData();
    auto dataCopy = randomData->copy();
	auto compressed = compressionTree.Compress(MoveSharedCudaPtr<int, char>(dataCopy));

	CompressionTree decompressionTree;
	auto decompressed = decompressionTree.Decompress(compressed);

	auto expected = randomData;
	auto actual = MoveSharedCudaPtr<char, int>(decompressed);

	ASSERT_EQ(expected->size(), actual->size());
	EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );

//	printf("Uncompressed size = %d\n", d_int_random_data->size()*sizeof(int));
//	printf("Compressed size = %d\n", compressed->size());
}

TEST_P(CompressionTreeTest, SimpleOneNodeTree_Delta_Int_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto node = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto leafNode = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	node->AddChild(leafNode);
	ASSERT_TRUE( compressionTree.AddNode(node, 0) );
	TreeCompressionTest_Compress_Decompress(compressionTree);
}

TEST_P(CompressionTreeTest, SimpleOneNodeTree_Gfc_Float_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto node = boost::make_shared<CompressionNode>(boost::make_shared<GfcEncodingFactory>(DataType::d_float));
	ASSERT_TRUE( compressionTree.AddNode(node, 0) );
	compressionTree.Fix();
	TreeCompressionTest_Compress_Decompress(compressionTree);
}

TEST_P(CompressionTreeTest, SimpleTwoNodeTree_DeltaAndScale_Int_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto node = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto secondNode = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto leafNode = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	secondNode->AddChild(leafNode);
	node->AddChild(secondNode);
	ASSERT_TRUE( compressionTree.AddNode(node, 0) );
	TreeCompressionTest_Compress_Decompress(compressionTree);
}

//     DICT
//    /    \
// SCALE   DELTA
//   |
// DELTA
TEST_P(CompressionTreeTest, ComplexTree_Dict_Compress_Decompress)
{
    CompressionTree compressionTree;
    auto root = boost::make_shared<CompressionNode>(boost::make_shared<DictEncodingFactory>(DataType::d_int));
	auto right = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto left = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
    auto leftChild = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto noneLeft = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
    auto noneRight = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
    leftChild->AddChild(noneLeft);
    left->AddChild(leftChild);
    root->AddChild(left);
    right->AddChild(noneRight);
    root->AddChild(right);
	ASSERT_TRUE( compressionTree.AddNode(root, 0) );
	TreeCompressionTest_Compress_Decompress(compressionTree);
}

//         PATCH
//        /     \
//      DICT   DELTA
//     /   \
//  DELTA SCALE
TEST_P(CompressionTreeTest, ComplexTree_Patch_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::outside);
	pef->factor = 0.1;
	auto root = boost::make_shared<CompressionNode>(boost::shared_ptr<PatchEncodingFactory<int>>(pef));
	auto right = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto left = boost::make_shared<CompressionNode>(boost::make_shared<DictEncodingFactory>(DataType::d_int));
	auto leftChildLeft = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto leftChildRight = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto leaf1 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	auto leaf2 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	auto leaf3 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	leftChildLeft->AddChild(leaf1);
	leftChildRight->AddChild(leaf2);
	right->AddChild(leaf3);
	left->AddChild(leftChildLeft);
	left->AddChild(leftChildRight);
	root->AddChild(left);
	root->AddChild(right);

	ASSERT_TRUE( compressionTree.AddNode(root, 0) );
	TreeCompressionTest_Compress_Decompress(compressionTree);
}

//		SCALE
//		  |
//	    DELTA
//	      |
//	    PATCH
//	   /	 \
//	  AFL	NONE
TEST_P(CompressionTreeTest, ComplexTree_Delta_Scale_Patch_Afl_RealData_Time_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile());
	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::lower);
	pef->factor = 0.2;
	auto root = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto delta = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto patch = boost::make_shared<CompressionNode>(boost::shared_ptr<PatchEncodingFactory<int>>(pef));
	auto afl = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto leaf1 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	auto leaf2 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	afl->AddChild(leaf1);
	patch->AddChild(afl);
	patch->AddChild(leaf2);
	delta->AddChild(patch);
	root->AddChild(delta);

	ASSERT_TRUE( compressionTree.AddNode(root, 0) );

	auto compressed = compressionTree.Compress(CastSharedCudaPtr<int, char>(data));
	CompressionTree decompressionTree;
	auto decompressed = decompressionTree.Decompress(compressed);
	auto expected = data;
	auto actual = MoveSharedCudaPtr<char, int>(decompressed);

	//	printf("size before compression = %d\n", data->size()*sizeof(int));
	//	printf("size after comrpession = %d\n", compressed->size()*sizeof(char));

	ASSERT_EQ(expected->size(), actual->size());
	EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
}

//		SCALE
//		  |
//		 RLE
//		  |
//		DELTA
//		  |
//		UNIQUE
TEST_P(CompressionTreeTest, ComplexTree_Scale_Rle_Delta_Unique_RealData_Time_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile());
	auto scale = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto delta = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto rle = boost::make_shared<CompressionNode>(boost::make_shared<RleEncodingFactory>(DataType::d_int));
	auto unique = boost::make_shared<CompressionNode>(boost::make_shared<UniqueEncodingFactory>(DataType::d_int));
	auto leaf = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	unique->AddChild(leaf);
	delta->AddChild(unique);
	rle->AddChild(delta);
	scale->AddChild(delta);

	ASSERT_TRUE( compressionTree.AddNode(scale, 0) );

	auto compressed = compressionTree.Compress(CastSharedCudaPtr<int, char>(data));
	CompressionTree decompressionTree;
	auto decompressed = decompressionTree.Decompress(compressed);

//	HelperPrint::PrintSharedCudaPtr(data, "data");
//	HelperPrint::PrintSharedCudaPtr(CastSharedCudaPtr<char, int>(decompressed), "decompressed");

	auto expected = data;
	auto actual = MoveSharedCudaPtr<char, int>(decompressed);

//	printf("size before compression = %d\n", data->size()*sizeof(int));
//	printf("size after comrpession = %d\n", compressed->size()*sizeof(char));

	ASSERT_EQ(expected->size(), actual->size());
	EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
}

//		DELTA
//		  |
//		 AFL
TEST_P(CompressionTreeTest, SimpleTree_Delta_Afl_PredictSizeAfterCompression_RandomInt)
{
	CompressionTree compressionTree;
	auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile());
	auto delta = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto afl = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto leaf = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	afl->AddChild(leaf);
	delta->AddChild(afl);

	ASSERT_TRUE( compressionTree.AddNode(delta, 0) );

	auto compressed = compressionTree.Compress(CastSharedCudaPtr<int, char>(data));

	auto expected = compressed->size();
	auto actual = compressionTree.GetPredictedSizeAfterCompression(
			CastSharedCudaPtr<int, char>(data), DataType::d_int);

	EXPECT_EQ(expected, actual);
}

//		PATCH
//	   /     \
//	 AFL     AFL
TEST_P(CompressionTreeTest, DISABLED_SimpleTree_Patch_Afl_RandomInt_BigSize)
{
	CompressionTree compressionTree;
	auto data =
			CastSharedCudaPtr<int, char>(
				CudaArrayGenerator().GenerateRandomIntDeviceArray(1<<20, 10, 1000));

	auto patch = boost::make_shared<CompressionNode>(DefaultEncodingFactory().Get(EncodingType::patch, DataType::d_int));
	auto aflLeft = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto aflRight = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto leafLeft = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	auto leafRight = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	aflLeft->AddChild(leafLeft);
	aflRight->AddChild(leafRight);
	patch->AddChild(aflLeft);
	patch->AddChild(aflRight);
	ASSERT_TRUE( compressionTree.AddNode(patch, 0) );

	auto compressed = compressionTree.Compress(data);
	CompressionTree decompressionTree;
	auto decompressed = decompressionTree.Decompress(compressed);
	auto expected = CastSharedCudaPtr<char, int>(data);
	auto actual = MoveSharedCudaPtr<char, int>(decompressed);

	//	printf("size before compression = %d\n", data->size()*sizeof(int));
	//	printf("size after comrpession = %d\n", compressed->size()*sizeof(char));
	//	HelperPrint::PrintSharedCudaPtr(expected, "expected");
	//	HelperPrint::PrintSharedCudaPtr(actual, "actual");

	ASSERT_EQ(expected->size(), actual->size());
	EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
}

//		SCALE
//		  |
//	    DELTA
//	      |
//	    PATCH
//	   /	 \
//	  AFL	AFL
TEST_P(CompressionTreeTest, UpdateStatistics_ComplexTree)
{
	auto stats = CompressionStatistics::make_shared(5);
	CompressionTree compressionTree(stats);

	auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromTestFile());
	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::lower);
	pef->factor = 0.2;
	auto root = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto delta = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto patch = boost::make_shared<CompressionNode>(boost::shared_ptr<PatchEncodingFactory<int>>(pef));
	auto afl_left = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto afl_right = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));

	patch->AddChild(afl_left);
	patch->AddChild(afl_right);
	delta->AddChild(patch);
	root->AddChild(delta);

	ASSERT_TRUE( compressionTree.AddNode(root, 0) );
	compressionTree.Fix();

	auto compressed = compressionTree.Compress(CastSharedCudaPtr<int, char>(data));
	CompressionTree decompressionTree;
	auto decompressed = decompressionTree.Decompress(compressed);
	auto expected = data;
	auto actual = MoveSharedCudaPtr<char, int>(decompressed);

//	printf("size before compression = %lu\n", data->size()*sizeof(int));
//	printf("size after comrpession = %lu\n", compressed->size()*sizeof(char));
//	stats->Print();

	EXPECT_EQ(EncodingType::scale, stats->GetAny(0).type.first);
	EXPECT_EQ(EncodingType::delta, stats->GetAny(0).type.second);
	EXPECT_EQ(EncodingType::patch, stats->GetAny(6).type.first);
	EXPECT_EQ(EncodingType::afl, stats->GetAny(6).type.second);

	ASSERT_EQ(expected->size(), actual->size());
	EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );
}

//			PATCH
//			/	\
//		CONST	FloatToInt
//					|
//				   RLE
//				  /   \
//			  CONST   SCALE
//						|
//					   AFL
TEST_P(CompressionTreeTest, FakeDataPatternA_Float_GoodTree)
{
	Path path
	{
		EncodingType::patch,
			EncodingType::constData, EncodingType::none,
			EncodingType::floatToInt,
				EncodingType::rle,
					EncodingType::constData, EncodingType::none,
					EncodingType::scale, EncodingType::afl, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_float);

	auto floatData = GetFakeDataWithPatternA<float>(0, 1e2, 1.5f, -10.0f, 1e6f, 1e6);
	auto data = CastSharedCudaPtr<float, char>(floatData);

	auto compressedData = tree.Compress(data);

	tree.Print(tree.GetCompressionRatio());

	printf("Size before compression = %lu\n", data->size());
	printf("Size after compression = %lu\n", compressedData->size());
	printf("CompressionRatio = %f\n", (float)data->size()/(float)compressedData->size());

	auto decompressedData = tree.Decompress(compressedData);

	ASSERT_EQ(decompressedData->size(), data->size());
	EXPECT_TRUE( CompareDeviceArrays(decompressedData->get(), data->get(), data->size()) );
}

} /* namespace ddj */
