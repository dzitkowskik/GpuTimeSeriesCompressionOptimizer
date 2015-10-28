/*
 *  compression_tree_unittest.cpp
 *
 *  Created on: 15/10/2015
 *      Author: Karol Dzitkowski
 */

#include "test/unittest_base.hpp"
#include "helpers/helper_comparison.cuh"
#include "tree/compression_tree.hpp"
#include <boost/make_shared.hpp>
#include <gtest/gtest.h>

#include "compression/delta/delta_encoding.hpp"
#include "compression/dict/dict_encoding.hpp"
#include "compression/none/none_encoding.hpp"
#include "compression/patch/patch_encoding.hpp"
#include "compression/rle/rle_encoding.hpp"
#include "compression/scale/scale_encoding.hpp"
#include "compression/unique/unique_encoding.hpp"

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
//TEST_P(CompressionTreeTest, ComplexTree_Patch_Compress_Decompress)
//{
//	CompressionTree compressionTree;
//	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::outside);
//	pef->min = 100;
//	pef->max = 1000;
//	pef->factor = 0.1;
//	auto root = boost::make_shared<CompressionNode>(boost::shared_ptr<PatchEncodingFactory<int>>(pef));
//	auto right = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
//	auto left = boost::make_shared<CompressionNode>(boost::make_shared<DictEncodingFactory>(DataType::d_int));
//	auto leftChildLeft = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
//	auto leftChildRight = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
//	auto leaf = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
//
//	leftChildLeft->AddChild(leaf);
//	leftChildRight->AddChild(leaf);
//	right->AddChild(leaf);
//	left->AddChild(leftChildLeft);
//	left->AddChild(leftChildRight);
//	root->AddChild(left);
//	root->AddChild(right);
//
//	ASSERT_TRUE( compressionTree.AddNode(root, 0) );
//	TreeCompressionTest_Compress_Decompress(compressionTree);
//}
//
//--gtest_repeat=10 --gtest_filter=CompressionTree_Test_Inst/CompressionTreeTest.ComplexTree_Patch_Compress_Decompress/*
//

} /* namespace ddj */
