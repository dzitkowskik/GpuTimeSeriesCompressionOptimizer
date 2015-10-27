/*
 *  compression_tree_unittest.cpp
 *
 *  Created on: 15/10/2015
 *      Author: Karol Dzitkowski
 */

#include "compression_tree_unittest.hpp"
#include "helpers/helper_comparison.cuh"
#include <boost/make_shared.hpp>

namespace ddj {

INSTANTIATE_TEST_CASE_P(
    CompressionTree_Test_Inst,
    CompressionTreeTest,
    ::testing::Values(10, 1000, 10000));

TEST_P(CompressionTreeTest, SimpleOneNodeTree_Delta_Int_Compress_NoException)
{
	CompressionTree tree;
	auto node = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
	ASSERT_TRUE( tree.AddNode(node, 0) );
	tree.Compress(MoveSharedCudaPtr<int, char>(d_int_random_data->copy()));
}

void CompressionTreeTest::TreeCompressionTest_Compress_Decompress(CompressionTree& compressionTree)
{

    auto dataCopy = d_int_random_data->copy();
	auto compressed = compressionTree.Compress(MoveSharedCudaPtr<int, char>(dataCopy));

	CompressionTree decompressionTree;
	auto decompressed = decompressionTree.Decompress(compressed);

	auto expected = d_int_random_data;
	auto actual = MoveSharedCudaPtr<char, int>(decompressed);

	ASSERT_EQ(expected->size(), actual->size());
	EXPECT_TRUE( CompareDeviceArrays(expected->get(), actual->get(), expected->size()) );

//	printf("Uncompressed size = %d\n", d_int_random_data->size()*sizeof(int));
//	printf("Compressed size = %d\n", compressed->size());
}

TEST_P(CompressionTreeTest, SimpleOneNodeTree_Delta_Int_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto node = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
	auto leafNode = boost::make_shared<CompressionNode>(EncodingType::none, DataType::d_int);
	node->AddChild(leafNode);
	ASSERT_TRUE( compressionTree.AddNode(node, 0) );
	TreeCompressionTest_Compress_Decompress(compressionTree);
}

TEST_P(CompressionTreeTest, SimpleTwoNodeTree_DeltaAndScale_Int_Compress_Decompress)
{
	CompressionTree compressionTree;
	auto node = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
	auto secondNode = boost::make_shared<CompressionNode>(EncodingType::scale, DataType::d_int);
	auto leafNode = boost::make_shared<CompressionNode>(EncodingType::none, DataType::d_int);
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
    auto root = boost::make_shared<CompressionNode>(EncodingType::dict, DataType::d_int);
	auto right = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
	auto left = boost::make_shared<CompressionNode>(EncodingType::scale, DataType::d_int);
    auto leftChild = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
	auto noneLeft = boost::make_shared<CompressionNode>(EncodingType::none, DataType::d_int);
    auto noneRight = boost::make_shared<CompressionNode>(EncodingType::none, DataType::d_int);
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
//	auto root = boost::make_shared<CompressionNode>(EncodingType::patch, DataType::d_int);
//	auto right = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
//	auto left = boost::make_shared<CompressionNode>(EncodingType::dict, DataType::d_int);
//	auto leftChildLeft = boost::make_shared<CompressionNode>(EncodingType::delta, DataType::d_int);
//	auto leftChildRight = boost::make_shared<CompressionNode>(EncodingType::scale, DataType::d_int);
//	auto leaf = boost::make_shared<CompressionNode>(EncodingType::none, DataType::d_int);
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

} /* namespace ddj */
