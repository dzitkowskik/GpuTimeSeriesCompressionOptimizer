/*
 * optimal_tree_unittest.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: Karol Dzitkowski
 */

#include "test/unittest_base.hpp"
#include "tree/compression_tree.hpp"
#include "optimizer/optimal_tree.hpp"
#include "helpers/helper_comparison.cuh"
#include "helpers/helper_print.hpp"
#include <gtest/gtest.h>

namespace ddj
{

class OptimalTreeTest : public UnittestBase {};

/* FAKE TREE:
				root(RLE)
			   /   		\
			DICT		 PATCH
		   /   \      	/	  \
		AFL	  DELTA	  SCALE	   PATCH
	   /		|		|		 |   \
	NONE	   AFL	   AFL		AFL	  AFL
				|		|		 |	   |
			   NONE	   NONE	    NONE  NONE
*/
CompressionTree GenerateFakeTree()
{
	CompressionTree result;

	auto dt = DataType::d_int;
	auto root = CompressionNode::make_shared(EncodingType::rle, dt);
	auto left = CompressionNode::make_shared(EncodingType::dict, dt);
	auto leftleft = CompressionNode::make_shared(EncodingType::afl, dt);
	auto leftright = CompressionNode::make_shared(EncodingType::delta, dt);
	auto leftrightleft = CompressionNode::make_shared(EncodingType::afl, dt);
	auto right = CompressionNode::make_shared(EncodingType::patch, dt);
	auto rightleft = CompressionNode::make_shared(EncodingType::scale, dt);
	auto rightleftleft = CompressionNode::make_shared(EncodingType::afl, dt);
	auto rightright = CompressionNode::make_shared(EncodingType::patch, dt);
	auto rightrightleft = CompressionNode::make_shared(EncodingType::afl, dt);
	auto rightrightright = CompressionNode::make_shared(EncodingType::afl, dt);

	root->AddChild(left);
	root->AddChild(right);
	left->AddChild(leftleft);
	left->AddChild(leftright);
	leftright->AddChild(leftrightleft);
	right->AddChild(rightleft);
	right->AddChild(rightright);
	rightleft->AddChild(rightleftleft);
	rightright->AddChild(rightrightleft);
	rightright->AddChild(rightrightright);

	EXPECT_TRUE( result.AddNode(root, 0) );
	result.Fix();
	return result;
}

/* FAKE STATISTICS:
0:
	RLE - DICT (3)
	RLE - PATCH (2.5)
	AFL	- NONE (2)
	PATCH - DELTA (1.5)
1:
	RLE - PATCH (4)
	RLE - DELTA (1.5)
	PATCH - SCALE (1)
	PATCH - PATCH (2)
2:
	DICT - AFL (3)
	DELTA - AFL (1)
3:
	DICT - DELTA (2)
4:
	PATCH - SCALE (2.5)
5:
	PATCH - PATCH (3.5)
6:
	AFL	- NONE (4)
8:
	DELTA - AFL (4.23)
10:
	SCALE - AFL (3)
12:
	PATCH - AFL	(5)
13:
	PATCH - AFL	(1)
18:
	AFL - NONE (4)
20:
	AFL - NONE (8)
26:
	AFL - NONE (11)
28:
	AFL - NONE (12)
*/
SharedCompressionStatisticsPtr GenerateFakeStatistics()
{
	auto result = CompressionStatistics::make_shared(5);

	result->Set(0, EdgeType {EncodingType::rle, EncodingType::dict}, 3);
	result->Set(0, EdgeType {EncodingType::rle, EncodingType::patch}, 2.5);
	result->Set(0, EdgeType {EncodingType::afl, EncodingType::none}, 2);
	result->Set(0, EdgeType {EncodingType::patch, EncodingType::delta}, 1.5);

	result->Set(1, EdgeType {EncodingType::rle, EncodingType::patch}, 4);
	result->Set(1, EdgeType {EncodingType::rle, EncodingType::delta}, 1.5);
	result->Set(1, EdgeType {EncodingType::patch, EncodingType::scale}, 1);
	result->Set(1, EdgeType {EncodingType::patch, EncodingType::patch}, 2);

	result->Set(2, EdgeType {EncodingType::dict, EncodingType::afl}, 3);
	result->Set(2, EdgeType {EncodingType::delta, EncodingType::afl}, 1);

	result->Set(3, EdgeType {EncodingType::dict, EncodingType::delta}, 2);
	result->Set(4, EdgeType {EncodingType::patch, EncodingType::scale}, 2.5);
	result->Set(5, EdgeType {EncodingType::patch, EncodingType::patch}, 3.5);
	result->Set(6, EdgeType {EncodingType::afl, EncodingType::none}, 4);
	result->Set(8, EdgeType {EncodingType::delta, EncodingType::afl}, 4.23);
	result->Set(10, EdgeType {EncodingType::scale, EncodingType::afl}, 3);
	result->Set(12, EdgeType {EncodingType::patch, EncodingType::afl}, 5);
	result->Set(13, EdgeType {EncodingType::patch, EncodingType::afl}, 1);
	result->Set(18, EdgeType {EncodingType::afl, EncodingType::none}, 4);

	result->Set(20, EdgeType {EncodingType::afl, EncodingType::none}, 8);
	result->Set(26, EdgeType {EncodingType::afl, EncodingType::none}, 11);
	result->Set(28, EdgeType {EncodingType::afl, EncodingType::none}, 12);

	return result;
}

TEST_F(OptimalTreeTest, Replace_FakedTree_FakedStats_right_ReplaceEdgeOnly)
{
	// Prepare tree
	auto tree = GenerateFakeTree();
	auto stats = GenerateFakeStatistics();
	tree.SetStatistics(stats);
	tree.FindNode(0)->SetCompressionRatio(5.0);
	OptimalTree optTree(tree);

//	printf("Before:\n");
//	tree.Print(tree.GetCompressionRatio());

	// Change statistics and compression ratio
	stats->Set(4, EdgeType {EncodingType::patch, EncodingType::delta}, 1);
	stats->Set(5, EdgeType {EncodingType::patch, EncodingType::delta}, 4);
	stats->Set(12, EdgeType {EncodingType::delta, EncodingType::afl}, 6);
	tree.FindNode(0)->SetCompressionRatio(1.0);

	// Try to correct tree
	auto result = optTree.TryCorrectTree();

//	printf("After:\n");
//	tree.Print(tree.GetCompressionRatio());

	for(auto& edge : tree.GetEdges())
		if(edge.GetNo() == 12)
		{
			EXPECT_EQ(EncodingType::delta, edge.GetType().first);
			EXPECT_EQ(EncodingType::afl, edge.GetType().second);
		} else if(edge.GetNo() == 10) {
			EXPECT_EQ(EncodingType::scale, edge.GetType().first);
			EXPECT_EQ(EncodingType::afl, edge.GetType().second);
		}

	ASSERT_TRUE(result);
}


} /* namespace ddj */


