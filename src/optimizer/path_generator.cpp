/*
 *  path_generator.cpp
 *
 *  Created on: 12/11/2015
 *      Author: Karol Dzitkowski
 */

#include "compression/default_encoding_factory.hpp"
#include "util/statistics/cuda_array_statistics.hpp"

#include <optimizer/path_generator.hpp>
#include <boost/assign/list_of.hpp>
#include <iostream>

#define MAX_LEVEL 3

namespace ddj
{

PathList PathGenerator::GeneratePaths()
{
	PathList firstStage = boost::assign::list_of<Path>
			(Path {})
			(Path {EncodingType::floatToInt})
			(Path {EncodingType::delta})
			(Path {EncodingType::scale})
			(Path {EncodingType::delta, EncodingType::floatToInt})
			(Path {EncodingType::scale, EncodingType::floatToInt})
			;

	PathList secondStage = boost::assign::list_of<Path>
			(Path
					{
					EncodingType::dict,
						EncodingType::afl, EncodingType::none,
						EncodingType::afl, EncodingType::none
					}
			)
			(Path {EncodingType::constData, EncodingType::afl, EncodingType::none})
			(Path {EncodingType::rle, EncodingType::afl, EncodingType::none})
			(Path {EncodingType::afl, EncodingType::none})
			(Path
					{
					EncodingType::patch,
						EncodingType::afl, EncodingType::none,
						EncodingType::afl, EncodingType::none
					}
			)
			;

	PathList result;

	for(auto& a : firstStage)
		for(auto& b : secondStage)
		{
			Path temp = a;
			temp.insert(temp.end(), b.begin(), b.end());
			result.push_back(temp);
		}

	return result;
}

void CreateTree(CompressionTree& tree, Path& path, DataType type, int parentId, int& id)
{
	auto encType = path[id];
	auto encFactory = DefaultEncodingFactory().Get(encType, type);
	auto node = boost::make_shared<CompressionNode>(encFactory);
	tree.AddNode(node, parentId);
	if(encType == EncodingType::none) return;

	auto encoding = encFactory->Get();
	type = encoding->GetReturnType(type);
	auto childrenCnt = encoding->GetNumberOfResults();
	for(int i = 1; i <= childrenCnt; i++)
		CreateTree(tree, path, type, node->GetNo(), ++id);
}

CompressionTree PathGenerator::GenerateTree(Path path, DataType type)
{
	CompressionTree tree;
	int id = 0;
	CreateTree(tree, path, type, 0, id);
	return tree;
}

Path PathGenerator::GetContinuations(EncodingType et, DataType dt, Statistics stats, int level)
{
	Path result;

	if(et == EncodingType::afl || et == EncodingType::gfc) return result;

	if(level < MAX_LEVEL)
	{
		// DELTA
		if(et != EncodingType::delta)
			result.push_back(EncodingType::delta);

		// SCALE
		if(et != EncodingType::scale && stats.min != 0)
			result.push_back(EncodingType::scale);

		// FLOAT
		if(et != EncodingType::floatToInt)
		{
			if(dt != DataType::d_float || dt != DataType::d_double)
				result.push_back(EncodingType::floatToInt);
		}

		// PATCH
		if(et != EncodingType::patch)
			result.push_back(EncodingType::patch);

		// DICT
		if(et != EncodingType::dict)
			result.push_back(EncodingType::dict);

		if(stats.sorted || stats.rlMetric > 2)
		{
			// CONST
			if(et != EncodingType::constData)
				result.push_back(EncodingType::constData);

			// RLE
			if(et != EncodingType::rle)
				result.push_back(EncodingType::rle);
		}
	}

	// AFL & GFC
	if(dt == DataType::d_float || dt == DataType::d_double)
		result.push_back(EncodingType::gfc);
	else
		result.push_back(EncodingType::afl);

	return result;
}

std::vector<PossibleTree> PathGenerator::CrossTrees(
		PossibleTree parent,
		std::vector<PossibleTree> children)
{
	std::vector<PossibleTree> results;
	for(auto& child : children)
	{
		auto tree = parent.first.Copy();
		auto root = child.first.FindNode(0)->Copy();
		tree.AddNode(root, 0);
		results.push_back(std::make_pair(tree, child.second));
	}
	return results;
}

std::vector<PossibleTree> PathGenerator::CrossTrees(
		PossibleTree parent,
		std::vector<PossibleTree> childrenLeft,
		std::vector<PossibleTree> childrenRight)
{
	std::vector<PossibleTree> results;
	for(auto& childLeft : childrenLeft)
	{
		for(auto& childRight : childrenRight)
		{
			printf("a\n");
			auto tree = parent.first.Copy();
			tree.AddNode(childLeft.first.FindNode(0)->Copy(), 0);
			tree.AddNode(childRight.first.FindNode(0)->Copy(), 0);
			results.push_back(std::make_pair(tree, childLeft.second + childRight.second));
		}
	}
	return results;
}

std::vector<PossibleTree> PathGenerator::Phase1(
		SharedCudaPtr<char> data,
		EncodingType et,
		DataType dt,
		Statistics stats,
		int level)
{
	Path cont;
	DefaultEncodingFactory factory;
	CudaArrayStatistics crs;
	std::vector<PossibleTree> result;
	std::vector<PossibleTree> part1, part2;
	PossibleTree parent;
	printf("phase1 data size = %lu\n", data->size());
	cont = GetContinuations(et, dt, stats, level);
	for(auto& c : cont)
	{
		printf("Enc type = %s\n", GetEncodingTypeString(c).c_str());
		parent.first = CompressionTree(c, dt);
		auto encoding = factory.Get(c, dt)->Get(data);
		printf("try to encode\n");
		auto compr = encoding->Encode(data, dt);
		printf("encoded\n");
		parent.second = compr[0]->size();
		printf("try to gen stats\n");
		stats = crs.GenerateStatistics(data, dt);
		printf("compr size = %d\n", compr.size());
		if(encoding->GetNumberOfResults() == 1)
		{
			printf("s1\n");
			parent.second += compr[1]->size();
			part1 = Phase1(compr[1], c, dt, stats, level+1);
			part1 = CrossTrees(parent, part1);
			printf("e1\n");
		}
		else if(encoding->GetNumberOfResults() == 2)
		{
			printf("s2\n");
			parent.second += compr[1]->size();
			part1 = Phase1(compr[1], c, dt, stats, level+1);
			parent.second += compr[2]->size();
			part2 = Phase1(compr[2], c, dt, stats, level+1);
			part1 = CrossTrees(parent, part1, part2);
			printf("e2\n");
		}

		if(part1.size() == 0)
			part1 = std::vector<PossibleTree> { parent };

		result.insert(result.end(), part1.begin(), part1.end());
	}

	return result;
}

} /* namespace ddj */
