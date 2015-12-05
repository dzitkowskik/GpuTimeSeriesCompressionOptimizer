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
			(Path
					{
					EncodingType::rle,
						EncodingType::afl, EncodingType::none,
						EncodingType::afl, EncodingType::none
					}
			)
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
		std::vector<PossibleTree> children,
		size_t inputSize,
		size_t parentMetadataSize)
{
	std::vector<PossibleTree> results;
	for(auto& child : children)
	{
		auto tree = parent.first.Copy();
		auto root = child.first.FindNode(0)->Copy();
		auto outputSize = child.second + parentMetadataSize;
		tree.AddNode(root, 0);
		tree.FindNode(0)->SetCompressionRatio(Encoding::GetCompressionRatio(inputSize, outputSize));
		results.push_back(std::make_pair(tree, outputSize));
	}
	return results;
}

std::vector<PossibleTree> PathGenerator::CrossTrees(
		PossibleTree parent,
		std::vector<PossibleTree> childrenLeft,
		std::vector<PossibleTree> childrenRight,
		size_t inputSize,
		size_t parentMetadataSize)
{
	std::vector<PossibleTree> results;
	for(auto& childLeft : childrenLeft)
	{
		for(auto& childRight : childrenRight)
		{
			auto tree = parent.first.Copy();
			size_t outputSize = childLeft.second + childRight.second + parentMetadataSize;
			tree.AddNode(childLeft.first.FindNode(0)->Copy(), 0);
			tree.AddNode(childRight.first.FindNode(0)->Copy(), 0);
			tree.FindNode(0)->SetCompressionRatio(Encoding::GetCompressionRatio(inputSize, outputSize));
			results.push_back(std::make_pair(tree, outputSize));
		}
	}
	return results;
}

size_t _getSize(SharedCudaPtrVector<char> data)
{
	size_t result = 0;
	for(auto& d : data)
		result += d->size();
	return result;
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
	std::vector<PossibleTree> result, part1, part2;
	PossibleTree parent;

	cont = GetContinuations(et, dt, stats, level);
	for(auto& c : cont)
	{
		parent.first = CompressionTree(c, dt);
		auto encoding = factory.Get(c, dt)->Get(data);
		auto compr = encoding->Encode(data, dt);
		parent.second = _getSize(compr);
		parent.first.FindNode(0)->SetCompressionRatio(
				Encoding::GetCompressionRatio(data->size(), parent.second));
		stats = crs.GenerateStatistics(data, dt);
		if(encoding->GetNumberOfResults() == 1)
		{
			part1 = Phase1(compr[1], c, dt, stats, level+1);
			part1 = CrossTrees(parent, part1, data->size(), compr[0]->size());
		}
		else if(encoding->GetNumberOfResults() == 2)
		{
			part1 = Phase1(compr[1], c, dt, stats, level+1);
			part2 = Phase1(compr[2], c, dt, stats, level+1);
			part1 = CrossTrees(parent, part1, part2, data->size(), compr[0]->size());
		}

		if(part1.size() == 0)
			part1 = std::vector<PossibleTree> { parent };

		result.insert(result.end(), part1.begin(), part1.end());
	}

	return result;
}

} /* namespace ddj */
