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
	auto returnTypes = encoding->GetReturnTypes(type);
	auto childrenCnt = encoding->GetNumberOfResults();
	for(int i = 1; i <= childrenCnt; i++)
		CreateTree(tree, path, returnTypes[i-1], node->GetNo(), ++id);
}

CompressionTree PathGenerator::GenerateTree(Path path, DataType type)
{
	CompressionTree tree;
	int id = 0;
	CreateTree(tree, path, type, 0, id);
	return tree;
}

Path PathGenerator::GetContinuations(EncodingType et, DataType dt, DataStatistics stats, int level)
{
	Path result;

	if(et == EncodingType::afl || et == EncodingType::gfc || stats.size <= 0)
		return result;

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
			if(dt == DataType::d_float)
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

} /* namespace ddj */
