/*
 *  compression_optimizer.cpp
 *
 *  Created on: 14/11/2015
 *      Author: Karol Dzitkowski
 */

#include "optimizer/compression_optimizer.hpp"
#include "optimizer/path_generator.hpp"
#include "compression/none/none_encoding.hpp"
#include <algorithm>

namespace ddj
{

CompressionTree CompressionOptimizer::OptimizeTree(SharedCudaPtr<char> data, DataType type)
{
	auto paths = PathGenerator().GeneratePaths();
	size_t minSize = data->size();
	CompressionTree result;
	auto leaf = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(type));
	result.AddNode(leaf, 0);

	for(auto& path : paths)
	{
		auto tree = PathGenerator().GenerateTree(path, type);
		auto actual = tree.GetPredictedSizeAfterCompression(data, type);
		if(actual < minSize)
		{
			minSize = actual;
			result = tree;
		}
	}

	return result;
}

std::vector<PossibleTree> CompressionOptimizer::CrossTrees(
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

std::vector<PossibleTree> CompressionOptimizer::CrossTrees(
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

std::vector<PossibleTree> CompressionOptimizer::FullStatisticsUpdate(
		SharedCudaPtr<char> data,
		EncodingType et,
		DataType dt,
		DataStatistics stats,
		int level)
{
	Path cont;
	DefaultEncodingFactory factory;
	CudaArrayStatistics crs;
	std::vector<PossibleTree> result, part1, part2;
	PossibleTree parent;

	cont = _pathGenerator.GetContinuations(et, dt, stats, level);
	for(auto& c : cont)
	{
		parent.first = CompressionTree(c, dt);
		auto encoding = factory.Get(c, dt)->Get(data);
//		printf("Compress %s using %s\n", GetDataTypeString(dt).c_str(), GetEncodingTypeString(c).c_str());
		auto compr = encoding->Encode(data, dt);
//		printf("Compression success!\n");
		parent.second = _getSize(compr);
		parent.first.FindNode(0)->SetCompressionRatio(
				Encoding::GetCompressionRatio(data->size(), parent.second));
//		printf("Try generate statistics\n");
		stats = crs.GenerateStatistics(data, dt);
//		printf("Statistics generated!\n");
		if(encoding->GetNumberOfResults() == 1)
		{
			part1 = FullStatisticsUpdate(compr[1], c, dt, stats, level+1);
			part1 = CrossTrees(parent, part1, data->size(), compr[0]->size());
		}
		else if(encoding->GetNumberOfResults() == 2)
		{
			part1 = FullStatisticsUpdate(compr[1], c, dt, stats, level+1);
			part2 = FullStatisticsUpdate(compr[2], c, dt, stats, level+1);
			part1 = CrossTrees(parent, part1, part2, data->size(), compr[0]->size());
		}

		if(part1.size() == 0)
			part1 = std::vector<PossibleTree> { parent };

		result.insert(result.end(), part1.begin(), part1.end());
	}

	return result;
}

bool CompressionOptimizer::IsFullUpdateNeeded()
{
	if(_optimalTree == nullptr || _partsProcessed % 10 == 7) return true;
	return false;
}

size_t CompressionOptimizer::GetSampleDataForFullUpdateSize(size_t partDataSize, DataType type)
{
	size_t typeSizeInBytes = GetDataTypeSize(type);
	return std::min(partDataSize/(typeSizeInBytes*1000), 1000*typeSizeInBytes);
}

SharedCudaPtr<char> CompressionOptimizer::CompressData(SharedCudaPtr<char> dataPart, DataType type)
{
	if(IsFullUpdateNeeded())
	{
		auto dataSampleForFullUpdate = dataPart->copy(GetSampleDataForFullUpdateSize(dataPart->size(), type));
		auto dataSampleStatistics = CudaArrayStatistics().GenerateStatistics(dataSampleForFullUpdate, type);
		auto possibleTrees = FullStatisticsUpdate(
				dataSampleForFullUpdate,
				EncodingType::none,
				type,
				dataSampleStatistics,
				0);
//		printf("Full statistics update DONE\n");
		for(auto& tree : possibleTrees)
		{
			tree.first.Fix();
			tree.first.UpdateStatistics(_statistics);
			tree.first.SetStatistics(_statistics);
		}

		auto bestTree = std::min_element(
			possibleTrees.begin(),
			possibleTrees.end(),
			[&](PossibleTree A, PossibleTree B){ return A.second < B.second; });


		_optimalTree = OptimalTree::make_shared((*bestTree).first);
	}

	auto compressedData = _optimalTree->GetTree().Compress(dataPart);
	bool treeCorrected = _optimalTree->TryCorrectTree();
//	printf("Tree corrected = %s\n", treeCorrected ? "true" : "false");
	_partsProcessed++;
	_totalBytesProcessed += dataPart->size();
	return compressedData;
}

} /* namespace ddj */
