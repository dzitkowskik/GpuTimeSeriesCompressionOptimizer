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
		auto outputSize = child.second + parentMetadataSize + 8;
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
			size_t outputSize = childLeft.second + childRight.second + parentMetadataSize + 8;
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
		int level)
{
	Path cont;
	DefaultEncodingFactory factory;
	CudaArrayStatistics crs;
	std::vector<PossibleTree> result, part1, part2;
	PossibleTree parent;

	LOG4CPLUS_DEBUG(_logger, "Try generate statistics");
	DataStatistics stats = crs.GenerateStatistics(data, dt);
	LOG4CPLUS_DEBUG(_logger, "Statistics generated");

	cont = _pathGenerator.GetContinuations(et, dt, stats, level);
	for(auto& c : cont)
	{
		parent.first = CompressionTree(c, dt);
		auto encoding = factory.Get(c, dt)->Get(data);

		LOG4CPLUS_DEBUG_FMT(_logger, "Compress %s of size %lu using %s",
			GetDataTypeString(dt).c_str(), data->size(), GetEncodingTypeString(c).c_str());
		auto compr = encoding->Encode(data, dt);
		LOG4CPLUS_DEBUG(_logger, "Compression success");

		parent.second = _getSize(compr);
		parent.first.FindNode(0)->SetCompressionRatio(
				Encoding::GetCompressionRatio(data->size(), parent.second));

		auto returnTypes = encoding->GetReturnTypes(dt);

		if(encoding->GetNumberOfResults() == 1)
		{
			part1 = FullStatisticsUpdate(compr[1], c, returnTypes[0], level+1);
			part1 = CrossTrees(parent, part1, data->size(), compr[0]->size());
		}
		else if(encoding->GetNumberOfResults() == 2)
		{
			part1 = FullStatisticsUpdate(compr[1], c, returnTypes[0], level+1);
			part2 = FullStatisticsUpdate(compr[2], c, returnTypes[1], level+1);
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
	if(_optimalTree == nullptr || _partsProcessed % 5 == 4) return true;
	return false;
}

size_t CompressionOptimizer::GetSampleDataForFullUpdateSize(size_t partDataSize, DataType type)
{
	size_t typeSizeInBytes = GetDataTypeSize(type);
	size_t numberOfElements = partDataSize / typeSizeInBytes;
	if (numberOfElements <= 10000) return partDataSize;
	else return 10000*typeSizeInBytes;
}

SharedCudaPtr<char> CompressionOptimizer::CompressData(SharedCudaPtr<char> dataPart, DataType type)
{
	if(IsFullUpdateNeeded())
	{
		LOG4CPLUS_DEBUG(_logger, "START full statistics update");
		auto dataSampleForFullUpdate = dataPart->copy(
			GetSampleDataForFullUpdateSize(dataPart->size(), type));
		auto possibleTrees = FullStatisticsUpdate(
			dataSampleForFullUpdate, EncodingType::none, type, 0);
		LOG4CPLUS_DEBUG(_logger, "END full statistics update");

		for(auto& tree : possibleTrees)
		{
			tree.first.Fix();
			tree.first.UpdateStatistics(_statistics);
			tree.first.SetStatistics(_statistics);
		}

		auto bestTree = std::max_element(
			possibleTrees.begin(),
			possibleTrees.end(),
			[&](PossibleTree A, PossibleTree B){ return A.first.GetCompressionRatio() < B.first.GetCompressionRatio(); });

	//		for(int i = 0; i < 10; i++)
	//		{
	//			LOG4CPLUS_INFO_FMT(_logger, "PossibleTree[%d](size = %lu)(val = %f): %s",
	//					i, possibleTrees[i].second, possibleTrees[i].first.GetCompressionRatio(),
	//					possibleTrees[i].first.ToString().c_str());
	//		}

		LOG4CPLUS_INFO_FMT(_logger, "Found %d interesting trees (best score = %f)",
				possibleTrees.size(), (*bestTree).first.GetCompressionRatio());

		_optimalTree = OptimalTree::make_shared((*bestTree).first);
	}

	auto compressedData = _optimalTree->GetTree().Compress(dataPart);
	LOG4CPLUS_INFO(_logger, "Optimal tree: " << _optimalTree->GetTree().ToString());
	bool treeCorrected = _optimalTree->TryCorrectTree();
	LOG4CPLUS_INFO_FMT(_logger, "Tree corrected = %s\n", treeCorrected ? "true" : "false");
	_partsProcessed++;
	_totalBytesProcessed += dataPart->size();
	return compressedData;
}

} /* namespace ddj */
