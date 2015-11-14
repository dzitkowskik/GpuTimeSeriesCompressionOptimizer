/*
 * compression_optimizer.cpp
 *
 *  Created on: 14 lis 2015
 *      Author: ghash
 */

#include "optimizer/compression_optimizer.hpp"
#include "optimizer/path_generator.hpp"
#include "compression/none/none_encoding.hpp"

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

} /* namespace ddj */
