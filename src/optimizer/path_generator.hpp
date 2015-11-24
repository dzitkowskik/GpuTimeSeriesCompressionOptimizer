/*
 *  path_generator.hpp
 *
 *  Created on: 12/11/2015
 *      Author: Karol Dzitkowski
 */

#ifndef PATH_GENERATOR_HPP_
#define PATH_GENERATOR_HPP_

#include "compression/encoding_type.hpp"
#include "compression/data_type.hpp"
#include "tree/compression_tree.hpp"
#include "util/statistics/cuda_array_statistics.hpp"

#include <vector>
#include <list>

namespace ddj
{

using Path = std::vector<EncodingType>;
using PathList = std::list<Path>;
using PossibleTree = std::pair<CompressionTree, size_t>;

class PathGenerator
{
public:
	PathList GeneratePaths();
	CompressionTree GenerateTree(Path path, DataType type);
	Path GetContinuations(EncodingType et, DataType dt, Statistics stats, int level);

	std::vector<PossibleTree> CrossTrees(
			PossibleTree parent,
			std::vector<PossibleTree> children);

	std::vector<PossibleTree> CrossTrees(
			PossibleTree parent,
			std::vector<PossibleTree> childrenLeft,
			std::vector<PossibleTree> childrenRight);

	std::vector<PossibleTree> Phase1(
			SharedCudaPtr<char> data,
			EncodingType et,
			DataType dt,
			Statistics stats,
			int level);
};

} /* namespace ddj */

#endif /* PATH_GENERATOR_HPP_ */
