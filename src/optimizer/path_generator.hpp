/*
 * path_generator.hpp
 *
 *  Created on: 12 lis 2015
 *      Author: ghash
 */

#ifndef PATH_GENERATOR_HPP_
#define PATH_GENERATOR_HPP_

#include "compression/encoding_type.hpp"
#include "compression/data_type.hpp"
#include "tree/compression_tree.hpp"

#include <vector>
#include <list>

namespace ddj
{

using Path = std::vector<EncodingType>;
using PathList = std::list<Path>;

class PathGenerator
{
public:
	PathList GeneratePaths();
	CompressionTree GenerateTree(Path path, DataType type);
};

} /* namespace ddj */

#endif /* PATH_GENERATOR_HPP_ */
