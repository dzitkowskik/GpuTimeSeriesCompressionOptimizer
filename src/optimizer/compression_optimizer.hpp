/*
 * compression_optimizer.hpp
 *
 *  Created on: 14 lis 2015
 *      Author: ghash
 */

#ifndef COMPRESSION_OPTIMIZER_HPP_
#define COMPRESSION_OPTIMIZER_HPP_

#include "tree/compression_tree.hpp"

namespace ddj
{

class CompressionOptimizer
{
public:
	CompressionTree OptimizeTree(SharedCudaPtr<char> data, DataType type);
};

} /* namespace ddj */

#endif /* COMPRESSION_OPTIMIZER_HPP_ */
