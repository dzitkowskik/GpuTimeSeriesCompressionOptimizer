/*
 *  compression_optimizer.hpp
 *
 *  Created on: 14/11/2015
 *      Author: Karol Dzitkowski
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
