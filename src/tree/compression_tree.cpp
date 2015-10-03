/*
 *  compression_tree.cpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#include "tree/compression_tree.hpp"

namespace ddj {


uint CompressionTree::AddNode(SharedCompressionNodePtr node, uint parentNo)
{
	if(_root.get() == nullptr) // we should add root
	{
		nextNo = 0;
		_root = node;
		nextNo++;
		return 0;
	} else {

	}
	return 0;
}

SharedCudaPtr<char> CompressionTree::Compress(SharedCudaPtr<char> data)
{
    throw std::runtime_error("Not implemented exception!");
}

} /* namespace ddj */
