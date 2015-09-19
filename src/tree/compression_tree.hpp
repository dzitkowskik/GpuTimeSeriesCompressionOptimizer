/*
 *  compression_tree.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_TREE_HPP_
#define DDJ_COMPRESSION_TREE_HPP_

#include "compression_node.hpp"
#include "core/cuda_ptr.hpp"

namespace ddj {

class CompressionTree
{
public:
    SharedCudaPtr<char> Compress(SharedCudaPtr<char> data);
    SharedCudaPtr<char> Decompress(SharedCudaPtr<char> data);

    uint AddNode(SharedCompressionNodePtr node, uint parentNo);
    void RemoveNode(uint nodeNo);
    SharedCompressionNodePtr GetNode(uint nodeNo);
    SharedCompressionNodePtr GetRoot();

private:
    SharedCompressionNodePtr _root;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_TREE_HPP_ */
