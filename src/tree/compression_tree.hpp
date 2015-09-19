/*
 *  compression_tree.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_TREE_HPP_
#define DDJ_COMPRESSION_TREE_HPP_

namespace ddj {

#include "compression_node.hpp"
#include "core/cuda_ptr.hpp"

class CompressionTree
{
public:
    SharedCudaPtr<char> Compress(SharedCudaPtr<char> data);
    SharedCudaPtr<char> Decompress(SharedCudaPtr<char> data);

    uint AddNode(CompressionNodePtr node, uint parentNo);
    void RemoveNode(uint nodeNo);
    CompressionNodePtr GetNode(uint nodeNo);
    CompressionNodePtr GetRoot();

private:
    CompressionNodePtr _root;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_TREE_HPP_ */
