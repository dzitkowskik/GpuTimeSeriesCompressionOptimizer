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
	CompressionTree();
	CompressionTree(EncodingType et, DataType dt);
	~CompressionTree();
	CompressionTree(const CompressionTree& other);

public:
    SharedCudaPtr<char> Compress(SharedCudaPtr<char> data);
    SharedCudaPtr<char> Decompress(SharedCudaPtr<char> data);

    SharedCudaPtr<char> Serialize();
    static CompressionTree Deserialize(SharedCudaPtr<char> data);

    SharedCompressionNodePtr FindNode(uint nodeNo);
    bool AddNode(SharedCompressionNodePtr node, uint parentNo);
    bool RemoveNode(uint nodeNo);
    void Reset();

    size_t GetPredictedSizeAfterCompression(SharedCudaPtr<char> data, DataType type);

    void Print();

    std::vector<CompressionTree> CrossTree(std::vector<CompressionTree> subtrees);

private:
    uint GetNextNo();
    void ResetNodeNumbers();

private:
    SharedCompressionNodePtr _root;
    uint _nextNo;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_TREE_HPP_ */
