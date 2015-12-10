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
#include "tree/compression_statistics.hpp"
#include "tree/compression_edge.hpp"

namespace ddj {

class CompressionTree
{
public:
	CompressionTree();
	CompressionTree(SharedCompressionStatisticsPtr stats);
	CompressionTree(SharedCompressionStatisticsPtr stats, int maxHeight);
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

    CompressionTree Copy();
    size_t GetPredictedSizeAfterCompression(SharedCudaPtr<char> data, DataType type);
    void Print(size_t performance = 0);
    void Fix();
    std::vector<CompressionTree> CrossTree(std::vector<CompressionTree> subtrees);

    void UpdateStatistics(SharedCompressionStatisticsPtr stats);
    void SetStatistics(SharedCompressionStatisticsPtr stats) { this->_stats = stats; }
    SharedCompressionStatisticsPtr GetStatistics() { return this->_stats; }

    double GetCompressionRatio() { return this->_root->GetCompressionRatio(); }
    CompressionEdgeVector GetEdges();

private:
    uint GetNextNo();
    void ResetNodeNumbers();

private:
    SharedCompressionNodePtr _root;
    SharedCompressionStatisticsPtr _stats;
    int _maxHeight;
    uint _nextNo;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_TREE_HPP_ */
