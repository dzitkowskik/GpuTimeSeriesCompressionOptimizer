/*
 *  compression_node.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_NODE_HPP_
#define DDJ_COMPRESSION_NODE_HPP_

#include <boost/shared_ptr.hpp>
#include <core/cuda_ptr.hpp>
#include "compression/encoding_type.hpp"
#include "compression/data_type.hpp"
#include "compression/encoding_factory.hpp"

namespace ddj {

class CompressionNode;

using uint = unsigned int;
using SharedCompressionNodePtr = boost::shared_ptr<CompressionNode>;
using SharedCompressionNodePtrVector = std::vector<SharedCompressionNodePtr>;

class CompressionNode
{
public:
	CompressionNode(uint no, EncodingType encodingType, DataType dataType);
	~CompressionNode();
	CompressionNode(const CompressionNode& other);
//	CompressionNode(CompressionNode&& other) = default;

public:
    void Compress(SharedCudaPtr<char> data);
    void Decompress(SharedCudaPtr<char> data);

    SharedCudaPtr<char> Serialize();
    void Deserialize(SharedCudaPtr<char> data);

    void AddChild(SharedCompressionNodePtr node);
    uint GetNo();
    void RemoveChild(uint no);


private:
    SharedCompressionNodePtrVector _children;
    EncodingType _encodingType;
    DataType _dataType;
    EncodingFactory _encodingFactory;

    SharedCudaPtr<char> _data;
    SharedCudaPtr<char> _metadata;
    
    bool _isLeaf;
    uint _nodeNo;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_NODE_HPP_ */
