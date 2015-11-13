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
#include <boost/function.hpp>

#include "compression/encoding_type.hpp"
#include "compression/data_type.hpp"
#include "compression/encoding_factory.hpp"

namespace ddj {

class CompressionNode;

using uint = unsigned int;
using SharedEncodingFactoryPtr = boost::shared_ptr<EncodingFactory>;
using SharedCompressionNodePtr = boost::shared_ptr<CompressionNode>;
using SharedCompressionNodePtrVector = std::vector<SharedCompressionNodePtr>;

class CompressionNode
{
public:
	CompressionNode(SharedEncodingFactoryPtr encodingFactory);
	~CompressionNode();
	CompressionNode(const CompressionNode& other);

public:
    SharedCudaPtrVector<char> Compress(SharedCudaPtr<char> data);
    SharedCudaPtr<char> Decompress();

    SharedCudaPtr<char> Serialize();
    void Deserialize(SharedCudaPtr<char> data);

    void AddChild(SharedCompressionNodePtr node);
    void RemoveChild(uint no);
    SharedCompressionNodePtr FindChild(uint no);

    uint GetNo();
    void SetNo(uint no);
    void SetNo(boost::function<uint ()> nextNoFunction);

    uint GetParentNo();
    void SetParentNo(uint no);

    void SetMetadata(SharedCudaPtr<char> metadata);
    void SetData(SharedCudaPtr<char> data);

    template<typename T> static
    SharedCompressionNodePtr make_shared(DataType type)
    {
    	return boost::make_shared<CompressionNode>(boost::make_shared<T>(type));
    }

    size_t PredictCompressionSize(SharedCudaPtr<char> data, DataType type);

    void Print();

private:
    SharedCudaPtr<char> PrepareMetadata(SharedCudaPtr<char> encodingMetadata);

private:
    SharedEncodingFactoryPtr _encodingFactory;
    SharedCompressionNodePtrVector _children;
    EncodingType _encodingType;
    DataType _dataType;

    SharedCudaPtr<char> _data;
    SharedCudaPtr<char> _metadata;
    
    bool _isLeaf;
    uint _nodeNo;
    uint _parentNo;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_NODE_HPP_ */
