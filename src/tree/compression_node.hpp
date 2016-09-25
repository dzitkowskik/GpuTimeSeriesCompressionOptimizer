/*
 *  compression_node.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_NODE_HPP_
#define DDJ_COMPRESSION_NODE_HPP_

#include "core/cuda_ptr.hpp"
#include "compression/encoding_type.hpp"
#include "data/data_type.hpp"
#include "compression/encoding_factory.hpp"
#include "compression/default_encoding_factory.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <string>
#include <sstream>
#include <iostream>

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

    EncodingType GetEncodingType() { return _encodingType; }
    DataType GetDataType() { return _dataType; }
    SharedEncodingFactoryPtr GetEncodingFactory() { return _encodingFactory; }

    void SetMetadata(SharedCudaPtr<char> metadata);
    void SetData(SharedCudaPtr<char> data);

    size_t PredictCompressionSize(SharedCudaPtr<char> data, DataType type);

    void Print(std::ostream& stream = std::cout);
	std::string ToString();
    void Fix();
    void Reset(SharedEncodingFactoryPtr encodingFactory);

    SharedCompressionNodePtr Copy();
    SharedCompressionNodePtrVector& Children() { return _children; }

    void SetCompressionRatio(double ratio) { _compressionRatio = ratio; }
    double GetCompressionRatio() { return _compressionRatio; }

public:
    static SharedCompressionNodePtr make_shared(EncodingType et, DataType dt)
    {
    	return boost::make_shared<CompressionNode>(DefaultEncodingFactory().Get(et, dt));
    }

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
    double _compressionRatio;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_NODE_HPP_ */
