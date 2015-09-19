/*
 *  compression_node.hpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

 #ifndef DDJ_COMPRESSION_NODE_HPP_
 #define DDJ_COMPRESSION_NODE_HPP_

 namespace ddj {

class CompressionNode;

#include <boost/shared_ptr.hpp>

template<class T>
using SharedCompressionNodePtr = boost::shared_ptr<CompressionNode>;

template<class T>
using SharedCompressionNodePtrVector = std::vector<SharedNodePtr>;

class CompressionNode
{
public:
    void Compress(SharedCudaPtr<char> data);
    void Decompress(SharedCudaPtr<char> data);

    SharedCudaPtr<char> Serialize();
    void Deserialize(SharedCudaPtr<char> data);

private:
    SharedCompressionNodePtrVector _children;
    EncodingType _encodingType;
    DataType _dataType;
    EncodingFactory _encodingFactory;
    bool _isLeaf;
    SharedCudaPtr<char> _data;
    SharedCudaPtr<char> _metadata;
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_NODE_HPP_ */
