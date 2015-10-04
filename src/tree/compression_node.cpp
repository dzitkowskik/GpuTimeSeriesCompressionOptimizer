/*
 *  compression_node.cpp
 *
 *  Created on: 19-09-2015
 *      Author: Karol Dzitkowski
 */

#include "compression_node.hpp"
#include "compression/encoding_metadata_header.hpp"
#include <algorithm>
#include <vector>

namespace ddj {

CompressionNode::CompressionNode(EncodingType encodingType, DataType dataType)
	: _nodeNo(0), _parentNo(0), _isLeaf(true), _encodingType(encodingType), _dataType(dataType)
{}

CompressionNode::~CompressionNode(){}

CompressionNode::CompressionNode(const CompressionNode& other)
{
	_isLeaf = other._isLeaf;
	_nodeNo = other._nodeNo;
	_parentNo = other._parentNo;
	_encodingType = other._encodingType;
	_dataType = other._dataType;
	_data = other._data;
	_metadata = other._metadata;
	_children = other._children;
}

void CompressionNode::AddChild(SharedCompressionNodePtr node)
{
	_isLeaf = false;
	_children.push_back(node);
}

void CompressionNode::RemoveChild(uint no)
{
	_children.erase(
	    std::remove_if(_children.begin(), _children.end(),
	        [=](const SharedCompressionNodePtr& node) { return node->GetNo() == no; }),
	        _children.end());
}

SharedCompressionNodePtr CompressionNode::FindChild(uint no)
{
	SharedCompressionNodePtr result;

	for(auto& child : _children)
	{
		if(child->GetNo() == no) return child;
		auto childResult = child->FindChild(no);
		if(childResult != nullptr) return childResult;
	}

	return result;
}

uint CompressionNode::GetNo() { return _nodeNo; }
void CompressionNode::SetNo(uint no) { _nodeNo = no; }

void CompressionNode::SetNo(boost::function<uint ()> nextNoFunction)
{
	SetNo(nextNoFunction());
	if(_isLeaf) return;
	//ELSE
	for(auto& child : _children)
		child->SetNo(nextNoFunction);
}

uint CompressionNode::GetParentNo() { return _parentNo; }
void CompressionNode::SetParentNo(uint no) { _parentNo = no; }

void CompressionNode::SetMetadata(SharedCudaPtr<char> metadata) { _metadata = metadata; }
void CompressionNode::SetData(SharedCudaPtr<char> data) { _data = data; }

SharedCudaPtr<char> CompressionNode::PrepareMetadata(SharedCudaPtr<char> encodingMetadata)
{
	EncodingMetadataHeader header;
	header.EncodingType = (int16_t)_encodingType;
	header.DataType = (int16_t)_dataType;
	header.MetadataLength = encodingMetadata->size();
	auto result = CudaPtr<char>::make_shared(sizeof(EncodingMetadataHeader)+encodingMetadata->size());
	CUDA_CALL( cudaMemcpy(result->get(), &header, sizeof(EncodingMetadataHeader), CPY_HTD) );
	CUDA_CALL(
		cudaMemcpy(
			result->get()+sizeof(EncodingMetadataHeader),
			encodingMetadata->get(),
			encodingMetadata->size(),
			CPY_DTD
		)
	);
	return result;
}

SharedCudaPtrVector<char> CompressionNode::Compress(SharedCudaPtr<char> data)
{
	auto encoding = _encodingFactory.Get(_encodingType);
	auto encodingResult = encoding->Encode(data, _dataType);

	if(_isLeaf) return encodingResult;

	// ELSE
	int i = 0;
	auto encodingMetadata = encodingResult[i++];
	auto metadata = PrepareMetadata(encodingMetadata);
	SharedCudaPtrVector<char> result = {metadata};
	for(auto& child : _children)
	{
		auto childResult = child->Compress(encodingResult[i++]);
		result.insert(result.end(), childResult.begin(), childResult.end());
	}
	return result;
}

SharedCudaPtr<char> CompressionNode::Decompress()
{
	if(_isLeaf) return _data;

	// ELSE
	auto encoding = _encodingFactory.Get(_encodingType);
	SharedCudaPtrVector<char> data;
	for(auto& child : _children)
	{
		auto childResult = child->Decompress();
		data.push_back(childResult);
	}
	auto concatenatedData = Concatenate(data);
	return encoding->Decode(SharedCudaPtrVector<char> {_metadata, concatenatedData}, _dataType);
}

} /* namespace ddj */
