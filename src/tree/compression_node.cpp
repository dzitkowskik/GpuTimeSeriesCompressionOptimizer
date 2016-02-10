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
#include <iostream>

namespace ddj {

CompressionNode::CompressionNode(SharedEncodingFactoryPtr factory)
	: _nodeNo(0), _parentNo(0), _isLeaf(true), _encodingFactory(factory), _compressionRatio(1.0)
{
	_encodingType = factory->encodingType;
	_dataType = factory->dataType;
}

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
	_compressionRatio = other._compressionRatio;
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
	auto encoding = _encodingFactory->Get(data);

//	 printf("START Encoding using %s - data %s\n",
//	 		GetEncodingTypeString(_encodingFactory->encodingType).c_str(),
//	 		GetDataTypeString(_dataType).c_str());

	auto encodingResult = encoding->Encode(data, _dataType);

	int i = 0;
	auto encodingMetadata = encodingResult[i++];
	auto metadata = PrepareMetadata(encodingMetadata);
	SharedCudaPtrVector<char> result = {metadata};

	if(_isLeaf) result.push_back(encodingResult[i++]);
	else
		for(auto& child : _children)
		{
//			printf("Encoding Result[%d] size = %lu\n", i, encodingResult[i]->size());
			auto childResult = child->Compress(encodingResult[i++]);
			result.insert(result.end(), childResult.begin(), childResult.end());
		}

	// update compression ratio
	size_t inputSize = data->size();
	size_t outputSize = 0;
	for(auto& r : result) outputSize += r->size();
	this->SetCompressionRatio(Encoding::GetCompressionRatio(inputSize, outputSize));

//	 printf("END Encoding using %s - data %s\n",
//	 		GetEncodingTypeString(_encodingFactory->encodingType).c_str(),
//	 		GetDataTypeString(_dataType).c_str());

	return result;
}

SharedCudaPtr<char> CompressionNode::Decompress()
{
	auto encoding = _encodingFactory->Get();
	// printf("START Decoding using %s - data %s\n",
	// 		GetEncodingTypeString(_encodingFactory->encodingType).c_str(),
	// 		GetDataTypeString(_dataType).c_str());

	SharedCudaPtrVector<char> data { _metadata };

	if(_isLeaf) data.push_back(_data);
	else
		for(auto& child : _children)
		{
			auto childResult = child->Decompress();
			data.push_back(childResult);
		}
	auto result = encoding->Decode(data, _dataType);
	// printf("END Decoding using %s - data %s\n",
	// 		GetEncodingTypeString(_encodingFactory->encodingType).c_str(),
	// 		GetDataTypeString(_dataType).c_str());
	return result;
}

size_t CompressionNode::PredictCompressionSize(SharedCudaPtr<char> data, DataType type)
{
	auto encoding = _encodingFactory->Get(data);
	size_t size = encoding->GetMetadataSize(data, type) + sizeof(EncodingMetadataHeader);
	if(_isLeaf) return size + encoding->GetCompressedSize(data, type);
	auto compressed = encoding->Encode(data, type);
	auto returnTypes = encoding->GetReturnTypes(type);
	int i = 1;
	for(auto& child : _children)
	{
		size += child->PredictCompressionSize(compressed[i], returnTypes[i-1]);
		i++;
	}
	return size;
}


void CompressionNode::Print(std::ostream& stream)
{
	stream << GetEncodingTypeString(this->_encodingType) << "[" << this->_compressionRatio << "]" << ",";
	for(auto& child : this->_children)
		child->Print(stream);
}

std::string CompressionNode::ToString()
{
	std::stringstream ss;
	Print(ss);
	return ss.str();
}

SharedCompressionNodePtr CompressionNode::Copy()
{
	CompressionNode* node = new CompressionNode(this->_encodingFactory);
	node->_isLeaf = this->_isLeaf;
	node->_nodeNo = this->_nodeNo;
	node->_parentNo = this->_parentNo;
	node->_encodingType = this->_encodingType;
	node->_dataType = this->_dataType;
	node->_compressionRatio = this->_compressionRatio;
	for(auto& child : this->_children)
		node->_children.push_back(child->Copy());

	return boost::shared_ptr<CompressionNode>(node);
}

void CompressionNode::Fix()
{
	if(_isLeaf)
	{
		for(int i = 0; i < _encodingFactory->Get()->GetNumberOfResults(); i++)
		{
			auto leaf = boost::make_shared<CompressionNode>(
					DefaultEncodingFactory().Get(EncodingType::none, _dataType));
			leaf->SetCompressionRatio(1.0);
			this->AddChild(leaf);
		}
	}
	else
		for(auto& child : _children) child->Fix();
}

void CompressionNode::Reset(SharedEncodingFactoryPtr encodingFactory)
{
	// free resources
	this->_children.clear();
	if(this->_data != nullptr) this->_data->clear();
	if(this->_metadata != nullptr) this->_metadata->clear();

	// this is last and not used yet node
	this->_isLeaf = true;
	this->_compressionRatio = 1.0;

	// set new encoding factory
	this->_encodingFactory = encodingFactory;
	this->_encodingType = encodingFactory->encodingType;
	this->_dataType = encodingFactory->dataType;
}

} /* namespace ddj */
