/*
 *  compression_tree.cpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#include "tree/compression_tree.hpp"
#include "compression/encoding_metadata_header.hpp"
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>

#include "compression/default_encoding_factory.hpp"

namespace ddj {

CompressionTree::CompressionTree() : _nextNo(0) {}
CompressionTree::~CompressionTree(){}
CompressionTree::CompressionTree(const CompressionTree& other)
	: _nextNo(other._nextNo), _root(other._root) {}

void CompressionTree::Reset()
{
	_nextNo = 0;
	_root = nullptr;
}

uint CompressionTree::GetNextNo()
{
	return _nextNo++;
}

SharedCompressionNodePtr CompressionTree::FindNode(uint nodeNo)
{
	if(nodeNo == 0) return _root;
	else return _root->FindChild(nodeNo);
}

bool CompressionTree::AddNode(SharedCompressionNodePtr node, uint parentNo)
{
	auto getNextNoFunc = boost::bind(&CompressionTree::GetNextNo, this);

	if(_root == nullptr) // we should add root
	{
		node->SetNo(getNextNoFunc);
		_root = node;
		return true;
	} else {
		auto parent = FindNode(parentNo);
		if(parent != nullptr)
		{
			node->SetNo(getNextNoFunc);
			node->SetParentNo(parentNo);
			parent->AddChild(node);
			return true;
		}
	}
	return false;
}

bool CompressionTree::RemoveNode(uint no)
{
	if(_root != nullptr)
	{
		if(no == 0)
		{
			_root = SharedCompressionNodePtr();
		} else {
			auto node = _root->FindChild(no);
			auto parent = _root->FindChild(node->GetParentNo());
			parent->RemoveChild(no);
		}
	}

	return false;
}

SharedCudaPtr<char> CompressionTree::Compress(SharedCudaPtr<char> data)
{
    auto compressionResults = _root->Compress(data);
    return Concatenate(compressionResults);
}

SharedCompressionNodePtr DecompressNodes(SharedCudaPtr<char> compressed_data, size_t& offset)
{
	// READ METADATA HEADER
	EncodingMetadataHeader header;
	size_t metadataHeaderSize = sizeof(EncodingMetadataHeader);
	CUDA_CALL( cudaMemcpy(&header, compressed_data->get()+offset, metadataHeaderSize, CPY_DTH) );
	offset += sizeof(EncodingMetadataHeader);

	// CREATE NODE
	EncodingType encType = (EncodingType)header.EncodingType;
	auto defaultEncodingFactory = DefaultEncodingFactory::Get(encType, (DataType)header.DataType);
	SharedCompressionNodePtr node(new CompressionNode(defaultEncodingFactory));

	// READ METADATA AND UPDATE OFFSET + SAVE TO NODE
	auto metadata = CudaPtr<char>::make_shared();
	metadata->fill(compressed_data->get()+offset, header.MetadataLength);
	offset += header.MetadataLength;
	node->SetMetadata(metadata);

	if(encType == EncodingType::none) // IF LEAF
	{
		// READ DATA AND UPDATE OFFSET
		size_t data_size;
		CUDA_CALL( cudaMemcpy(&data_size, metadata->get(), sizeof(size_t), CPY_DTH) );
		auto data = CudaPtr<char>::make_shared();
		data->fill(compressed_data->get()+offset, data_size);
		offset += data_size;
		node->SetData(data);
	}
	else //	RUN RECURSIVELY TO CHILD NODES
	{
		auto encoding = defaultEncodingFactory->Get();
		int noChildren = encoding->GetNumberOfResults();
		for(int i = 0; i < noChildren; i++)
		{
			auto childNode = DecompressNodes(compressed_data, offset);
			node->AddChild(childNode);
		}
	}

	return node;
}

SharedCudaPtr<char> CompressionTree::Decompress(SharedCudaPtr<char> data)
{
	Reset();
	size_t offset = 0;
	auto root = DecompressNodes(data, offset);
	AddNode(root, 0);
	return _root->Decompress();
}

size_t CompressionTree::GetPredictedSizeAfterCompression(SharedCudaPtr<char> data, DataType type)
{
	return _root->PredictCompressionSize(data, type);
}

void CompressionTree::Print()
{
	this->_root->Print();
	std::cout << std::endl;
}
























} /* namespace ddj */
