/*
 *  compression_tree.cpp
 *
 *  Created on: 17-09-2015
 *      Author: Karol Dzitkowski
 */

#include "tree/compression_tree.hpp"
#include "compression/encoding_metadata_header.hpp"
#include "compression/default_encoding_factory.hpp"
#include "util/copy/cuda_array_copy.hpp"

#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <queue>

namespace ddj {

CompressionTree::CompressionTree() : _nextNo(0), _maxHeight(5){}

CompressionTree::CompressionTree(SharedCompressionStatisticsPtr stats)
	: _nextNo(0), _maxHeight(5), _stats(stats)
{}
CompressionTree::CompressionTree(SharedCompressionStatisticsPtr stats, int maxHeight)
	: _nextNo(0), _maxHeight(maxHeight), _stats(stats)
{}

CompressionTree::~CompressionTree(){}
CompressionTree::CompressionTree(const CompressionTree& other)
	: _nextNo(other._nextNo),
	  _root(other._root),
	  _maxHeight(other._maxHeight),
	  _stats(other._stats)
{}

CompressionTree::CompressionTree(EncodingType et, DataType dt)
	: _nextNo(0), _maxHeight(5)
{
	this->AddNode(CompressionNode::make_shared(et, dt), 0);
}

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
    if(_stats != nullptr) this->UpdateStatistics(_stats);
    return CudaArrayCopy().Concatenate(compressionResults);
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
	// printf("DataType = %s\n", GetDataTypeString(root->GetDataType()).c_str());
	return _root->Decompress();
}

size_t CompressionTree::GetPredictedSizeAfterCompression(SharedCudaPtr<char> data, DataType type)
{
	return _root->PredictCompressionSize(data, type);
}

void CompressionTree::Print(size_t performance)
{
	this->_root->Print();
	if(performance != 0) std::cout << "  -  " << performance;
	std::cout << std::endl;
}

std::vector<CompressionTree> CompressionTree::CrossTree(std::vector<CompressionTree> subtrees)
{
	std::vector<CompressionTree> result;
	for(auto& subtree : subtrees)
	{
		auto treeCopy = *this;
		treeCopy.AddNode(subtree.FindNode(0), 0);
		result.push_back(treeCopy);
	}
	return result;
}

CompressionTree CompressionTree::Copy()
{
	CompressionTree tree;
	tree._nextNo = this->_nextNo;
	tree._root = this->_root->Copy();
	return tree;
}

void CompressionTree::Fix()
{
	this->_root->Fix();
}

CompressionEdgeVector CompressionTree::GetEdges()
{
	CompressionEdgeVector result;
	int max = (1 << (_maxHeight+1)) - 2, no = 0;
	std::queue<SharedCompressionNodePtr> fifo;
	fifo.push(this->_root);
	auto fake = CompressionNode::make_shared(EncodingType::none, DataType::d_int);

	while(!fifo.empty() && no < max)
	{
		auto el = fifo.front(); fifo.pop();
		auto chld = el->Children();

		// LEFT
		if(chld.size() >= 1) {
			result.push_back(CompressionEdge(el, chld[0], no));
			fifo.push(chld[0]);
		} else {
			fifo.push(fake);
		} no++;

		// RIGHT
		if(chld.size() >= 2) {
			result.push_back(CompressionEdge(el, chld[1], no));
			fifo.push(chld[1]);
		} else {
			fifo.push(fake);
		} no++;
	}

	return result;
}

void CompressionTree::UpdateStatistics(SharedCompressionStatisticsPtr stats)
{
	auto edges = GetEdges();
	for(auto& edge : edges)
	{
		double ratio = (edge.From()->GetCompressionRatio() + edge.To()->GetCompressionRatio()) / 2.0;
		stats->Update(edge.GetNo(), edge.GetType(), ratio);
	}
}

} /* namespace ddj */
