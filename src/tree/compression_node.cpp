/*
 *  compression_node.cpp
 *
 *  Created on: 19-09-2015
 *      Author: Karol Dzitkowski
 */

#include "compression_node.hpp"
#include <algorithm>
#include <vector>

namespace ddj {

CompressionNode::CompressionNode(uint no, EncodingType encodingType, DataType dataType)
{
	_isLeaf = true;
	_nodeNo = no;
	_encodingType = encodingType;
	_dataType = dataType;
}

CompressionNode::~CompressionNode(){}

CompressionNode::CompressionNode(const CompressionNode& other)
{
	_isLeaf = other._isLeaf;
	_nodeNo = other._nodeNo;
	_encodingType = other._encodingType;
	_dataType = other._dataType;
	_data = other._data;
	_metadata = other._metadata;
	_children = other._children;
}

void CompressionNode::AddChild(SharedCompressionNodePtr node)
{
	_children.push_back(node);
}

void CompressionNode::RemoveChild(uint no)
{
	_children.erase(
	    std::remove_if(_children.begin(), _children.end(),
	        [=](const SharedCompressionNodePtr& node) { return node->GetNo() == no; }),
	        _children.end());
}

uint CompressionNode::GetNo() { return _nodeNo; }


} /* namespace ddj */
