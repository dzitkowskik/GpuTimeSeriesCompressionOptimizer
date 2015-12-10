/*
 * compression_edge.hpp
 *
 *  Created on: Dec 10, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_EDGE_HPP_
#define COMPRESSION_EDGE_HPP_

#include "tree/compression_statistics.hpp"
#include "tree/compression_node.hpp"

#include <vector>

namespace ddj {

class CompressionEdge;
using CompressionEdgeVector = std::vector<CompressionEdge>;

class CompressionEdge
{
public:
	CompressionEdge(SharedCompressionNodePtr from, SharedCompressionNodePtr to, int no)
		: _from(from), _to(to), _no(no)
	{
		this->_type.first = from->GetEncodingType();
		this->_type.second = to->GetEncodingType();
	}
	~CompressionEdge(){}
	CompressionEdge(const CompressionEdge& other)
		: _from(other._from), _to(other._to), _no(other._no)
	{
		this->_type.first = other._type.first;
		this->_type.second = other._type.second;
	}
	CompressionEdge(CompressionEdge&& other)
		: _from(std::move(other._from)),
		  _to(std::move(other._to)),
		  _no(std::move(other._no))
	{
		this->_type.first = std::move(other._type.first);
		this->_type.second = std::move(other._type.second);
	}

public:
	int GetNo() { return _no; }
	EdgeType GetType() { return _type; }
	SharedCompressionNodePtr From() { return _from; }
	SharedCompressionNodePtr To() { return _to; }

private:
	int	_no;
	EdgeType _type;
	SharedCompressionNodePtr _from;
	SharedCompressionNodePtr _to;
};

} /* namespace ddj */

#endif /* COMPRESSION_EDGE_HPP_ */
