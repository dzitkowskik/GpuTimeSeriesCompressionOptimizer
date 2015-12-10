/*
 * optimal_tree.hpp
 *
 *  Created on: Dec 9, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef OPTIMAL_TREE_HPP_
#define OPTIMAL_TREE_HPP_

#include "tree/compression_tree.hpp"
#include "tree/compression_statistics.hpp"

#include <queue>
#include <vector>

namespace ddj
{

class OptimalTree
{
public:
	OptimalTree(CompressionTree tree)
		: _maxHeight(5)
	{
		this->_ratio = tree.GetCompressionRatio();
		this->_tree = tree.Copy();
		this->_statistics = tree.GetStatistics()->Copy();
		this-> _maxEdgeNo = (1 << (_maxHeight+1)) - 2;

	}
	~OptimalTree(){}

public:
	CompressionTree& operator->() { return _tree; }
	bool TryCorrectTree();
	void Replace(
			SharedCompressionNodePtr node,
			SharedCompressionStatisticsPtr stats,
			int edgeNo);

private:
	int _maxHeight;
	int _maxEdgeNo;
	double _ratio;
	CompressionTree _tree;
	SharedCompressionStatisticsPtr _statistics;
};

} /* namespace ddj */

#endif /* OPTIMAL_TREE_HPP_ */
