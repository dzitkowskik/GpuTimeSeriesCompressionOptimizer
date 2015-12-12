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

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <queue>
#include <vector>

namespace ddj
{

class OptimalTree;
using SharedOptimalTreePtr = boost::shared_ptr<OptimalTree>;

class OptimalTree
{
public:
	OptimalTree(CompressionTree tree)
		: _maxHeight(5)
	{
		this->_ratio = tree.GetCompressionRatio();
		this->_tree = tree;
		this->_statistics = tree.GetStatistics()->Copy();
		this-> _maxEdgeNo = (1 << (_maxHeight+1)) - 2;

	}
	~OptimalTree(){}

public:
	CompressionTree& GetTree() { return _tree; }
	bool TryCorrectTree();
	void Replace(
			SharedCompressionNodePtr node,
			SharedCompressionStatisticsPtr stats,
			int edgeNo);

public:
	static SharedOptimalTreePtr make_shared(CompressionTree tree)
	{
		return boost::make_shared<OptimalTree>(tree);
	}

private:
	int _maxHeight;
	int _maxEdgeNo;
	double _ratio;
	CompressionTree _tree;
	SharedCompressionStatisticsPtr _statistics;
};

} /* namespace ddj */

#endif /* OPTIMAL_TREE_HPP_ */
