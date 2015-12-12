/*
 * optimal_tree.cpp
 *
 *  Created on: Dec 9, 2015
 *      Author: Karol Dzitkowski
 */

#include "optimizer/optimal_tree.hpp"
#include "compression/default_encoding_factory.hpp"

namespace ddj
{

void OptimalTree::Replace(
		SharedCompressionNodePtr node,
		SharedCompressionStatisticsPtr stats,
		int edgeNo)
{
	if(edgeNo > _maxEdgeNo) return;

	int resultCnt = node->GetEncodingFactory()->Get()->GetNumberOfResults();
	for(int i = 0; i < resultCnt; i++)
	{
		auto best = stats->GetBest(edgeNo+i, node->GetEncodingType());
		if(best.value > 1) {
			// TODO: Get correct data type
			auto newChld = CompressionNode::make_shared(best.type.second, DataType::d_int);
			node->AddChild(newChld);
			Replace(newChld, stats, 2*(edgeNo+i) + 2);
		} else node->AddChild(CompressionNode::make_shared(EncodingType::none, DataType::d_int));
	}

	return;
}

bool OptimalTree::TryCorrectTree()
{
	auto oldRatio = this->_ratio;
	auto newRatio = this->_tree.GetCompressionRatio();
	this->_ratio = this->_tree.GetCompressionRatio();
	if(oldRatio <= newRatio && oldRatio > 1) return false;

	// Get the weaker edge with minimal index
	bool replace_parent = false;


	auto newStats = this->_tree.GetStatistics();
	auto oldStats = this->_statistics;

	auto edges = this->_tree.GetEdges();
	for(int i = 0; i < edges.size(); i++)
	{
		auto edgeNo = edges[i].GetNo();
		auto oldValue = oldStats->Get(edgeNo, edges[i].GetType());
		// check if edge i should be replaced
		auto bestEdge = newStats->GetBest(edgeNo);
		if(bestEdge.value > oldValue)
		{
			// check if whole parent node must be replaced or only one edge
			auto newEdgeWithSameBeginning = newStats->GetBest(edgeNo, edges[i].GetType().first);
			if(newEdgeWithSameBeginning.value > oldValue)
			{
				// replace single edge
				// TODO: Get correct data type
				auto encFactory = DefaultEncodingFactory().Get(newEdgeWithSameBeginning.type.second, DataType::d_int);
				edges[i].To()->Reset(encFactory);
				Replace(edges[i].To(), newStats, 2*(edgeNo + 1));
			} else {
				// replace whole parent node
				// TODO: Get correct data type

				auto encFactory = DefaultEncodingFactory().Get(bestEdge.type.first, DataType::d_int);
				edges[i].From()->Reset(encFactory);
				Replace(edges[i].From(), newStats, edgeNo%2 ? edgeNo-1 : edgeNo);
			}
			break;
		}
	}

	return true;
}

} /* namespace ddj */
