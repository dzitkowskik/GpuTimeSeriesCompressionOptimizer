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

	auto encoding = node->GetEncodingFactory()->Get();
	int resultCnt = encoding->GetNumberOfResults();
	auto returnTypes = encoding->GetReturnTypes(node->GetEncodingFactory()->dataType);
	for(int i = 0; i < resultCnt; i++)
	{
		auto best = stats->GetBest(edgeNo+i, node->GetEncodingType());
		if(best.value > 1) {
			// TODO: Get correct data type
			auto newChld = CompressionNode::make_shared(best.type.second, returnTypes[i]);
			node->AddChild(newChld);
			Replace(newChld, stats, 2*(edgeNo+i) + 2);
		} else node->AddChild(CompressionNode::make_shared(EncodingType::none, returnTypes[i]));
	}

	return;
}

bool OptimalTree::TryCorrectTree()
{
	auto oldRatio = this->_ratio;
	auto newRatio = this->_tree.GetCompressionRatio();

	LOG4CPLUS_INFO_FMT(_logger, "Old Ratio = %f, New Ratio = %f", oldRatio, newRatio);

	if(oldRatio <= newRatio && oldRatio > 1) return false;

	// Get the weaker edge with minimal index
	auto newStats = this->_tree.GetStatistics();
	auto oldStats = this->_statistics;

	auto edges = this->_tree.GetEdges();
	for(int i = 0; i < edges.size(); i++)
	{
		auto edgeNo = edges[i].GetNo();
		auto oldValue = oldStats->Get(edgeNo, edges[i].GetType());
		auto newValue = newStats->Get(edgeNo, edges[i].GetType());

		auto bestEdge = newStats->GetBest(edgeNo);

		// check if edge i should be replaced
		if(newValue < oldValue || bestEdge.value > newValue)
		{
			// check if whole parent node must be replaced or only one edge
			auto newEdgeWithSameBeginning = newStats->GetBest(edgeNo, edges[i].GetType().first);
			if(newEdgeWithSameBeginning.value > newValue)
			{
				// replace single edge
				LOG4CPLUS_INFO_FMT(_logger, "Replace %s with %s",
						GetEncodingTypeString(edges[i].To()->GetEncodingType()).c_str(),
						GetEncodingTypeString(newEdgeWithSameBeginning.type.second).c_str());

				auto encFactory = DefaultEncodingFactory().Get(newEdgeWithSameBeginning.type.second, edges[i].To()->GetDataType());
				edges[i].To()->Reset(encFactory);
				Replace(edges[i].To(), newStats, 2*(edgeNo + 1));
			} else {
				// replace whole parent node
				LOG4CPLUS_INFO_FMT(_logger, "Replace %s with %s",
					GetEncodingTypeString(edges[i].From()->GetEncodingType()).c_str(),
					GetEncodingTypeString(bestEdge.type.first).c_str());

				auto encFactory = DefaultEncodingFactory().Get(bestEdge.type.first, edges[i].From()->GetDataType());
				edges[i].From()->Reset(encFactory);
				Replace(edges[i].From(), newStats, edgeNo%2 ? edgeNo-1 : edgeNo);
			}

			this->_ratio = this->_tree.GetCompressionRatio();
			this->_statistics = this->_tree.GetStatistics();
			return true;
		}
	}
	return false;
}

} /* namespace ddj */
