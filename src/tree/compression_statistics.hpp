/*
 * compression_statistics.hpp
 *
 *  Created on: Dec 5, 2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_STATISTICS_HPP_
#define COMPRESSION_STATISTICS_HPP_

#include "compression/encoding_type.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <unordered_map>
#include <vector>
#include <utility>
#include <mutex>

namespace ddj
{

class CompressionStatistics;

using CompressionEdge = std::pair<EncodingType, EncodingType>;
using Stat = std::vector<std::unordered_map<CompressionEdge, double>>;
using SharedCompressionStatisticsPtr = boost::shared_ptr<CompressionStatistics>;

class CompressionStatistics
{
public:
	CompressionStatistics(int treeHeight);
	virtual ~CompressionStatistics();

public:
	void Update(int edgeNo, CompressionEdge edgeType, double compressionRatio);
	CompressionEdge GetBest(int edge);
	CompressionEdge GetBest(int edge, EncodingType beginningType);
	void Print();

public:
	static SharedCompressionStatisticsPtr make_shared(int treeHeight)
	{
		return boost::make_shared<CompressionStatistics>(treeHeight);
	}

private:
	Stat* _stat;
	int _height;
	std::mutex _mutex;
};

} /* namespace ddj */

#endif /* COMPRESSION_STATISTICS_HPP_ */
