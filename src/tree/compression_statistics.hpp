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
#include <iostream>
#include <string>

namespace ddj
{

class CompressionStatistics;

using EdgeType = std::pair<EncodingType, EncodingType>;
using Stat = std::vector<std::unordered_map<EdgeType, double>>;
using SharedCompressionStatisticsPtr = boost::shared_ptr<CompressionStatistics>;

struct EdgeStatistic
{
	EdgeType type;
	double value;
};

class CompressionStatistics
{
public:
	CompressionStatistics(int treeHeight);
	virtual ~CompressionStatistics();

public:
	void Update(int edgeNo, EdgeType edgeType, double compressionRatio);
	void Set(int edgeNo, EdgeType edgeType, double compressionRatio);

	double Get(int edgeNo, EdgeType edgeType);
	EdgeStatistic GetAny(int edge);
	EdgeStatistic GetBest(int edge,  EdgeType oldEdge, bool sameBeginning = false);
	void Print(std::ostream& stream = std::cout);
	void PrintShort(std::ostream& stream = std::cout);
	std::string ToString();
	std::string ToStringShort();
	SharedCompressionStatisticsPtr Copy();
	size_t GetEdgeNumber();

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
