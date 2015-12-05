/*
 * compression_statistics.cpp
 *
 *  Created on: Dec 5, 2015
 *      Author: Karol Dzitkowski
 */

#include <tree/compression_statistics.hpp>

namespace ddj
{
	CompressionStatistics::CompressionStatistics(int treeHeight)
	{
		std::lock_guard<std::mutex> guard(this->_mutex);
		size_t size = (1 << (treeHeight+1)) - 2; // 2^(h+1) - 2
		_stat = new Stat(size);
		_height = treeHeight;
	}

	CompressionStatistics::~CompressionStatistics()
	{
		std::lock_guard<std::mutex> guard(this->_mutex);
		delete _stat;
	}

	void CompressionStatistics::Update(int edgeNo, CompressionEdge edgeType, double compressionRatio)
	{
		 std::lock_guard<std::mutex> guard(this->_mutex);
		 if((*_stat)[edgeNo][edgeType] < 1.0) (*_stat)[edgeNo][edgeType] = 1.0;
		 if(compressionRatio < 1.0) compressionRatio = 1.0;
		 (*_stat)[edgeNo][edgeType] += compressionRatio;
		 (*_stat)[edgeNo][edgeType] /= 2.0;
		 printf("START[%d] %s - %s ---> %f\n",
				 edgeNo,
				 GetEncodingTypeString(edgeType.first).c_str(),
				 GetEncodingTypeString(edgeType.second).c_str(),
				 compressionRatio);
		 printf("Value = %f\n", (*_stat)[edgeNo][edgeType]);
		 printf("END\n");
	}

	CompressionEdge CompressionStatistics::GetAny(int edge)
		{
			 std::lock_guard<std::mutex> guard(this->_mutex);
			 CompressionEdge bestEdge;
			 for(auto& opt : (*_stat)[edge])
				 bestEdge = opt.first;
			 return bestEdge;
		}

	CompressionEdge CompressionStatistics::GetBest(int edge)
	{
		 std::lock_guard<std::mutex> guard(this->_mutex);

		 CompressionEdge bestEdge;
		 double bestValue = 1;

		 for(auto& opt : (*_stat)[edge])
		 {
			 if(opt.second >= bestValue)
			 {
				 bestEdge = opt.first;
				 bestValue = opt.second;
			 }
		 }

		 return bestEdge;
	}

	CompressionEdge CompressionStatistics::GetBest(int edge, EncodingType beginningType)
	{
		 std::lock_guard<std::mutex> guard(this->_mutex);

		 CompressionEdge bestEdge = std::make_pair(beginningType, EncodingType::none);
		 double bestValue = 1;

		 for(auto& opt : (*_stat)[edge])
		 {
			 if(opt.first.first == beginningType && opt.second > bestValue)
			 {
				 bestEdge = opt.first;
				 bestValue = opt.second;
			 }
		 }

		 return bestEdge;
	}

	double CompressionStatistics::Get(int edgeNo, CompressionEdge edgeType)
	{
		return (*_stat)[edgeNo][edgeType];
	}

	void CompressionStatistics::Print()
	{
		printf("Compression tree statistics (tree height = %d)\n", _height);
		for(int edgeNo = 0; edgeNo < _stat->size(); edgeNo++)
		{
			for(auto& edge : (*_stat)[edgeNo])
			{
				printf("[%d]: %s - %s  ===  %lf\n",
						edgeNo,
						GetEncodingTypeString(edge.first.first).c_str(),
						GetEncodingTypeString(edge.first.second).c_str(),
						edge.second);
			}
		}
	}

	void CompressionStatistics::PrintShort()
	{
		printf("Compression tree short statistics (tree height = %d)\n", _height);
		for(int edgeNo = 0; edgeNo < _stat->size(); edgeNo++)
		{
			auto edge = GetBest(edgeNo);
			auto edgeStat = Get(edgeNo, edge);
			if(edgeStat > 1.0)
				printf("[%d]: %s - %s  ===  %lf\n",
					edgeNo,
					GetEncodingTypeString(edge.first).c_str(),
					GetEncodingTypeString(edge.second).c_str(),
					edgeStat);
		}
	}

} /* namespace ddj */
