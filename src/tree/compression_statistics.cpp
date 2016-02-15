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
		this->_height = treeHeight;
		size_t size = GetEdgeNumber();
		this->_stat = new Stat(size);
	}

	CompressionStatistics::~CompressionStatistics()
	{
		std::lock_guard<std::mutex> guard(this->_mutex);
		delete _stat;
	}

	size_t CompressionStatistics::GetEdgeNumber()
	{
		return (1 << (this->_height+1)) - 2; // 2^(h+1) - 2
	}

	void CompressionStatistics::Update(int edgeNo, EdgeType edgeType, double compressionRatio)
	{
		 std::lock_guard<std::mutex> guard(this->_mutex);
		 if((*_stat)[edgeNo][edgeType] < 1.0) (*_stat)[edgeNo][edgeType] = 1.0;
		 if(compressionRatio < 1.0) compressionRatio = 1.0;
		 (*_stat)[edgeNo][edgeType] += compressionRatio;
		 (*_stat)[edgeNo][edgeType] /= 2.0;
//		 printf("START[%d] %s - %s ---> %f\n",
//				 edgeNo,
//				 GetEncodingTypeString(edgeType.first).c_str(),
//				 GetEncodingTypeString(edgeType.second).c_str(),
//				 compressionRatio);
//		 printf("Value = %f\n", (*_stat)[edgeNo][edgeType]);
//		 printf("END\n");
	}

	EdgeStatistic CompressionStatistics::GetAny(int edge)
	{
		 std::lock_guard<std::mutex> guard(this->_mutex);
		 EdgeType bestEdge;
		 double bestValue = 1;
		 for(auto& opt : (*_stat)[edge])
		 {
			 bestEdge = opt.first;
			 bestValue = opt.second;
		 }
		 return EdgeStatistic { bestEdge, bestValue };
	}

	EdgeStatistic CompressionStatistics::GetBest(int edge, EdgeType oldEdge, bool sameBeginning)
	{
		 std::lock_guard<std::mutex> guard(this->_mutex);

		 EdgeType bestEdge = std::make_pair(oldEdge.first, EncodingType::none);
		 double bestValue = 1;

		 for(auto& opt : (*_stat)[edge])
		 {
			 if(sameBeginning && opt.first.first != oldEdge.first)
				 continue;
			 if(opt.first.second != oldEdge.second && opt.second > bestValue)
			 {
				 bestEdge = opt.first;
				 bestValue = opt.second;
			 }
		 }

		 return EdgeStatistic { bestEdge, bestValue };
	}

	double CompressionStatistics::Get(int edgeNo, EdgeType edgeType)
	{
		return (*_stat)[edgeNo][edgeType];
	}

	void CompressionStatistics::Print(std::ostream& stream)
	{
		stream << "Compression tree statistics (tree height = " << _height << "):" << std::endl;
		for(int edgeNo = 0; edgeNo < _stat->size(); edgeNo++)
		{
			for(auto& edge : (*_stat)[edgeNo])
			{
				stream << "\t[" << edgeNo << "]:";
				stream << GetEncodingTypeString(edge.first.first).c_str();
				stream << " - ";
				stream << GetEncodingTypeString(edge.first.second).c_str();
				stream << " === " << edge.second << std::endl;
			}
		}
	}

	void CompressionStatistics::PrintShort(std::ostream& stream)
	{
		stream << "Compression tree statistics (tree height = " << _height << "):" << std::endl;
		for(int edgeNo = 0; edgeNo < _stat->size(); edgeNo++)
		{
			auto edge = GetBest(edgeNo, EdgeType{EncodingType::none, EncodingType::none});
			if(edge.value > 1.0)
			{
				stream << "\t[" << edgeNo << "]:";
				stream << GetEncodingTypeString(edge.type.first).c_str();
				stream << " - ";
				stream << GetEncodingTypeString(edge.type.second).c_str();
				stream << " === " << edge.value << std::endl;
			}
		}
	}

	std::string CompressionStatistics::ToString()
	{
		std::stringstream ss;
		Print(ss);
		return ss.str();
	}

	std::string CompressionStatistics::ToStringShort()
	{
		std::stringstream ss;
		PrintShort(ss);
		return ss.str();
	}

	void CompressionStatistics::Set(int edgeNo, EdgeType edgeType, double compressionRatio)
	{
		(*_stat)[edgeNo][edgeType] = compressionRatio;
	}

	SharedCompressionStatisticsPtr CompressionStatistics::Copy()
	{
		auto result = CompressionStatistics::make_shared(this->_height);
		for(int i = 0; i < _stat->size(); i++)
			for(auto& edge : (*_stat)[i])
				result->Set(i, edge.first, edge.second);
		return result;
	}

} /* namespace ddj */
