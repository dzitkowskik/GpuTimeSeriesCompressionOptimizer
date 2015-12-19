/*
 *  compression_optimizer.hpp
 *
 *  Created on: 14/11/2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMPRESSION_OPTIMIZER_HPP_
#define COMPRESSION_OPTIMIZER_HPP_

#include "compression/encoding_type.hpp"
#include "compression/data_type.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "optimizer/optimal_tree.hpp"
#include "optimizer/path_generator.hpp"
#include "tree/compression_statistics.hpp"
#include <boost/noncopyable.hpp>
#include <gtest/gtest.h>

namespace ddj
{

class CompressionOptimizer : private boost::noncopyable
{
public:
	CompressionOptimizer()
		: _partsProcessed(0), _totalBytesProcessed(0), _maxTreeHeight(5)
	{
		_statistics = CompressionStatistics::make_shared(_maxTreeHeight);
	}
	~CompressionOptimizer(){}

public:
	CompressionTree OptimizeTree(SharedCudaPtr<char> data, DataType type);
	SharedCudaPtr<char> CompressData(SharedCudaPtr<char> dataPart, DataType type);

	SharedOptimalTreePtr GetOptimalTree() { return _optimalTree; }
	SharedCompressionStatisticsPtr GetStatistics() { return _statistics; }

private:
	bool IsFullUpdateNeeded();
	size_t GetSampleDataForFullUpdateSize(size_t partDataSize, DataType type);

	std::vector<PossibleTree> CrossTrees(
			PossibleTree parent,
			std::vector<PossibleTree> children,
			size_t inputSize,
			size_t parentMetadataSize);

	std::vector<PossibleTree> CrossTrees(
			PossibleTree parent,
			std::vector<PossibleTree> childrenLeft,
			std::vector<PossibleTree> childrenRight,
			size_t inputSize,
			size_t parentMetadataSize);

	std::vector<PossibleTree> FullStatisticsUpdate(
			SharedCudaPtr<char> data,
			EncodingType et,
			DataType dt,
			DataStatistics stats,
			int level);

private:
	SharedCompressionStatisticsPtr _statistics;
	SharedOptimalTreePtr _optimalTree;
	PathGenerator _pathGenerator;
	size_t _partsProcessed;
	size_t _totalBytesProcessed;
	int _maxTreeHeight;

private:
	// BENCHMARKS
	friend class CompressionOptimizerBenchmark;
	friend class CompressionOptimizerBenchmark_BM_CompressByBestTree_ChoosenInPhase1_RandomInt_Benchmark;
	friend class CompressionOptimizerBenchmark_BM_FullStatisticsUpdate_RawPhase1_RandomInt_Benchmark;
	friend class CompressionOptimizerBenchmark_BM_FullStatisticsUpdate_RawPhase1_Time_Benchmark;
	friend class CompressionOptimizerBenchmark_BM_UpdateStatistics_RawPhase2_RandomInt_Benchmark;
	friend class CompressionOptimizerBenchmark_BM_TryCorrectTree_RawPhase3_RandomInt_Benchmark;

	// UNIT TESTS
	friend class OptimizerTest;
	FRIEND_TEST(OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_Statistics);
	FRIEND_TEST(OptimizerTest, CompressionOptimizer_FullStatisticsUpdate_RandomInt_CompressByBestTree);
};

} /* namespace ddj */

#endif /* COMPRESSION_OPTIMIZER_HPP_ */
