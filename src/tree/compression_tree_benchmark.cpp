/*
 *  compression_tree_benchmark.cpp
 *
 *  Created on: 11/11/2015
 *      Author: Karol Dzitkowski
 */

#include "benchmarks/compression_tree_benchmark_base.hpp"
#include "util/generator/cuda_array_generator.hpp"
#include "compression/default_encoding_factory.hpp"
#include "util/transform/cuda_array_transform.hpp"

namespace ddj
{

class CompressionTreeBenchmark : public CompressionTreeBenchmarkBase {};

BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDelta_Encoding_RandomInt)(benchmark::State& state)
{
    auto data = GetIntRandomData(state.range_x(), 10,1000);

    CompressionTree compressionTree;
	auto scale = boost::make_shared<CompressionNode>(DefaultEncodingFactory::Get(EncodingType::scale, DataType::d_int));
	auto delta = boost::make_shared<CompressionNode>(DefaultEncodingFactory::Get(EncodingType::delta, DataType::d_int));
	auto none = boost::make_shared<CompressionNode>(DefaultEncodingFactory::Get(EncodingType::none, DataType::d_int));
	delta->AddChild(none);
	scale->AddChild(delta);
	compressionTree.AddNode(scale, 0);

	Benchmark_Tree_Encoding(compressionTree, CastSharedCudaPtr<int,char>(data), DataType::d_int, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDelta_Encoding_RandomInt)->Arg(1<<20);


//		SCALE
//		  |
//	    DELTA
//	      |
//	    PATCH
//	   /	 \
//	  AFL	NONE
BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDeltaPatchAfl_Encoding_RandomInt)(benchmark::State& state)
{
	CudaDevice hc;
	int devId = hc.SetCudaDeviceWithMaxFreeMem();
	printf("TEST SET UP ON DEVICE %d\n", devId);

    auto data = GetIntRandomData(state.range_x(), 10,1000);

    CompressionTree compressionTree;
	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::lower);
	pef->factor = 0.2;
	auto root = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto delta = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto patch = boost::make_shared<CompressionNode>(boost::shared_ptr<PatchEncodingFactory<int>>(pef));
	auto afl = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto leaf1 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	auto leaf2 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	afl->AddChild(leaf1);
	patch->AddChild(afl);
	patch->AddChild(leaf2);
	delta->AddChild(patch);
	root->AddChild(delta);
	compressionTree.AddNode(root, 0);

	Benchmark_Tree_Encoding(compressionTree, CastSharedCudaPtr<int,char>(data), DataType::d_int, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDeltaPatchAfl_Encoding_RandomInt)->Arg(1<<20);

//		SCALE
//		  |
//	    DELTA
//	      |
//	    PATCH
//	   /	 \
//	  AFL	NONE
BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDeltaPatchAfl_Encoding_TimeFromFile)(benchmark::State& state)
{
	CudaDevice hc;
	int devId = hc.SetCudaDeviceWithMaxFreeMem();
	printf("TEST SET UP ON DEVICE %d\n", devId);

    auto data = CudaArrayTransform().Cast<time_t, int>(GetTsIntDataFromFile(state.range_x()));

    CompressionTree compressionTree;
	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::lower);
	pef->factor = 0.2;
	auto root = boost::make_shared<CompressionNode>(boost::make_shared<ScaleEncodingFactory>(DataType::d_int));
	auto delta = boost::make_shared<CompressionNode>(boost::make_shared<DeltaEncodingFactory>(DataType::d_int));
	auto patch = boost::make_shared<CompressionNode>(boost::shared_ptr<PatchEncodingFactory<int>>(pef));
	auto afl = boost::make_shared<CompressionNode>(boost::make_shared<AflEncodingFactory>(DataType::d_int));
	auto leaf1 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));
	auto leaf2 = boost::make_shared<CompressionNode>(boost::make_shared<NoneEncodingFactory>(DataType::d_int));

	afl->AddChild(leaf1);
	patch->AddChild(afl);
	patch->AddChild(leaf2);
	delta->AddChild(patch);
	root->AddChild(delta);
	compressionTree.AddNode(root, 0);

	Benchmark_Tree_Encoding(compressionTree, CastSharedCudaPtr<int,char>(data), DataType::d_int, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDeltaPatchAfl_Encoding_TimeFromFile)->Arg(1<<20);


} /* namespace ddj */


