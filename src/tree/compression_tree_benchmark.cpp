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
#include "optimizer/path_generator.hpp"

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
	compressionTree.Print();

	Benchmark_Tree_Encoding(compressionTree, CastSharedCudaPtr<int,char>(data), DataType::d_int, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDelta_Encoding_RandomInt)
	->Arg(1<<18)
	->Arg(1<<19)
	->Arg(1<<20)
	->Arg(1<<21)
	->Arg(1<<22)
	->Arg(1<<23)
	->Arg(1<<24);


//		SCALE
//		  |
//	    DELTA
//	      |
//	    PATCH
//	   /	 \
//	  AFL	NONE
BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDeltaPatchAfl_Encoding_RandomInt)(benchmark::State& state)
{
    auto data = GetIntRandomData(state.range_x(), 10,1000);

    CompressionTree compressionTree;
	auto pef = new PatchEncodingFactory<int>(DataType::d_int, PatchType::lower);
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
	compressionTree.Print();

	Benchmark_Tree_Encoding(compressionTree, CastSharedCudaPtr<int,char>(data), DataType::d_int, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_ScaleDeltaPatchAfl_Encoding_RandomInt)
	->Arg(1<<18)
	->Arg(1<<19)
	->Arg(1<<20)
	->Arg(1<<21)
	->Arg(1<<22)
	->Arg(1<<23)
	->Arg(1<<24);

//			 DICT
//			/	 \
//	   SCALE     DELTA
//		|			|
//	  DELTA		FLOAT_TO_INT
//		|		    |
//	   GFC	       AFL
BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_FakeDataPatternA_Float_TreeWith_Dict_And_FloatToInt)(benchmark::State& state)
{
	Path path
	{
		EncodingType::dict,
			EncodingType::scale, EncodingType::delta,
				EncodingType::gfc, EncodingType::none, EncodingType::none,
			EncodingType::delta, EncodingType::floatToInt,
				EncodingType::afl, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_float);
	auto data = GetFakeDataWithPatternA<float>(0, state.range_x());
	tree.Print();

	Benchmark_Tree_Encoding(tree, CastSharedCudaPtr<float,char>(data), DataType::d_float, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_FakeDataPatternA_Float_TreeWith_Dict_And_FloatToInt)
	->Arg(1<<18)
	->Arg(1<<19)
	->Arg(1<<20)
	->Arg(1<<21)
	->Arg(1<<22)
	->Arg(1<<23)
	->Arg(1<<24);

//			PATCH
//			/	\
//		CONST	FloatToInt
//					|
//				   RLE
//				  /   \
//			  CONST   SCALE
//						|
//					   AFL
BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_FakeDataPatternA_Float_GoodTree)(benchmark::State& state)
{
	Path path
	{
		EncodingType::patch,
			EncodingType::constData, EncodingType::none,
			EncodingType::floatToInt,
				EncodingType::rle,
					EncodingType::constData, EncodingType::none,
					EncodingType::scale, EncodingType::afl, EncodingType::none
	};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_float);
	auto data = GetFakeDataWithPatternA<float>(0, state.range_x());
	tree.Print();

	Benchmark_Tree_Encoding(tree, CastSharedCudaPtr<float,char>(data), DataType::d_float, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_FakeDataPatternA_Float_GoodTree)
	->Arg(1<<18)
	->Arg(1<<19)
	->Arg(1<<20)
	->Arg(1<<21)
	->Arg(1<<22)
	->Arg(1<<23)
	->Arg(1<<24);

// GFC - 2xNONE
BENCHMARK_DEFINE_F(CompressionTreeBenchmark, BM_CompressionTree_GFC)(benchmark::State& state)
{
	Path path {
			EncodingType::gfc, EncodingType::none, EncodingType::none
		};
	auto tree = PathGenerator().GenerateTree(path, DataType::d_float);
	auto data = GetFakeDataWithPatternA<float>(0, state.range_x());
	tree.Print();

	Benchmark_Tree_Encoding(tree, CastSharedCudaPtr<float,char>(data), DataType::d_float, state);
}
BENCHMARK_REGISTER_F(CompressionTreeBenchmark, BM_CompressionTree_GFC)
	->Arg(1<<18)
	->Arg(1<<19)
	->Arg(1<<20)
	->Arg(1<<21)
	->Arg(1<<22)
	->Arg(1<<23)
	->Arg(1<<24);

} /* namespace ddj */


