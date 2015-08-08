#include "compression/delta/delta_encoding.hpp"
#include "templates.hpp"
#include <benchmark/benchmark.h>

namespace ddj
{

static void BM_Delta_Float_Encode(benchmark::State& state)
{
    Random_Float_Encode_Template<DeltaEncoding>(state);
}
BENCHMARK(BM_Delta_Float_Encode)->Arg(1<<15)->Arg(1<<20)->Arg(1<<22);

static void BM_Delta_Float_Decode(benchmark::State& state)
{
    Random_Float_Decode_Template<DeltaEncoding>(state);
}
BENCHMARK(BM_Delta_Float_Decode)->Arg(1<<15)->Arg(1<<20)->Arg(1<<22);

static void BM_Delta_Int_Encode(benchmark::State& state)
{
    Random_Int_Encode_Template<DeltaEncoding>(state);
}
BENCHMARK(BM_Delta_Int_Encode)->Arg(1<<15)->Arg(1<<20)->Arg(1<<22);

static void BM_Delta_Int_Decode(benchmark::State& state)
{
    Random_Int_Decode_Template<DeltaEncoding>(state);
}
BENCHMARK(BM_Delta_Int_Decode)->Arg(1<<15)->Arg(1<<20)->Arg(1<<22);

} /* namespace ddj */
