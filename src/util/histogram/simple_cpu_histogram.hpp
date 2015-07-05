/*
 * basic_thrust_histogram.cuh 04-07-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_SIMPLE_CPU_HISTOGRAM_CUH_
#define DDJ_UTIL_SIMPLE_CPU_HISTOGRAM_CUH_

#include <vector>
#include <map>

namespace ddj {

class SimpleCpuHistogram
{
public:
    template<typename T>
    std::vector<int> Histogram(std::vector<T>& data)
    {
        std::map<T, int> histogram;

        for (auto&& elem : data)
            ++histogram[elem];

        std::vector<int> result(histogram.size());

        for (int i = 0; i < data.size(); i++)
            result[i] = histogram[data[i]];

        return result;
    }
};

} /* namespace ddj */
#endif /* DDJ_UTIL_BASIC_THRUST_HISTOGRAM_CUH_ */
