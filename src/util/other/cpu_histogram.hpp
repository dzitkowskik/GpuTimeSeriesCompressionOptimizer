/*
 * cpu_histogram.hpp 04-07-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_OTHER_CPU_HISTOGRAM_CUH_
#define DDJ_UTIL_OTHER_CPU_HISTOGRAM_CUH_

#include <vector>
#include <map>
#include <algorithm>

namespace ddj {

class CpuHistogramSparse
{
public:
    template<typename T>
    std::map<T, int> Histogram(std::vector<T>& data)
    {
        std::map<T, int> histogram;

        for (auto&& elem : data)
            ++histogram[elem];

        return histogram;
    }
};

class CpuHistogramDense
{
public:
	template<typename T>
	std::map<T, int> Histogram(std::vector<T>& data)
	{
		std::map<T, int> histogram;

		auto minMax = std::minmax_element(data.begin(), data.end());
		T min = *(minMax.first);
		T max = *(minMax.second);

		for(T index = min; index <= max; index++)
			histogram.insert(std::make_pair(index, 0));

		for (auto&& elem : data)
			++histogram[elem];

		return histogram;
	}
};

} /* namespace ddj */
#endif /* DDJ_UTIL_OTHER_CPU_HISTOGRAM_CUH_ */
