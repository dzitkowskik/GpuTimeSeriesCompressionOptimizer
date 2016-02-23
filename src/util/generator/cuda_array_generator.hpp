/*
 * cuda_array_generator.hpp 26-03-2015 Karol Dzitkowski
 */

#ifndef DDJ_UTIL_CUDA_ARRAY_GENERATOR_HPP_
#define DDJ_UTIL_CUDA_ARRAY_GENERATOR_HPP_

#include "core/cuda_ptr.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include "core/execution_policy.hpp"

#include <boost/noncopyable.hpp>
#include <curand.h>

namespace ddj
{

class CudaArrayGenerator : private boost::noncopyable
{
public:
    CudaArrayGenerator();
    ~CudaArrayGenerator();

public:
    // TODO: translate to templates
    SharedCudaPtr<float> GenerateRandomFloatDeviceArray(int size);
    SharedCudaPtr<double> GenerateRandomDoubleDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomIntDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomIntDeviceArray(int size, int from, int to);
    SharedCudaPtr<int> GenerateConsecutiveIntDeviceArray(int size);
    SharedCudaPtr<int> GenerateRandomStencil(int size);
    template<typename T> SharedCudaPtr<T> CreateConsecutiveNumbersArray(int size, T start);
    template<typename T> SharedCudaPtr<T> CreateConsecutiveNumbersArray(int size, T start, T step);
    SharedCudaPtr<float> CreateRandomFloatsWithMaxPrecision(int size, int maxPrecision);

	// pattern suitable for time (monotonically increasing function)
	//                              ------------
	//                        ------
	//          --------------
	//       ---
	// ------											min
	SharedCudaPtr<time_t> GetFakeDataForTime(
			time_t min=0,
			double flatness=0.5,
			size_t size=1e6);


	// *		*		*		*		*	  *			max
	//					 -------
	//			 -------		 ------
	//	-------							------			min
	//  <-len->
	template<typename T>
	SharedCudaPtr<T> GetFakeDataWithPatternA(
			int part=0,
			size_t len=1e3,
			T step=10,
			T min=0,
			T max=1e6,
			size_t size=1e6);

	//     pattern 1	|		pattern2		|	pattern1
	//	* * * * * * * *	*                       * * * * * * * *		max-rand(0,5)
	//	 # # # # # # #	  *                   *  # # # # # # # 		max-rand(0,5)
	//						*               *
	//						  *           *
	//							*       *
	//  * * * * * * * *			  *   *			 * * * * * * * *	min+rand(0,5)
	//	 # # # # # # #				*			  # # # # # # # 	min+rand(0,5)
	//  <------len-----> <---------len---------> <-----len------>
	template<typename T>
	SharedCudaPtr<T> GetFakeDataWithPatternB(
			int part=0,
			size_t len=2*1e6,
			T min=-1e5,
			T max=+1e5,
			size_t size=1e6);

	// *	*	*	*	*	**	*	***	*	  **		max+-(rand(0,32))
	//					 -------
	//			 -------		 ------
	//	-------							------			min
	//  <-len->
	// outProb - chances for outliers "*"
	template<typename T>
	SharedCudaPtr<T> GetFakeDataWithOutliers(
			int part=0,
			size_t len=1e3,
			T step=10,
			T min=0,
			T max=1e6,
			double outProb=0.01,
			size_t size=1e6);

private:
    curandGenerator_t _gen;
    CudaArrayTransform _transform;
    ExecutionPolicy _policy;
};

} /* namespace ddj */
#endif /* DDJ_UTIL_CUDA_ARRAY_GENERATOR_HPP_ */
