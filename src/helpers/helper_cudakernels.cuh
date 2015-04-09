/*
 * helper_cudakernels.cuh
 *
 *  Created on: 09-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HELPER_CUDAKERNELS_CUH_
#define HELPER_CUDAKERNELS_CUH_

namespace ddj
{

class HelperCudaKernels
{
public:
	static void ModuloKernel(int* data, int size, int mod);
	static void ModuloThrust(int* data, int size, int mod);
};

} /* namespace ddj */
#endif /* HELPER_CUDAKERNELS_CUH_ */
