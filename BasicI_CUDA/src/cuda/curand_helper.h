/* cuRand helper.
*  All rights reserved. Prometheus 2020.
*  Contributor(s): Neil Z. Shao.
*/
#pragma once
#include "cuda_utils.h"
#include "DeviceBufferArray.h"
#include <curand_kernel.h>

namespace prometheus
{
// default block size for per-pixel init
#define CURAND_BLOCK_SIZE 4

	// generate
	namespace device 
	{
		__device__ float generateUniform(curandState &state);
	}

	///////////////////////////////////// random /////////////////////////////////////
	namespace device
	{
		static __device__ __forceinline__ float randomUniform(curandState& state)
		{
			//copy state to local mem
			curandState localState = state;

			//apply uniform distribution with calculated random
			float rndval = curand_uniform(&localState);
			//printf("[curand] val = %f\n", rndval);

			//update state
			state = localState;

			//return value
			return rndval;
		}

		static __device__ __forceinline__ int randomInt(int min_rand_int, int max_rand_int, curandState& curand_state)
		{
			// assume have already set up curand and generated state for each thread...
			// assume ranges vary by thread index
			float myrandf = randomUniform(curand_state);

			myrandf *= (max_rand_int - min_rand_int + 0.999999);
			myrandf += min_rand_int;
			return (int)truncf(myrandf);
		}

		static __device__ __forceinline__ float randomFloat(float min_val, float max_val, curandState& curand_state)
		{
			// assume have already set up curand and generated state for each thread...
			// assume ranges vary by thread index
			float myrandf = randomUniform(curand_state);

			myrandf *= (max_val - min_val);
			myrandf += min_val;
			return myrandf;
		}
	}

	// cuRand helper
	struct  cuRandHelper
	{
		// init N cuRand states
		void init(int N, cudaStream_t stream = 0);

		// generate a vector of float
		void generateUniform(pcl::gpu::DeviceArray<float> &rand_floats, int N, cudaStream_t stream = 0);
		void generateUniform(std::vector<float> &rand_floats, int N, cudaStream_t stream = 0);

		// cuRand states
		pcl::gpu::DeviceArray<curandState> states;
	};

}
