/* cuRand helper.
*  All rights reserved. Prometheus 2020.
*  Contributor(s): Neil Z. Shao.
*/
#include "curand_helper.h"
#include "cuda/data_transfer.h"

namespace prometheus
{
	namespace device
	{
		static __device__ float generateUniform(curandState &state)
		{
			//copy state to local mem
			curandState localState = state;

			//apply uniform distribution with calculated random
			float rndval = curand_uniform(&localState);

			//update state
			state = localState;

			//return value
			return rndval;
		}

		static __global__ void init_cuRand_kernel(
			pcl::gpu::PtrSz<curandState> states, int N, unsigned long seed)
		{
			int idx = blockIdx.x*blockDim.x + threadIdx.x;
			if (idx >= N)
				return;

			curand_init(seed, idx, 0, &states[idx]);
		}

		static __global__ void generateUniform_kernel(
			pcl::gpu::PtrSz<curandState> states, int N,
			pcl::gpu::PtrSz<float> rand_floats)
		{
			int idx = blockIdx.x*blockDim.x + threadIdx.x;
			if (idx >= N)
				return;

			rand_floats[idx] = generateUniform(states[idx]);
		}

	}

	// init N cuRand states
	void cuRandHelper::init(int N, cudaStream_t stream)
	{
		if (this->states.size() >= N) {
			return;
		}

		// init states
		int num = N * 1.5;
		this->states.create(num);

		srand(time(NULL));
		unsigned long seed = unsigned(time(NULL));

		dim3 blk(512);
		dim3 grid(divUp(num, blk.x));
		device::init_cuRand_kernel << <grid, blk, 0, stream >> > (
			this->states, num, seed);
		cudaSafeCall(cudaGetLastError());
	}

	// generate a vector of float
	void cuRandHelper::generateUniform(pcl::gpu::DeviceArray<float> &rand_floats, int N, cudaStream_t stream)
	{
		this->init(N, stream);
		rand_floats.create(N);

		dim3 blk(512);
		dim3 grid(divUp(N, blk.x));
		device::generateUniform_kernel << <grid, blk, 0, stream >> > (
			this->states, N,
			rand_floats);
		cudaSafeCall(cudaGetLastError());
	}

	void cuRandHelper::generateUniform(std::vector<float> &rand_floats, int N, cudaStream_t stream)
	{
		pcl::gpu::DeviceArray<float> d_rand_floats;
		generateUniform(d_rand_floats, N, stream);
		surfelwarp::downloadDeviceArray(d_rand_floats, rand_floats, stream);
		cudaSafeCall(cudaGetLastError());
	}

}
