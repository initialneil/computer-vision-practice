#include <vector>
#include <cuda_runtime_api.h>
#include "cuda/cuda_utils.h"

using namespace std;

int calcAplusB(int a, int b)
{
	return a + b;
}

namespace device
{
	__global__ void cudaCalcAplusB_kernel(int N, int* __restrict__ darr_a, int* __restrict__ darr_b, int* __restrict__ darr_c)
	{
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= N)
			return;

		darr_c[idx] = darr_a[idx] + darr_b[idx];
	}
}

void cudaCalcAplusB(int N, int* darr_a, int* darr_b, int* darr_c, cudaStream_t stream)
{
	dim3 blk(512);
	dim3 grid(divUp(N, blk.x));
	device::cudaCalcAplusB_kernel << <grid, blk, 0, stream >> > (N, darr_a, darr_b, darr_c);
	cudaSafeCall(cudaGetLastError());
}
