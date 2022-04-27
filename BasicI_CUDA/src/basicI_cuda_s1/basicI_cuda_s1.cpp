#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime_api.h>

#pragma comment(lib, "cudart.lib")

namespace prometheus
{
	int calcAplusB(int a, int b);
}

using namespace std;
using namespace prometheus;

static default_random_engine e(time(0));

static void makeRandomVector(int N, int max_val, std::vector<int>& harr)
{
	uniform_real_distribution<float> u(0, max_val);

	harr.resize(N);
	for (int i = 0; i < N; ++i)
		harr[i] = u(e);
}

///////////////////////////////////// cuda func /////////////////////////////////////
static void cudaAllocMemory(int** dptr_arr, int num_bytes)
{
	cudaMalloc((void**)dptr_arr, num_bytes);
}

static void cudaCopyHostToDevice(const std::vector<int>& harr, int* dptr_arr, cudaStream_t stream = 0)
{
	// sync
	//cudaMemcpy(dptr_arr, harr.data(), sizeof(int) * harr.size(), cudaMemcpyHostToDevice);

	// async
	cudaMemcpyAsync(dptr_arr, harr.data(), sizeof(int) * harr.size(), cudaMemcpyHostToDevice, stream);
}

static void cudaCopyDeviceToHost(int N, int* dptr_arr, std::vector<int>& harr, cudaStream_t stream = 0)
{
	harr.resize(N);
	cudaMemcpyAsync(harr.data(), dptr_arr, sizeof(int) * harr.size(), cudaMemcpyDeviceToHost, stream);
}

void cudaCalcAplusB(int N, int* darr_a, int* darr_b, int* darr_c, cudaStream_t stream = 0);

///////////////////////////////////// main /////////////////////////////////////
int main(int argc, char** argv)
{
	const int N = 10, MAX_VAL = 10;
	vector<int> harr_a, harr_b;
	makeRandomVector(N, MAX_VAL, harr_a);
	makeRandomVector(N, MAX_VAL, harr_b);

	printf("A = [ ");
	for (int i = 0; i < N; ++i)
		printf("%d ", harr_a[i]);
	printf("]\n");

	printf("B = [ ");
	for (int i = 0; i < N; ++i)
		printf("%d ", harr_b[i]);
	printf("]\n");

	// alloc gpu memory and copy to cuda
	int* darr_a = nullptr, * darr_b = nullptr;
	cudaAllocMemory(&darr_a, sizeof(int) * N);
	cudaCopyHostToDevice(harr_a, darr_a);
	cudaAllocMemory(&darr_b, sizeof(int) * N);
	cudaCopyHostToDevice(harr_b, darr_b);

	int* darr_c = nullptr;
	cudaAllocMemory(&darr_c, sizeof(int) * N);

	// calc a+b on cuda
	cudaCalcAplusB(N, darr_a, darr_b, darr_c);

	// copy back to cpu
	vector<int> harr_c(N);
	cudaCopyDeviceToHost(N, darr_c, harr_c);

	printf("C = [ ");
	for (int i = 0; i < N; ++i)
		printf("%d ", harr_c[i]);
	printf("]\n");

	cudaFree(darr_a);
	cudaFree(darr_b);
	cudaFree(darr_c);

	return 0;
}
