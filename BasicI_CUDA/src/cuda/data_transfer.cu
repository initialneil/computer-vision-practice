/* Undistort camera
*  All rights reserved. Prometheus 2019.
*  Contributor(s): Neil Z. Shao
*/
#include "data_transfer.h"
#include "cuda/common_texture_utils.h"

namespace surfelwarp
{
	using pcl::gpu::DeviceArray;
	using pcl::gpu::DeviceArray2D;
	using pcl::gpu::PtrSz;
	using pcl::gpu::PtrStep;
	using pcl::gpu::PtrStepSz;

	namespace device {
		template<typename T>
		__global__ void textureToMap2DKernel(
			cudaTextureObject_t texture,
			PtrStepSz<T> map
		) {
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= map.cols || y >= map.rows) return;
			T element = readTexture<T>(texture, x, y);
			map.ptr(y)[x] = element;
		}
	}

	template<typename T>
	void textureToMap2D(
		cudaTextureObject_t texture,
		DeviceArray2D<T>& map,
		cudaStream_t stream
	) {
		dim3 blk(16, 16);
		dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
		device::textureToMap2DKernel<T> << <grid, blk, 0, stream >> > (texture, map);
	}

	///////////////////////////////////// func /////////////////////////////////////
	// down uchar1
	void downloadCudaTextureUchar1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<unsigned char>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<unsigned char>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureUchar1(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<unsigned char> d_mat;
		d_mat.create(height, width);
		textureToMap2D<unsigned char>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_8U);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayUchar1(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_8UC1);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(uchar), height,
			cudaMemcpyDeviceToHost, stream);
	}

	// down uchar4
	void downloadCudaTextureUchar4(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<uchar4>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<uchar4>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureUchar4(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<uchar4> d_mat;
		d_mat.create(height, width);
		textureToMap2D<uchar4>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_8UC4);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayUchar4(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_8UC4);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(uchar4), height,
			cudaMemcpyDeviceToHost, stream);
	}

	// down ushort1
	void downloadCudaTextureUshort1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<ushort>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<ushort>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureUshort1(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<ushort> d_mat;
		d_mat.create(height, width);
		textureToMap2D<ushort>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_16U);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayUshort1(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_16UC1);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(ushort), height,
			cudaMemcpyDeviceToHost, stream);
	}

	// down float1
	void downloadCudaTextureFloat1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<float>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<float>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureFloat1(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<float> d_mat;
		d_mat.create(height, width);
		textureToMap2D<float>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_32F);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayFloat1(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_32FC1);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(float), height,
			cudaMemcpyDeviceToHost, stream);
	}

	// down float2
	void downloadCudaTextureFloat2(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<float2>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<float2>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureFloat2(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<float2> d_mat;
		d_mat.create(height, width);
		textureToMap2D<float2>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_32FC2);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayFloat2(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_32FC2);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(float2), height,
			cudaMemcpyDeviceToHost, stream);
	}

	// down float4
	void downloadCudaTextureFloat4(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<float4>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<float4>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureFloat4(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<float4> d_mat;
		d_mat.create(height, width);
		textureToMap2D<float4>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_32FC4);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayFloat4(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_32FC4);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(float4), height,
			cudaMemcpyDeviceToHost, stream);
	}

	// down int1
	void downloadCudaTextureInt1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<int>& d_mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		d_mat.create(height, width);
		textureToMap2D<int>(t_dat, d_mat, stream);
	}

	void downloadCudaTextureInt1(cudaTextureObject_t t_dat, cv::Mat& mat,
		cudaStream_t stream)
	{
		//Query the size of texture
		unsigned width = 0, height = 0;
		query2DTextureExtent(t_dat, width, height);

		//Download it to device
		DeviceArray2D<int> d_mat;
		d_mat.create(height, width);
		textureToMap2D<int>(t_dat, d_mat, stream);

		//Download it to host
		mat.create(height, width, CV_32S);
		d_mat.download(mat.data, mat.step);
	}

	void downloadCudaArrayInt1(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream)
	{
		mat.create(height, width, CV_32S);
		if (width == 0 || height == 0)
			return;

		cudaMemcpy2DFromArrayAsync(mat.data, mat.step,
			d_array, 0, 0, width * sizeof(int), height,
			cudaMemcpyDeviceToHost, stream);
	}

	///////////////////////////////////// download 3d /////////////////////////////////////
	namespace device
	{
		template <typename T>
		__global__ void cudaReadRowOfTexture3D_kernel(cudaTextureObject_t t_src, int width, int height, int depth,
			int row_idx, cudaSurfaceObject_t s_slice)
		{
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= width || y >= depth)
				return;

			T val = tex3D<T>(t_src, x, row_idx, y);
			surf2Dwrite<T>(val, s_slice, x * sizeof(T), y);
		}
	}

	void cudaReadRowOfTexture3D(cudaTextureObject_t t_src, int width, int height, int depth, int row_idx,
		cudaSurfaceObject_t s_slice, cudaStream_t stream)
	{
		// the output is the row slice of source texture3d
		// the output height is the source depth
		// the output width is the source width
		dim3 blk(32, 32);
		dim3 grid(divUp(width, blk.x), divUp(depth, blk.y));
		device::cudaReadRowOfTexture3D_kernel<float> << <grid, blk, 0, stream >> > (t_src, width, height, depth,
			row_idx, s_slice);
	}

	///////////////////////////////////// upload /////////////////////////////////////
	namespace device {
		template<typename T>
		__global__ void map2DToSurface_kernel(
			PtrStepSz<T> map,
			cudaSurfaceObject_t surface
		) {
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= map.cols || y >= map.rows) return;
			T element = map(y, x);
			surf2Dwrite(element, surface, x * sizeof(T), y);
		}
	}

	template<typename T>
	void map2DToSurface(
		DeviceArray2D<T>& map,
		cudaSurfaceObject_t surface,
		cudaStream_t stream
	) {
		dim3 blk(16, 16);
		dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
		device::map2DToSurface_kernel<T> << <grid, blk, 0, stream >> > (map, surface);
	}

	// upload uchar4
	void uploadCudaSurfaceUchar4(pcl::gpu::DeviceArray2D<uchar4>& d_mat, cudaSurfaceObject_t s_dat,
		cudaStream_t stream)
	{
		//Query the size of d_mat
		unsigned width = d_mat.cols(), height = d_mat.rows();

		//write it to surface
		map2DToSurface<uchar4>(d_mat, s_dat, stream);
	}

	///////////////////////////////////// copy ///////////////////////////////////////
	namespace device {
		template<typename T>
		__global__ void cudaCopyTexture2D_kernel(
			cudaTextureObject_t texture,
			cudaSurfaceObject_t surface,
			int width, int height
		) {
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= width || y >= height) return;

			T element = readTexture<T>(texture, x, y);
			surf2Dwrite(element, surface, x * sizeof(T), y);
		}
	}

	template<typename T>
	void cudaCopyTexture2D(
		cudaTextureObject_t texture,
		cudaSurfaceObject_t surface,
		int width, int height,
		cudaStream_t stream
	) {
		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaCopyTexture2D_kernel<T> << <grid, blk, 0, stream >> > (texture, surface, width, height);
	}

	// texture to surface
	void cudaCopyTexture2DUchar1(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream)
	{
		cudaCopyTexture2D<uchar1>(t_src, s_dst, width, height, stream);
	}

	void cudaCopyTexture2DUchar4(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream)
	{
		cudaCopyTexture2D<uchar4>(t_src, s_dst, width, height, stream);
	}

	void cudaCopyTexture2DFloat1(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream)
	{
		cudaCopyTexture2D<float>(t_src, s_dst, width, height, stream);
	}

	void cudaCopyTexture2DFloat2(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream)
	{
		cudaCopyTexture2D<float2>(t_src, s_dst, width, height, stream);
	}

	void cudaCopyTexture2DFloat4(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream)
	{
		cudaCopyTexture2D<float4>(t_src, s_dst, width, height, stream);
	}

	void cudaCopyTexture2DInt1(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream)
	{
		cudaCopyTexture2D<int>(t_src, s_dst, width, height, stream);
	}

	// device2d to surface
	namespace device
	{
		template <typename T0, typename T1>
		__global__ void cudaCopyDeviceArray2DToTexture_kernel(pcl::gpu::PtrStepSz<T0> d_arr2d,
			cudaSurfaceObject_t s_map, int width, int height,
			float alpha, float gamma)
		{
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= width || y >= height)
				return;

			float value = d_arr2d(y, x) * alpha + gamma;
			T1 element = value;
			surf2Dwrite(element, s_map, x * sizeof(T1), y);
		}
	}

	template <typename T0, typename T1>
	void cudaCopyDeviceArray2DToTextureT(const pcl::gpu::DeviceArray2D<T0>& d_arr2d, cudaSurfaceObject_t s_map, int width, int height,
		float alpha, float gamma, cudaStream_t stream)
	{
		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaCopyDeviceArray2DToTexture_kernel<T0, T1> << <grid, blk, 0, stream >> > (d_arr2d, s_map, width, height, alpha, gamma);
	}

	void cudaCopyDeviceArray2DToTexture(const pcl::gpu::DeviceArray2D<int>& d_arr2d, prometheus::Texture2DFloat1& t_map,
		float alpha, float gamma, cudaStream_t stream)
	{
		t_map.create(d_arr2d.rows(), d_arr2d.cols());
		cudaCopyDeviceArray2DToTextureT<int, float>(d_arr2d, t_map.surface(), d_arr2d.cols(), d_arr2d.rows(), alpha, gamma, stream);
		cudaSafeCall(cudaGetLastError());
	}

	// texture to cuda array
	void cudaCopyTexture2DUchar1(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream)
	{
		CudaTextureSurface collect;
		collect.d_array = darr_dst;
		bind2DTextureSurface(collect);
		cudaCopyTexture2DUchar1(t_src, collect.surface, width, height, stream);

		collect.d_array = 0;
		releaseTextureCollect(collect);
	}

	void cudaCopyTexture2DUchar4(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream)
	{
		CudaTextureSurface collect;
		collect.d_array = darr_dst;
		bind2DTextureSurface(collect);
		cudaCopyTexture2DUchar4(t_src, collect.surface, width, height, stream);

		collect.d_array = 0;
		releaseTextureCollect(collect);
	}

	void cudaCopyTexture2DFloat1(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream)
	{
		CudaTextureSurface collect;
		collect.d_array = darr_dst;
		bind2DTextureSurface(collect);
		cudaCopyTexture2DFloat1(t_src, collect.surface, width, height, stream);

		collect.d_array = 0;
		releaseTextureCollect(collect);
	}

	void cudaCopyTexture2DInt1(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream)
	{
		CudaTextureSurface collect;
		collect.d_array = darr_dst;
		bind2DTextureSurface(collect);
		cudaCopyTexture2DInt1(t_src, collect.surface, width, height, stream);

		collect.d_array = 0;
		releaseTextureCollect(collect);
	}

	///////////////////////////////////// opencv cuda /////////////////////////////////////
	namespace device
	{
		__global__ void cudaSplitUchar4Mask_kernel(pcl::gpu::PtrStepSz<uchar4> src,
			cv::cuda::PtrStep<uchar3> dst_clr, cv::cuda::PtrStep<uchar> dst_mask)
		{
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= src.cols || y >= src.rows) return;

			uchar4 clrw = src(y, x);
			dst_clr(y, x) = make_uchar3(clrw.x, clrw.y, clrw.z);
			dst_mask(y, x) = uchar(clrw.w);
		}

		__global__ void cudaMergeUchar4Mask_kernel(const cv::cuda::PtrStepSz<uchar3> src_clr,
			const cv::cuda::PtrStepSz<uchar> src_mask,
			cudaSurfaceObject_t s_frame)
		{
			const auto x = threadIdx.x + blockDim.x * blockIdx.x;
			const auto y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= src_clr.cols || y >= src_clr.rows) return;

			uchar3 clr = src_clr(y, x);
			uchar mask = src_mask(y, x);

			uchar4 clrw = make_uchar4(clr.x, clr.y, clr.z, mask);
			surf2Dwrite(clrw, s_frame, x * sizeof(uchar4), y);
		}

	}

	void cudaSplitUchar4Mask(const pcl::gpu::DeviceArray2D<uchar4>& d_mat,
		cv::cuda::GpuMat& d_clr, cv::cuda::GpuMat& d_mask, cudaStream_t stream)
	{
		int width = d_mat.cols();
		int height = d_mat.rows();
		d_clr.create(height, width, CV_8UC3);
		d_mask.create(height, width, CV_8U);

		dim3 blk(32, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaSplitUchar4Mask_kernel << <grid, blk, 0, stream >> > (d_mat, d_clr, d_mask);
	}

	void cudaMergeUchar4Mask(const cv::cuda::GpuMat& d_clr, const cv::cuda::GpuMat& d_mask,
		cudaSurfaceObject_t s_frame, cudaStream_t stream)
	{
		int width = d_clr.cols;
		int height = d_clr.rows;

		dim3 blk(32, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaMergeUchar4Mask_kernel << <grid, blk, 0, stream >> > (d_clr, d_mask, s_frame);
	}

	///////////////////////////////////// set value /////////////////////////////////////
	namespace device
	{
		template <typename T>
		__global__ void cudaSetValue_kernel(T* darr_ptr, int num, T value)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < num) {
				darr_ptr[idx] = value;
			}
		}

		template <typename T>
		__global__ void cudaSetValue_kernel(cudaSurfaceObject_t surface, int width, int height, T value)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x < width && y < height) {
				surf2Dwrite(value, surface, x * sizeof(T), y);
			}
		}

		template <typename T>
		__global__ void cudaSurface3DSetValue_kernel(
			cudaSurfaceObject_t surface, int width, int height, int depth, T value)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x < width && y < height) {
				for (int i = 0; i < depth; ++i) {
					surf3Dwrite(value, surface, x * sizeof(T), y, i);
				}
			}
		}
	}

	void cudaSetValue(uchar* darr_ptr, int num, uchar value, cudaStream_t stream)
	{
		if (num <= 0)
			return;

		dim3 blk(512);
		dim3 grid(divUp(num, blk.x));
		device::cudaSetValue_kernel<uchar> << <grid, blk, 0, stream >> > (darr_ptr, num, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(uchar3* darr_ptr, int num, uchar3 value, cudaStream_t stream)
	{
		if (num <= 0)
			return;

		dim3 blk(512);
		dim3 grid(divUp(num, blk.x));
		device::cudaSetValue_kernel<uchar3> << <grid, blk, 0, stream >> > (darr_ptr, num, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(uchar4* darr_ptr, int num, uchar4 value, cudaStream_t stream)
	{
		if (num <= 0)
			return;

		dim3 blk(512);
		dim3 grid(divUp(num, blk.x));
		device::cudaSetValue_kernel<uchar4> << <grid, blk, 0, stream >> > (darr_ptr, num, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(cudaSurfaceObject_t s_color, int width, int height, float value, cudaStream_t stream)
	{
		if (width <= 0 || height <= 0)
			return;

		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaSetValue_kernel << <grid, blk, 0, stream >> > (s_color, width, height, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(cudaSurfaceObject_t s_color, int width, int height, uchar4 value, cudaStream_t stream)
	{
		if (width <= 0 || height <= 0)
			return;

		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaSetValue_kernel << <grid, blk, 0, stream >> > (s_color, width, height, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSurface3DSetValue(cudaSurfaceObject_t s_volume, int width, int height, int depth,
		float value, cudaStream_t stream)
	{
		if (width <= 0 || height <= 0)
			return;

		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaSurface3DSetValue_kernel << <grid, blk, 0, stream >> > (
			s_volume, width, height, depth, value);
		cudaSafeCall(cudaGetLastError());
	}

	///////////////////////////////////// set surface to value /////////////////////////////////////
	namespace device
	{
		template <typename T>
		__global__ void cudaSetValue_kernel(cudaSurfaceObject_t surface, int width, int height, T value, cudaTextureObject_t t_mask)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x < width && y < height) {
				if (t_mask != 0) {
					uchar m = readTexture<uchar>(t_mask, x, y);
					if (m == 255)
						surf2Dwrite(value, surface, x * sizeof(T), y);
				}
				else {
					surf2Dwrite(value, surface, x * sizeof(T), y);
				}
			}
		}
	}

	template <typename T_TEXTURE, typename T_VALUE>
	void cudaSetValueT(T_TEXTURE& t_value, T_VALUE value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream)
	{
		int width = t_value.width;
		int height = t_value.height;
		if (width <= 0 || height <= 0)
			return;

		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaSetValue_kernel<T_VALUE> << <grid, blk, 0, stream >> > (t_value.surface(), width, height, value, t_mask.texture());
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(prometheus::Texture2DUchar1& t_value, uchar value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream)
	{
		cudaSetValueT(t_value, value, t_mask, stream);
	}

	void cudaSetValue(prometheus::Texture2DUchar4& t_value, uchar4 value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream)
	{
		cudaSetValueT(t_value, value, t_mask, stream);
	}

	void cudaSetValue(prometheus::Texture2DFloat1& t_value, float value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream)
	{
		cudaSetValueT(t_value, value, t_mask, stream);
	}

	void cudaSetValue(prometheus::Texture2DFloat2& t_value, float2 value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream)
	{
		cudaSetValueT(t_value, value, t_mask, stream);
	}

	// set value for 3d
	namespace device
	{
		template <typename T>
		__global__ void cudaSetValue3D_kernel(cudaSurfaceObject_t surface, int width, int height, int depth, T value)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int z = threadIdx.z + blockDim.z * blockIdx.z;

			if (x < width && y < height && z < depth) {
				surf3Dwrite(value, surface, x * sizeof(T), y, z);
			}
		}
	}

	template <typename T_TEXTURE, typename T_VALUE>
	void cudaSetValue3DT(T_TEXTURE& t_value, T_VALUE value, cudaStream_t stream)
	{
		int width = t_value.width;
		int height = t_value.height;
		int depth = t_value.depth;
		if (width <= 0 || height <= 0 || depth <= 0)
			return;

		dim3 blk(16, 16, 4);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y), divUp(depth, blk.z));
		device::cudaSetValue3D_kernel<T_VALUE> << <grid, blk, 0, stream >> > (t_value.surface(), width, height, depth, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(prometheus::Texture3DFloat1& t_value, float value, cudaStream_t stream)
	{
		cudaSetValue3DT(t_value, value, stream);
	}

	///////////////////////////////////// slice 3d /////////////////////////////////////
	namespace device
	{
		template <typename T_VALUE>
		__global__ void cudaSliceTexture3DByY_T_kernel(cudaTextureObject_t t_value, int width, int depth,
			int y, cudaSurfaceObject_t s_slice)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int z = threadIdx.y + blockDim.y * blockIdx.y;
			if (x >= width || z >= depth)
				return;

			T_VALUE val = readTexture<T_VALUE>(t_value, x, y, z);
			surf2Dwrite<T_VALUE>(val, s_slice, x * sizeof (T_VALUE), z);
		}

	}

	template <typename T_VALUE, typename T_TEXTURE_3D, typename T_TEXTURE_2D>
	void cudaSliceTexture3DByY_T(T_TEXTURE_3D& t_value, int y, T_TEXTURE_2D& t_slice, cudaStream_t stream)
	{
		int width = t_value.width;
		int height = t_value.height;
		int depth = t_value.depth;
		if (width <= 0 || height <= 0 || depth <= 0)
			return;

		t_slice.create(depth, width);

		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(depth, blk.y));
		device::cudaSliceTexture3DByY_T_kernel<T_VALUE> << <grid, blk, 0, stream >> > (t_value.texture(), width, depth, y, t_slice.surface());
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSliceTexture3DByY(const prometheus::Texture3DFloat1& t_value, int y, prometheus::Texture2DFloat1& t_slice, cudaStream_t stream)
	{
		cudaSliceTexture3DByY_T<float>(t_value, y, t_slice, stream);
	}

	///////////////////////////////////// set device2d to value /////////////////////////////////////
	namespace device
	{
		template <typename T>
		__global__ void cudaSetValue_kernel(pcl::gpu::PtrStepSz<T> d_arr2d, int width, int height, T value)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x < width && y < height) {
				d_arr2d(y, x) = value;
			}
		}
	}

	template <typename T>
	void cudaSetValueT(pcl::gpu::DeviceArray2D<T>& d_arr2d, T value, cudaStream_t stream)
	{
		int width = d_arr2d.cols();
		int height = d_arr2d.rows();
		if (width <= 0 || height <= 0)
			return;

		dim3 blk(16, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaSetValue_kernel<T> << <grid, blk, 0, stream >> > (d_arr2d, width, height, value);
		cudaSafeCall(cudaGetLastError());
	}

	void cudaSetValue(pcl::gpu::DeviceArray2D<int>& d_arr2d, int value, cudaStream_t stream)
	{
		cudaSetValueT<int>(d_arr2d, value, stream);
	}

	// set color texture from channels
	namespace device
	{
		__global__ void cudaMergeColorToTexture_kernel(
			const unsigned char *d_chanR, int pitchR,
			const unsigned char *d_chanG, int pitchG,
			const unsigned char *d_chanB, int pitchB,
			int width, int height,
			cudaSurfaceObject_t s_rgba)
		{
			const auto x = threadIdx.x + blockDim.x*blockIdx.x;
			const auto y = threadIdx.y + blockDim.y*blockIdx.y;
			if (x >= width || y >= height) return;

			uchar r = d_chanR[y * pitchR + x];
			uchar g = d_chanG[y * pitchG + x];
			uchar b = d_chanB[y * pitchB + x];
			surf2Dwrite(make_uchar4(r, g, b, 255), s_rgba, x * sizeof(uchar4), y);
		}

		__global__ void cudaReadTextureToVector4To3_kernel(
			cudaTextureObject_t t_bgra, int width, int height,
			unsigned char *d_arr)
		{
			const auto x = threadIdx.x + blockDim.x*blockIdx.x;
			const auto y = threadIdx.y + blockDim.y*blockIdx.y;
			if (x >= width || y >= height) return;

			uchar4 bgra = readTexture<uchar4>(t_bgra, x, y);
			int idx = y * width + x;
			d_arr[idx * 3 + 0] = bgra.x;
			d_arr[idx * 3 + 1] = bgra.y;
			d_arr[idx * 3 + 2] = bgra.z;
		}

		__global__ void cudaReadTextureToVector4To3_kernel(
			cudaTextureObject_t t_bgra, int x0, int y0, int width, int height,
			unsigned char *d_arr, int x1, int y1)
		{
			const auto x = threadIdx.x + blockDim.x*blockIdx.x;
			const auto y = threadIdx.y + blockDim.y*blockIdx.y;
			if (x >= width || y >= height) return;

			uchar4 bgra = tex2D<uchar4>(t_bgra, x0 + x, y0 + y);
			int idx = (y1 + y) * width + (x0 + x);
			d_arr[idx * 3 + 0] = bgra.x;
			d_arr[idx * 3 + 1] = bgra.y;
			d_arr[idx * 3 + 2] = bgra.z;
		}

	}

	void cudaMergeColorToTexture(const unsigned char *d_chanR, int pitchR,
		const unsigned char *d_chanG, int pitchG,
		const unsigned char *d_chanB, int pitchB,
		int width, int height,
		cudaSurfaceObject_t s_rgba, cudaStream_t stream)
	{
		dim3 blk(32, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaMergeColorToTexture_kernel << <grid, blk, 0, stream >> > (d_chanR, pitchR,
			d_chanG, pitchG, d_chanB, pitchB, width, height, 
			s_rgba);
	}

	// split color texture to vector
	void cudaReadTextureToVector4To3(cudaTextureObject_t t_bgra, int width, int height,
		surfelwarp::DeviceBufferArray<uchar> &d_arr,
		cudaStream_t stream)
	{
		d_arr.AllocateBuffer(width * height * 3);
		d_arr.ResizeArray(width * height * 3);

		dim3 blk(32, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaReadTextureToVector4To3_kernel << <grid, blk, 0, stream >> > (
			t_bgra, width, height,
			d_arr.Ptr());
	}

	void cudaReadTextureToVector4To3(cudaTextureObject_t t_bgra,
		int x0, int y0, int width, int height,
		surfelwarp::DeviceBufferArray<uchar> &d_arr,
		int x1, int y1, int full_width, int full_height,
		cudaStream_t stream)
	{
		// from src's rect(x0, y0, width, height)
		// to dst's rect(x1, y1, width, height)
		d_arr.AllocateBuffer(full_width * full_height * 3);
		d_arr.ResizeArray(full_width * full_height * 3);

		dim3 blk(32, 16);
		dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
		device::cudaReadTextureToVector4To3_kernel << <grid, blk, 0, stream >> > (
			t_bgra, x0, y0, width, height,
			d_arr.Ptr(), x1, y1);
	}

	///////////////////////////////////// read values /////////////////////////////////////
	namespace device
	{
		template <typename T>
		__global__ void cudaReadTextureValues_kernel(
			cudaTextureObject_t t_src, int width, int height, pcl::gpu::PtrSz<float2> xy_list, int num,
			pcl::gpu::PtrSz<T> values)
		{
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= num)
				return;

			float2 xy = xy_list[idx];

			T val = readTexture<T>(t_src, xy.x * width, xy.y * height);
			values[idx] = val;
		}
	}

	template <typename T>
	void cudaReadTextureValues(cudaTextureObject_t t_src, int width, int height, const std::vector<float2> &xy_list,
		std::vector<T> &values, cudaStream_t stream)
	{
		if (t_src == 0 || xy_list.empty())
			return;

		int num = xy_list.size();
		pcl::gpu::DeviceArray<float2> d_xyList(num);
		cudaSafeCall(cudaMemcpyAsync(d_xyList.ptr(), xy_list.data(), num * sizeof(float2), cudaMemcpyHostToDevice, stream));

		pcl::gpu::DeviceArray<T> d_values(num);

		dim3 blk(512);
		dim3 grid(divUp(num, blk.x));
		device::cudaReadTextureValues_kernel<T> << <grid, blk, 0, stream >> > (
			t_src, width, height, d_xyList, num, 
			d_values);

		values.resize(num);
		cudaSafeCall(cudaMemcpyAsync(values.data(), d_values.ptr(), num * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}

	void cudaReadTextureValuesFloat4(cudaTextureObject_t t_src, int width, int height, const std::vector<float2> &xy_list,
		std::vector<float4> &values, cudaStream_t stream)
	{
		cudaReadTextureValues<float4>(t_src, width, height, xy_list, values, stream);
		cudaSafeCall(cudaGetLastError());
	}

}
