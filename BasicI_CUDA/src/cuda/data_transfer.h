/* Undistort camera
*  All rights reserved. Prometheus 2019.
*  Contributor(s): Neil Z. Shao
*/
#pragma once
#include <cuda/cuda_utils.h>
#include <cuda/DeviceBufferArray.h>
#include <pcl/gpu/containers/device_array.h>
#include <opencv2/opencv.hpp>
#include "promdata/texture2d.h"
#include "promdata/texture3d.h"

namespace surfelwarp
{
	///////////////////////////////////// download 1d /////////////////////////////////////
	template <typename T>
	void downloadDeviceArray(const pcl::gpu::DeviceArray<T> d_vec, std::vector<T> &h_vec, cudaStream_t stream = 0)
	{
		int num = d_vec.size();
		h_vec.resize(num);
		if (num == 0)
			return;

		cudaMemcpyAsync(h_vec.data(), d_vec.ptr(), num * sizeof(T),
			cudaMemcpyDeviceToHost, stream);
	}

	template <typename T>
	void uploadDeviceArray(const std::vector<T> &h_vec, DeviceBufferArray<T> &d_vec, cudaStream_t stream = 0)
	{
		int num = h_vec.size();
		d_vec.AllocateBuffer(num);
		d_vec.ResizeArray(num);
		if (num == 0)
			return;

		cudaMemcpyAsync(d_vec.Ptr(), h_vec.data(), num * sizeof(T),
			cudaMemcpyHostToDevice, stream);
	}

	template <typename T>
	void uploadDeviceArray(const std::vector<T> &h_vec, pcl::gpu::DeviceArray<T> &d_vec, cudaStream_t stream = 0)
	{
		int num = h_vec.size();
		d_vec.create(num);
		if (num == 0)
			return;

		cudaMemcpyAsync(d_vec.ptr(), h_vec.data(), num * sizeof(T),
			cudaMemcpyHostToDevice, stream);
	}

	template <typename T>
	void copyDeviceArray(const pcl::gpu::DeviceArray<T> &d_vec0, pcl::gpu::DeviceArray<T> &d_vec1, cudaStream_t stream = 0)
	{
		int num = d_vec0.size();
		d_vec1.create(num);
		if (num == 0)
			return;

		cudaMemcpyAsync(d_vec1.ptr(), d_vec0.ptr(), num * sizeof(T),
			cudaMemcpyDeviceToDevice, stream);
	}

	///////////////////////////////////// download texture /////////////////////////////////////
	// down uchar1
	void downloadCudaTextureUchar1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<unsigned char> &d_mat,
		cudaStream_t stream = 0);
    void downloadCudaTextureUchar1(cudaTextureObject_t t_dat, cv::Mat &mat, cudaStream_t stream = 0);
	void downloadCudaArrayUchar1(cudaArray_t d_array, int width, int height, cv::Mat &mat, cudaStream_t stream = 0);

    // down uchar4
    void downloadCudaTextureUchar4(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<uchar4> &d_mat,
        cudaStream_t stream = 0);
    void downloadCudaTextureUchar4(cudaTextureObject_t t_dat, cv::Mat &mat, cudaStream_t stream = 0);
	void downloadCudaArrayUchar4(cudaArray_t d_array, int width, int height, cv::Mat &mat, cudaStream_t stream = 0);

    // down ushort1
    void downloadCudaTextureUshort1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<ushort> &d_mat,
        cudaStream_t stream = 0);
    void downloadCudaTextureUshort1(cudaTextureObject_t t_dat, cv::Mat &mat, cudaStream_t stream = 0);
	void downloadCudaArrayUshort1(cudaArray_t d_array, int width, int height, cv::Mat &mat, cudaStream_t stream = 0);

    // down float1
    void downloadCudaTextureFloat1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<float> &d_mat,
        cudaStream_t stream = 0);
    void downloadCudaTextureFloat1(cudaTextureObject_t t_dat, cv::Mat &mat, cudaStream_t stream = 0);
	void downloadCudaArrayFloat1(cudaArray_t d_array, int width, int height, cv::Mat &mat, cudaStream_t stream = 0);

    // down float2
    void downloadCudaTextureFloat2(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<float2> &d_mat,
        cudaStream_t stream = 0);
    void downloadCudaTextureFloat2(cudaTextureObject_t t_dat, cv::Mat &mat, cudaStream_t stream = 0);
	void downloadCudaArrayFloat2(cudaArray_t d_array, int width, int height, cv::Mat &mat, cudaStream_t stream = 0);

    // down float3
    void downloadCudaTextureFloat4(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<float4> &d_mat,
        cudaStream_t stream = 0);
    void downloadCudaTextureFloat4(cudaTextureObject_t t_dat, cv::Mat &mat, cudaStream_t stream = 0);
	void downloadCudaArrayFloat4(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream = 0);

	// down int1
	void downloadCudaTextureInt1(cudaTextureObject_t t_dat, pcl::gpu::DeviceArray2D<int>& d_mat,
		cudaStream_t stream = 0);
	void downloadCudaTextureInt1(cudaTextureObject_t t_dat, cv::Mat& mat, cudaStream_t stream = 0);
	void downloadCudaArrayInt1(cudaArray_t d_array, int width, int height, cv::Mat& mat, cudaStream_t stream = 0);

	///////////////////////////////////// download 3d /////////////////////////////////////
	void cudaReadRowOfTexture3D(cudaTextureObject_t t_src, int width, int height, int depth, int row_idx,
		cudaSurfaceObject_t s_slice, cudaStream_t stream = 0);

	///////////////////////////////////// upload /////////////////////////////////////
	// upload uchar4
	void uploadCudaSurfaceUchar4(pcl::gpu::DeviceArray2D<uchar4> &d_mat, cudaSurfaceObject_t s_dat,
		cudaStream_t stream = 0);

	// upload from
	template <typename D_TYPE>
	void uploadDeviceArray2D(const cv::Mat& cvmat, pcl::gpu::DeviceArray2D<D_TYPE>& d_vec, cudaStream_t stream = 0)
	{
		d_vec.create(cvmat.rows, cvmat.cols);
		cudaMemcpy2DAsync((void*)d_vec.ptr(), d_vec.step(),
			cvmat.data, cvmat.step,
			cvmat.cols * sizeof(D_TYPE), cvmat.rows,
			cudaMemcpyHostToDevice, stream);
	}

	// download to
	template <int CV_TYPE, typename D_TYPE>
	void downloadDeviceArray2D(const pcl::gpu::DeviceArray2D<D_TYPE>& d_vec, cv::Mat& cvmat, cudaStream_t stream = 0)
	{
		cvmat.create(d_vec.rows(), d_vec.cols(), CV_TYPE);
		cudaMemcpy2DAsync((void*)cvmat.data, cvmat.step,
			d_vec.ptr(), d_vec.step(),
			cvmat.cols * sizeof(D_TYPE), cvmat.rows,
			cudaMemcpyDeviceToHost, stream);
	}

	///////////////////////////////////// copy ///////////////////////////////////////
	// texture to surface
	void cudaCopyTexture2DUchar1(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DUchar4(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DFloat1(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DFloat2(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DFloat4(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DInt1(cudaTextureObject_t t_src, cudaSurfaceObject_t s_dst, int width, int height,
		cudaStream_t stream = 0);

	// device2d to surface
	void cudaCopyDeviceArray2DToTexture(const pcl::gpu::DeviceArray2D<int>& d_arr2d, prometheus::Texture2DFloat1& t_map,
		float alpha, float gamma, cudaStream_t stream = 0);

	// texture to cuda array
	void cudaCopyTexture2DUchar1(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DUchar4(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DFloat1(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream = 0);
	void cudaCopyTexture2DInt1(cudaTextureObject_t t_src, cudaArray_t darr_dst, int width, int height,
		cudaStream_t stream = 0);

	///////////////////////////////////// opencv cuda /////////////////////////////////////
	void cudaSplitUchar4Mask(const pcl::gpu::DeviceArray2D<uchar4> &d_mat, 
		cv::cuda::GpuMat &d_clr, cv::cuda::GpuMat &d_mask, cudaStream_t stream = 0);

	void cudaMergeUchar4Mask(const cv::cuda::GpuMat &d_clr, const cv::cuda::GpuMat &d_mask, 
		cudaSurfaceObject_t s_frame, cudaStream_t stream = 0);

	///////////////////////////////////// set value /////////////////////////////////////
	void cudaSetValue(uchar *darr_ptr, int num, uchar value, cudaStream_t stream = 0);
	void cudaSetValue(uchar3 *darr_ptr, int num, uchar3 value, cudaStream_t stream = 0);
	void cudaSetValue(uchar4 *darr_ptr, int num, uchar4 value, cudaStream_t stream = 0);

	void cudaSetValue(cudaSurfaceObject_t s_color, int width, int height, float value, cudaStream_t stream = 0);
	void cudaSetValue(cudaSurfaceObject_t s_color, int width, int height, uchar4 value, cudaStream_t stream = 0);
	void cudaSurface3DSetValue(cudaSurfaceObject_t s_volume, int width, int height, int depth,
		float value, cudaStream_t stream = 0);

	void cudaSetValue(prometheus::Texture2DUchar1& t_value, uchar value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream = 0);
	void cudaSetValue(prometheus::Texture2DUchar4& t_value, uchar4 value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream = 0);
	void cudaSetValue(prometheus::Texture2DFloat1& t_value, float value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream = 0);
	void cudaSetValue(prometheus::Texture2DFloat2& t_value, float2 value, const prometheus::Texture2DUchar1& t_mask, cudaStream_t stream = 0);

	void cudaSetValue(pcl::gpu::DeviceArray2D<int>& d_arr2d, int value, cudaStream_t stream = 0);

	// set value for 3d
	void cudaSetValue(prometheus::Texture3DFloat1 & t_value, float value, cudaStream_t stream = 0);

	///////////////////////////////////// slice 3d /////////////////////////////////////
	void cudaSliceTexture3DByY(const prometheus::Texture3DFloat1& t_value, int y, prometheus::Texture2DFloat1& t_slice, cudaStream_t stream = 0);

	// set color texture from channels
	void cudaMergeColorToTexture(const unsigned char *d_chanR, int pitchR, 
		const unsigned char *d_chanG, int pitchG, 
		const unsigned char *d_chanB, int pitchB,
		int width, int height, 
		cudaSurfaceObject_t s_rgba, cudaStream_t stream = 0);

	// split color texture to vector
	void cudaReadTextureToVector4To3(cudaTextureObject_t t_bgra, int width, int height,
		surfelwarp::DeviceBufferArray<uchar> &d_arr,
		cudaStream_t stream = 0);
	void cudaReadTextureToVector4To3(cudaTextureObject_t t_bgra,
		int x0, int y0, int width, int height,
		surfelwarp::DeviceBufferArray<uchar> &d_arr,
		int x1, int y1, int full_width, int full_height,
		cudaStream_t stream = 0);

	///////////////////////////////////// read values /////////////////////////////////////
	void cudaReadTextureValuesFloat4(cudaTextureObject_t t_src, int width, int height, const std::vector<float2> &xy_list,
		std::vector<float4> &values, cudaStream_t stream = 0);

}
