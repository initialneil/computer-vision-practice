#pragma once
#include "cuda/cuda_utils.h"
#include <pcl/gpu/containers/device_array.h>
#include "DeviceBufferArray.h"
#include <cuda_texture_types.h>
#include <exception>
#include <cuda.h>
#include <memory>

namespace surfelwarp
{
	using pcl::gpu::DeviceArray;
	using pcl::gpu::DeviceArray2D;
	using pcl::gpu::PtrSz;
	using pcl::gpu::PtrStep;
	using pcl::gpu::PtrStepSz;

	// shared pointer version of CudaTextureSurface
	struct CudaTextureSurfaceT
	{
		cudaTextureObject_t texture = 0;
		cudaSurfaceObject_t surface = 0;
		cudaArray_t d_array = 0;

		~CudaTextureSurfaceT();
		void release();
	};

	/**
	* \brief Create of 1d linear float texturem, accessed by fetch1DLinear.
	*        Using the array as the underline memory
	*/
	cudaTextureObject_t create1DLinearTexture(const DeviceArray<float>& array);
	cudaTextureObject_t create1DLinearTexture(const DeviceBufferArray<float>& array);

	/**
	* \brief Create TextureDesc for default 2D texture
	*/
    void createDefault2DTextureDesc(cudaTextureDesc& desc, cudaTextureAddressMode border_mode = cudaAddressModeBorder);
    void createLinear2DTextureDesc(cudaTextureDesc& desc, cudaTextureAddressMode border_mode = cudaAddressModeBorder);


    /**
    * \brief Bind texture/surface to cuda array
    */
    void bind2DTextureSurface(cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
        cudaArray_t& d_array, cudaTextureAddressMode border_mode = cudaAddressModeBorder);
	void bind2DTextureSurface(CudaTextureSurface& collect,
		cudaTextureAddressMode border_mode = cudaAddressModeBorder);
	void bind2DTextureSurface(std::shared_ptr<CudaTextureSurfaceT>& collect, 
		cudaTextureAddressMode border_mode = cudaAddressModeBorder);

	/**
	* \brief Create 2D uint16 textures (and surfaces) for depth image
	*/
	void createUShort1Texture(const unsigned height, const unsigned width,
		cudaTextureObject_t &texture, cudaArray_t &d_array);
	void createUShort1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array);
	void createUShort1TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface& texture_collect);
	void createUShort1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect);

	/**
	* \brief Create 2D float4 textures (and surfaces) for all kinds of use
	*/
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect
	);

	/**
	* \brief Create 2D float1 textures (and surfaces) for mean-field inference
	*/
	void createFloat1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array, bool linear = false);
	void createFloat1TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface& texture_collect, bool linear = false);
	void createFloat1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect, bool linear = false);

	/**
	* \brief Create 2D float2 textures (and surfaces) for gradient map
	*/
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect
	);

	/**
	* \brief Create 2D uchar1 textures (and surfaces) for binary mask
	*/
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect
	);

    /**
    * \brief Create 2D uchar4 textures (and surfaces) for binary mask
    */
    void createUChar4TextureSurface(
        const unsigned rows, const unsigned cols,
        cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
        cudaArray_t& d_array
    );
    void createUChar4TextureSurface(
        const unsigned rows, const unsigned cols,
        CudaTextureSurface& texture_collect
	);
	void createUChar4TextureSurface(
		const unsigned rows, const unsigned cols,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect
	);

	/**
	* \brief Create 2D int1 textures (and surfaces) for mean-field inference
	*/
	void createInt1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array, bool linear = false);
	void createInt1TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface& texture_collect, bool linear = false);
	void createInt1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect, bool linear = false);

	/**
	* \brief Release 2D texture
	*/
	void releaseTextureCollect(CudaTextureSurface& texture_collect);

	/**
	* \brief The query functions for 2D texture
	*/
    void query2DTextureExtent(cudaTextureObject_t texture, unsigned& width, unsigned& height);
    void queryCudaArrayExtent(cudaArray_t cu_array, unsigned& width, unsigned& height);

	///////////////////////////////////// 3D /////////////////////////////////////
	/**
	* \brief Create TextureDesc for default 3D texture
	*/
	void createDefault3DTextureDesc(cudaTextureDesc& desc);
	
	/**
	* \brief Create 3D float1 textures (and surfaces) for mean-field inference
	*/
	void createFloat1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createFloat1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect
	);

	void createUChar1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createUChar1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect
	);
	
} // namespace surfelwarp