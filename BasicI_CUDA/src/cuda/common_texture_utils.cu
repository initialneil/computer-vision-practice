#include "common_texture_utils.h"

namespace surfelwarp
{

	// shared pointer version of CudaTextureSurface
	CudaTextureSurfaceT::~CudaTextureSurfaceT()
	{
		release();
	}

	void CudaTextureSurfaceT::release()
	{
		//printf("[CudaTextureSurfaceT] release %d (%d %d %d)\n", (int)this,
		//	(int)this->texture, (int)this->surface, (int)this->d_array);

		if (this->texture)
			cudaSafeCall(cudaDestroyTextureObject(this->texture));
		if (this->surface)
			cudaSafeCall(cudaDestroySurfaceObject(this->surface));
		if (this->d_array)
			cudaSafeCall(cudaFreeArray(this->d_array));

		this->texture = 0;
		this->surface = 0;
		this->d_array = 0;
	}

	cudaTextureObject_t create1DLinearTexture(const DeviceArray<float> &array)
	{
		cudaTextureDesc texture_desc;
		memset(&texture_desc, 0, sizeof(cudaTextureDesc));
		texture_desc.normalizedCoords = 0;
		texture_desc.addressMode[0] = cudaAddressModeBorder; //Return 0 outside the boundary
		texture_desc.addressMode[1] = cudaAddressModeBorder;
		texture_desc.addressMode[2] = cudaAddressModeBorder;
		texture_desc.filterMode = cudaFilterModePoint;
		texture_desc.readMode = cudaReadModeElementType;
		texture_desc.sRGB = 0;

		//Create resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeLinear;
		resource_desc.res.linear.devPtr = (void*)array.ptr();
		resource_desc.res.linear.sizeInBytes = array.sizeBytes();
		resource_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resource_desc.res.linear.desc.x = 32;
		resource_desc.res.linear.desc.y = 0;
		resource_desc.res.linear.desc.z = 0;
		resource_desc.res.linear.desc.w = 0;

		//Allocate the texture
		cudaTextureObject_t d_texture;
		cudaSafeCall(cudaCreateTextureObject(&d_texture, &resource_desc, &texture_desc, nullptr));
		return d_texture;
	}

	cudaTextureObject_t create1DLinearTexture(const DeviceBufferArray<float>& array)
	{
		DeviceArray<float> pcl_array((float*)array.Ptr(), array.Capacity());
		return create1DLinearTexture(pcl_array);
	}

	void createDefault2DTextureDesc(cudaTextureDesc &desc, cudaTextureAddressMode border_mode)
	{
		// cudaAddressModeBorder: Return 0 outside the boundary

		memset(&desc, 0, sizeof(desc));
		desc.addressMode[0] = border_mode;
		desc.addressMode[1] = border_mode;
		desc.addressMode[2] = border_mode;
		desc.filterMode = cudaFilterModePoint;
		desc.readMode = cudaReadModeElementType;
		desc.normalizedCoords = 0;
	}

	void createLinear2DTextureDesc(cudaTextureDesc &desc, cudaTextureAddressMode border_mode)
	{
		memset(&desc, 0, sizeof(desc));
		desc.addressMode[0] = border_mode;
		desc.addressMode[1] = border_mode;
		desc.addressMode[2] = border_mode;
		desc.filterMode = cudaFilterModeLinear;
		desc.readMode = cudaReadModeElementType;
		desc.normalizedCoords = 0;
	}

	////////////////////////////// bind collect //////////////////////////////
	void bind2DTextureSurface(cudaTextureObject_t &texture, cudaSurfaceObject_t &surface,
		cudaArray_t &d_array, cudaTextureAddressMode border_mode)
	{
		//The texture description
		cudaTextureDesc depth_texture_desc;
		createDefault2DTextureDesc(depth_texture_desc, border_mode);

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void bind2DTextureSurface(CudaTextureSurface & collect, cudaTextureAddressMode border_mode)
	{
		bind2DTextureSurface(collect.texture, collect.surface, collect.d_array, border_mode);
	}

	void bind2DTextureSurface(std::shared_ptr<CudaTextureSurfaceT>& collect, cudaTextureAddressMode border_mode)
	{
		if (!collect)
			return;
		bind2DTextureSurface(collect->texture, collect->surface, collect->d_array, border_mode);
	}

	////////////////////////////// depth ushort1 //////////////////////////////
	void createUShort1Texture(const unsigned height, const unsigned width,
		cudaTextureObject_t &texture, cudaArray_t &d_array)
	{
		//The texture description
		cudaTextureDesc depth_texture_desc;
		createDefault2DTextureDesc(depth_texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &depth_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
	}

	void createUShort1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t &texture,
		cudaSurfaceObject_t &surface,
		cudaArray_t &d_array) 
	{
		//The texture description
		cudaTextureDesc depth_texture_desc;
		createDefault2DTextureDesc(depth_texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &depth_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void createUShort1TextureSurface(const unsigned height, const unsigned width, CudaTextureSurface & texture_collect)
	{
		createUShort1TextureSurface(height, width,
			texture_collect.texture, texture_collect.surface, texture_collect.d_array);
	}

	void createUShort1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createUShort1TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
	}

	////////////////////////////// float4 texture //////////////////////////////
	void createFloat4TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t &texture,
		cudaSurfaceObject_t &surface,
		cudaArray_t &d_array) 
	{
		//The texture description
		cudaTextureDesc float4_texture_desc;
		createDefault2DTextureDesc(float4_texture_desc, cudaAddressModeClamp);

		//Create channel descriptions
		cudaChannelFormatDesc float4_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &float4_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float4_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void createFloat4TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface & texture_collect)
	{
		createFloat4TextureSurface(
			rows, cols,
			texture_collect.texture,
			texture_collect.surface,
			texture_collect.d_array
		);
	}

	void createFloat4TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect) 
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createFloat4TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
	}

	////////////////////////////// float1 //////////////////////////////
	void createFloat1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array, bool linear)
	{
		//The texture description
		cudaTextureDesc float1_texture_desc;
		if (!linear)
			createDefault2DTextureDesc(float1_texture_desc);
		else
			createLinear2DTextureDesc(float1_texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &float1_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}


	void createFloat1TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface & texture_collect, bool linear)
	{
		createFloat1TextureSurface(
			height, width,
			texture_collect.texture,
			texture_collect.surface,
			texture_collect.d_array,
			linear
		);
	}

	void createFloat1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect, bool linear)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createFloat1TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array,
			linear
		);
	}

	////////////////////////////// float2 //////////////////////////////
	void createFloat2TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t & texture,
		cudaSurfaceObject_t & surface,
		cudaArray_t & d_array) 
	{
		//The texture description
		cudaTextureDesc float2_texture_desc;
		createDefault2DTextureDesc(float2_texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc float2_channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &float2_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float2_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void createFloat2TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface & texture_collect)
	{
		createFloat2TextureSurface(
			height, width,
			texture_collect.texture,
			texture_collect.surface,
			texture_collect.d_array
		);
	}

	void createFloat2TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createFloat2TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
	}


	void createUChar1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t & texture,
		cudaSurfaceObject_t & surface,
		cudaArray_t & d_array)
	{
		//The texture description
		cudaTextureDesc uchar1_texture_desc;
		createDefault2DTextureDesc(uchar1_texture_desc, cudaAddressModeClamp);

		//Create channel descriptions
		cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &uchar1_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void createUChar1TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface & texture_collect)
	{
		createUChar1TextureSurface(
			height, width,
			texture_collect.texture,
			texture_collect.surface,
			texture_collect.d_array
		);
	}

	void createUChar1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createUChar1TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
	}


	void createUChar4TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t & texture,
		cudaSurfaceObject_t & surface,
		cudaArray_t & d_array)
	{
		//The texture description
		cudaTextureDesc uchar4_texture_desc;
		createDefault2DTextureDesc(uchar4_texture_desc, cudaAddressModeClamp);
		//createLinear2DTextureDesc(uchar4_texture_desc, cudaAddressModeClamp);

		//Create channel descriptions
		cudaChannelFormatDesc uchar4_channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &uchar4_channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &uchar4_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void createUChar4TextureSurface(
		const unsigned height, const unsigned width,
		CudaTextureSurface & texture_collect)
	{
		createUChar4TextureSurface(
			height, width,
			texture_collect.texture,
			texture_collect.surface,
			texture_collect.d_array
		);
	}

	void createUChar4TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
		createUChar4TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
	}

	////////////////////////////// int1 //////////////////////////////
	void createInt1TextureSurface(const unsigned height, const unsigned width,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array, bool linear)
	{
		//The texture description
		cudaTextureDesc texture_desc;
		if (!linear)
			createDefault2DTextureDesc(texture_desc);
		else
			createLinear2DTextureDesc(texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

		//Allocate the cuda array
		cudaSafeCall(cudaMallocArray(&d_array, &channel_desc, width, height));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}


	void createInt1TextureSurface(const unsigned height, const unsigned width,
		CudaTextureSurface& texture_collect, bool linear)
	{
		createInt1TextureSurface(
			height, width,
			texture_collect.texture,
			texture_collect.surface,
			texture_collect.d_array,
			linear
		);
	}

	void createInt1TextureSurface(const unsigned height, const unsigned width,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect, bool linear)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createInt1TextureSurface(
			height, width,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array,
			linear
		);
	}

	////////////////////////////// query //////////////////////////////
	void query2DTextureExtent(cudaTextureObject_t texture, unsigned &width, unsigned &height)
	{
		cudaResourceDesc texture_res;
		cudaSafeCall(cudaGetTextureObjectResourceDesc(&texture_res, texture));
		cudaArray_t cu_array = texture_res.res.array.array;
		cudaChannelFormatDesc channel_desc;
		cudaExtent extent;
		unsigned int flag;
		cudaSafeCall(cudaArrayGetInfo(&channel_desc, &extent, &flag, cu_array));

		width = extent.width;
		height = extent.height;
	}

	void queryCudaArrayExtent(cudaArray_t cu_array, unsigned& width, unsigned& height)
	{
		cudaChannelFormatDesc channel_desc;
		cudaExtent extent;
		unsigned int flag;
		cudaSafeCall(cudaArrayGetInfo(&channel_desc, &extent, &flag, cu_array));

		width = extent.width;
		height = extent.height;
	}

	// release
	void releaseTextureCollect(CudaTextureSurface & texture_collect)
	{
		if (texture_collect.texture)
			cudaSafeCall(cudaDestroyTextureObject(texture_collect.texture));
		if (texture_collect.surface)
			cudaSafeCall(cudaDestroySurfaceObject(texture_collect.surface));
		if (texture_collect.d_array)
			cudaSafeCall(cudaFreeArray(texture_collect.d_array));

		texture_collect.texture = 0;
		texture_collect.surface = 0;
		texture_collect.d_array = 0;
	}

	///////////////////////////////////// 3D /////////////////////////////////////
	/**
	* \brief Create TextureDesc for default 3D texture
	*/
	void createDefault3DTextureDesc(cudaTextureDesc &desc)
	{
		memset(&desc, 0, sizeof(desc));
		desc.addressMode[0] = cudaAddressModeBorder; //Return 0 outside the boundary
		desc.addressMode[1] = cudaAddressModeBorder;
		desc.addressMode[2] = cudaAddressModeBorder;
		desc.filterMode = cudaFilterModePoint;
		desc.readMode = cudaReadModeElementType;
		desc.normalizedCoords = 0;
	}

	/**
	* \brief Create 3D float1 textures (and surfaces) for mean-field inference
	*/
	void createFloat1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array)
	{
		//The texture description
		cudaTextureDesc float1_texture_desc;
		createDefault3DTextureDesc(float1_texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		//Allocate the cuda array
		cudaExtent ext;
		ext.width = (unsigned)width;
		ext.height = (unsigned)height;
		ext.depth = (unsigned)depth;
		cudaSafeCall(cudaMalloc3DArray(&d_array, &float1_channel_desc, ext));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}


	void createFloat1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createFloat1TextureSurface3D(
			width, height, depth,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
	}

	/**
	* \brief Create 3D float1 textures (and surfaces) for mean-field inference
	*/
	void createUChar1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array)
	{
		//The texture description
		cudaTextureDesc uchar1_texture_desc;
		createDefault3DTextureDesc(uchar1_texture_desc);

		//Create channel descriptions
		cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

		//Allocate the cuda array
		cudaExtent ext;
		ext.width = (unsigned)width;
		ext.height = (unsigned)height;
		ext.depth = (unsigned)depth;
		cudaSafeCall(cudaMalloc3DArray(&d_array, &uchar1_channel_desc, ext));

		//Create the resource desc
		cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(cudaResourceDesc));
		resource_desc.resType = cudaResourceTypeArray;
		resource_desc.res.array.array = d_array;

		//Allocate the texture
		cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0));
		cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
	}

	void createUChar1TextureSurface3D(
		const unsigned width, const unsigned height, const unsigned depth,
		std::shared_ptr<CudaTextureSurfaceT>& texture_collect)
	{
		if (!texture_collect)
			texture_collect = std::make_shared<CudaTextureSurfaceT>();

		if (height == 0 || width == 0) {
			texture_collect->release();
			return;
		}

		createUChar1TextureSurface3D(
			width, height, depth,
			texture_collect->texture,
			texture_collect->surface,
			texture_collect->d_array
		);
	}

}
