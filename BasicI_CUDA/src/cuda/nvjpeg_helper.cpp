/* MJpeg decode using NvJpeg after CUDA 10.2.
*  All rights reserved. Prometheus 2020.
*  Contributor(s): Neil Z. Shao
*/
#pragma once
#include "nvjpeg_helper.h"
#include "cuda/cuda_utils.h"
#include "cuda/helper_cuda.h"
#include "cuda/data_transfer.h"
#include "utils/timer.h"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
static int dev_free(void *p) { return (int)cudaFree(p); }

namespace prometheus
{
	NvJpegHelper::NvJpegHelper()
	{}

	NvJpegHelper::~NvJpegHelper()
	{
		//printf("[NvJpegHelper] destroy\n");
		release_buffers(m_nvimg);

		if (m_handler.nv_handle) {
			checkCudaErrors(nvjpegEncoderParamsDestroy(m_handler.encoder_params));
			checkCudaErrors(nvjpegEncoderStateDestroy(m_handler.encoder_state));
			checkCudaErrors(nvjpegJpegStateDestroy(m_handler.nvjpeg_state));
			checkCudaErrors(nvjpegDestroy(m_handler.nv_handle));
			m_handler.nv_handle = 0;
		}

		//cudaSafeCall(cudaDeviceSynchronize());
		//cudaSafeCall(cudaGetLastError());
	}

	void NvJpegHelper::init()
	{
		cudaDeviceProp props;
		checkCudaErrors(cudaGetDeviceProperties(&props, m_options.dev));

		printf("[NvJpegHelper] Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
			m_options.dev, props.name, props.multiProcessorCount,
			props.maxThreadsPerMultiProcessor, props.major, props.minor,
			props.ECCEnabled ? "on" : "off");

		nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };
		checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
			&m_handler.nv_handle));
		//nvjpegCreateSimple(&m_handler.nv_handle);

		// decoder
		checkCudaErrors(nvjpegJpegStateCreate(m_handler.nv_handle, &m_handler.nvjpeg_state));
		checkCudaErrors(nvjpegDecodeBatchedInitialize(m_handler.nv_handle, m_handler.nvjpeg_state,
				1, 1, m_options.fmt));

		// encoder
		checkCudaErrors(nvjpegEncoderStateCreate(m_handler.nv_handle, &m_handler.encoder_state, 0));
		checkCudaErrors(nvjpegEncoderParamsCreate(m_handler.nv_handle, &m_handler.encoder_params, 0));
		checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(m_handler.encoder_params, NVJPEG_CSS_420, 0));
		checkCudaErrors(nvjpegEncoderParamsSetQuality(m_handler.encoder_params, 95, 0));
	}

	///////////////////////////////////// inner /////////////////////////////////////
	// prepare buffers for RGBi output format
	int NvJpegHelper::prepareBuffers(const std::vector<char> &file_data, size_t file_len,
		int &img_width, int &img_height,
		nvjpegImage_t &ibuf, nvjpegImage_t &isz)
	{
		int widths[NVJPEG_MAX_COMPONENT];
		int heights[NVJPEG_MAX_COMPONENT];
		int channels;
		nvjpegChromaSubsampling_t subsampling;

		if (EXIT_SUCCESS != nvjpegGetImageInfo(
			m_handler.nv_handle, (unsigned char *)file_data.data(), file_len,
			&channels, &subsampling, widths, heights))
			return EXIT_FAILURE;

		img_width = widths[0];
		img_height = heights[0];

		if (m_options.verbose) {
			std::cout << "Image is " << channels << " channels." << std::endl;
			for (int c = 0; c < channels; c++) {
				std::cout << "Channel #" << c << " size: " << widths[c] << " x "
					<< heights[c] << std::endl;
			}

			switch (subsampling) {
			case NVJPEG_CSS_444:
				std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
				break;
			case NVJPEG_CSS_440:
				std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
				break;
			case NVJPEG_CSS_422:
				std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
				break;
			case NVJPEG_CSS_420:
				std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
				break;
			case NVJPEG_CSS_411:
				std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
				break;
			case NVJPEG_CSS_410:
				std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
				break;
			case NVJPEG_CSS_GRAY:
				std::cout << "Grayscale JPEG " << std::endl;
				break;
			case NVJPEG_CSS_UNKNOWN:
				std::cout << "Unknown chroma subsampling" << std::endl;
				return EXIT_FAILURE;
			}
		}

		int mul = 1;
		// in the case of interleaved RGB output, write only to single channel, but
		// 3 samples at once
		if (m_options.fmt == NVJPEG_OUTPUT_RGBI || m_options.fmt == NVJPEG_OUTPUT_BGRI) {
			channels = 1;
			mul = 3;
		}
		// in the case of rgb create 3 buffers with sizes of original image
		else if (m_options.fmt == NVJPEG_OUTPUT_RGB ||
			m_options.fmt == NVJPEG_OUTPUT_BGR) {
			channels = 3;
			widths[1] = widths[2] = widths[0];
			heights[1] = heights[2] = heights[0];
		}

		// realloc output buffer if required
		for (int c = 0; c < channels; c++) {
			int aw = mul * widths[c];
			int ah = heights[c];
			int sz = aw * ah;
			ibuf.pitch[c] = aw;
			if (sz > isz.pitch[c]) {
				if (isz.pitch[c] > 0) {
					checkCudaErrors(cudaFree(ibuf.channel[c]));
				}
				checkCudaErrors(cudaMalloc((void **)&ibuf.channel[c], sz));
				isz.pitch[c] = sz;
			}
		}

		return EXIT_SUCCESS;
	}

	int NvJpegHelper::prepareBuffers(const std::vector<std::vector<char>> &file_data, std::vector<size_t> &file_len,
		std::vector<int> &img_width, std::vector<int> &img_height,
		std::vector<nvjpegImage_t> &ibuf,
		std::vector<nvjpegImage_t> &isz)
	{
		int widths[NVJPEG_MAX_COMPONENT];
		int heights[NVJPEG_MAX_COMPONENT];
		int channels;
		nvjpegChromaSubsampling_t subsampling;

		for (int i = 0; i < file_data.size(); i++) {
			if (EXIT_SUCCESS != nvjpegGetImageInfo(
				m_handler.nv_handle, (unsigned char *)file_data[i].data(), file_len[i],
				&channels, &subsampling, widths, heights))
			{
				printf("[NvJpegHelper] get image info failed\n");
				return EXIT_FAILURE;
			}

			img_width[i] = widths[0];
			img_height[i] = heights[0];

			if (m_options.verbose) {
				std::cout << "Processing: " << i << std::endl;
				std::cout << "Image is " << channels << " channels." << std::endl;
				for (int c = 0; c < channels; c++) {
					std::cout << "Channel #" << c << " size: " << widths[c] << " x "
						<< heights[c] << std::endl;
				}

				switch (subsampling) {
				case NVJPEG_CSS_444:
					std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
					break;
				case NVJPEG_CSS_440:
					std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
					break;
				case NVJPEG_CSS_422:
					std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
					break;
				case NVJPEG_CSS_420:
					std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
					break;
				case NVJPEG_CSS_411:
					std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
					break;
				case NVJPEG_CSS_410:
					std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
					break;
				case NVJPEG_CSS_GRAY:
					std::cout << "Grayscale JPEG " << std::endl;
					break;
				case NVJPEG_CSS_UNKNOWN:
					std::cout << "Unknown chroma subsampling" << std::endl;
					printf("[NvJpegHelper] Unknown chroma subsampling\n");
					return EXIT_FAILURE;
				}
			}

			int mul = 1;
			// in the case of interleaved RGB output, write only to single channel, but
			// 3 samples at once
			if (m_options.fmt == NVJPEG_OUTPUT_RGBI || m_options.fmt == NVJPEG_OUTPUT_BGRI) {
				channels = 1;
				mul = 3;
			}
			// in the case of rgb create 3 buffers with sizes of original image
			else if (m_options.fmt == NVJPEG_OUTPUT_RGB ||
				m_options.fmt == NVJPEG_OUTPUT_BGR) {
				channels = 3;
				widths[1] = widths[2] = widths[0];
				heights[1] = heights[2] = heights[0];
			}

			// realloc output buffer if required
			for (int c = 0; c < channels; c++) {
				int aw = mul * widths[c];
				int ah = heights[c];
				int sz = aw * ah;
				ibuf[i].pitch[c] = aw;
				if (sz > isz[i].pitch[c]) {
					if (ibuf[i].channel[c]) {
						checkCudaErrors(cudaFree(ibuf[i].channel[c]));
					}
					checkCudaErrors(cudaMalloc((void **)&ibuf[i].channel[c], sz));
					isz[i].pitch[c] = sz;
				}
			}
		}

		return EXIT_SUCCESS;
	}

	int NvJpegHelper::decodeImage(const std::vector<char> &img_data, size_t img_len,
		nvjpegImage_t &out, cudaStream_t stream)
	{
		//checkCudaErrors(cudaStreamSynchronize(streams[0]));
		nvjpegStatus_t status = nvjpegDecode(m_handler.nv_handle, m_handler.nvjpeg_state,
			(const unsigned char *)img_data.data(),
			img_len, m_options.fmt, &out,
			stream);
		return status;
	}

	void NvJpegHelper::release_buffers(std::vector<nvjpegImage_t> &ibuf) 
	{
		for (int i = 0; i < ibuf.size(); i++) {
			for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
				if (ibuf[i].channel[c]) checkCudaErrors(cudaFree(ibuf[i].channel[c]));
		}
	}

	int NvJpegHelper::prepareInputData(const std::vector<cv::Mat> &rawJpegs, const std::vector<cudaStream_t> &streams,
		std::vector<std::vector<char>> &input_data,
		std::vector<size_t> &input_len,
		std::vector<cudaStream_t> &input_stream)
	{
		int num = rawJpegs.size();
		input_data.resize(num);
		input_len.resize(num);
		input_stream.resize(num);

		for (int i = 0; i < num; ++i) {
			input_data[i] = vector<char>(rawJpegs[i].data, rawJpegs[i].data + rawJpegs[i].cols);
			input_len[i] = rawJpegs[i].cols;
			if (i < streams.size())
				input_stream[i] = streams[i];
			else
				input_stream[i] = 0;
		}

		// output buffers
		m_nvimg.resize(num);
		// output buffer sizes, for convenience
		m_isz.resize(num);
		for (int i = 0; i < m_nvimg.size(); i++) {
			for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
				m_nvimg[i].channel[c] = NULL;
				m_nvimg[i].pitch[c] = 0;
				m_isz[i].pitch[c] = 0;
			}
		}

		m_widths.resize(num);
		m_heights.resize(num);
		if (prepareBuffers(input_data, input_len, m_widths, m_heights, m_nvimg, m_isz))
			return EXIT_FAILURE;

		return EXIT_SUCCESS;
	}

	int NvJpegHelper::prepareInputData(const cv::Mat &rawJpegs, cudaStream_t stream,
		std::vector<char> &input_data,
		size_t &input_len)
	{
		input_data = vector<char>(rawJpegs.data, rawJpegs.data + rawJpegs.cols);
		input_len = rawJpegs.cols;

		// output buffers
		m_nvimg.resize(1);
		// output buffer sizes, for convenience
		m_isz.resize(1);
		for (int i = 0; i < m_nvimg.size(); i++) {
			for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
				m_nvimg[i].channel[c] = NULL;
				m_nvimg[i].pitch[c] = 0;
				m_isz[i].pitch[c] = 0;
			}
		}

		m_widths.resize(1);
		m_heights.resize(1);
		if (prepareBuffers(input_data, input_len, m_widths[0], m_heights[0], m_nvimg[0], m_isz[0]))
			return EXIT_FAILURE;

		return EXIT_SUCCESS;
	}

	///////////////////////////////////// decode api /////////////////////////////////////
	int NvJpegHelper::decodeFrames(const std::vector<cv::Mat> &rawJpegs, const std::vector<cudaStream_t> &streams)
	{
		if (!m_handler.nv_handle)
			init();

		// buffer for all frames
		int num = rawJpegs.size();
		std::vector<std::vector<char>> input_data(num);
		std::vector<size_t> input_len(num);
		std::vector<cudaStream_t> input_streams(num);
		if (prepareInputData(rawJpegs, streams, input_data, input_len, input_streams)) {
			printf("[NvJpegHelper] prepare input data failed\n");
			return EXIT_FAILURE;
		}

		// decode
		for (int i = 0; i < num; ++i) {
			if (decodeImage(input_data[i], input_len[i], m_nvimg[i], input_streams[i])) {
				printf("[NvJpegHelper] decode failed\n");
				return EXIT_FAILURE;
			}
		}

		return EXIT_SUCCESS;
	}

	int NvJpegHelper::decodeFrames(const std::vector<cv::Mat> &rawJpegs, std::vector<cv::Mat> &decodedFrames,
		const std::vector<cudaStream_t> &streams)
	{
		int ret = decodeFrames(rawJpegs, streams);
		if (ret != EXIT_SUCCESS)
			return ret;

		int num = rawJpegs.size();
		decodedFrames.resize(num);
		for (int i = 0; i < num; ++i) {
			downloadNvJpegImage(m_nvimg[i], m_widths[i], m_heights[i], decodedFrames[i], streams[i]);
		}
		return EXIT_SUCCESS;
	}

	int NvJpegHelper::decodeFrames(const std::vector<cv::Mat> &rawJpegs, std::vector<Texture2DUchar4 *> &decodeColorTs,
		const std::vector<cudaStream_t> &streams)
	{
		int ret = decodeFrames(rawJpegs, streams);
		if (ret != EXIT_SUCCESS)
			return ret;

		int num = rawJpegs.size();
		if (decodeColorTs.size() != num)
			decodeColorTs.resize(num);

		for (int i = 0; i < num; ++i) {
			cudaStream_t stream = i < streams.size() ? streams[i] : 0;
			copyNvJpegImage(m_nvimg[i], m_widths[i], m_heights[i], *decodeColorTs[i], stream);
		}

		return EXIT_SUCCESS;
	}

	void NvJpegHelper::allocBuffer(int num)
	{
		if (!m_handler.nv_handle)
			init();

		if (m_nvimg.size() >= num)
			return;

		release_buffers(m_nvimg);

		// output buffers
		m_nvimg.resize(num);
		// output buffer sizes, for convenience
		m_isz.resize(num);
		for (int i = 0; i < m_nvimg.size(); i++) {
			for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
				m_nvimg[i].channel[c] = NULL;
				m_nvimg[i].pitch[c] = 0;
				m_isz[i].pitch[c] = 0;
			}
		}

		m_widths.resize(num);
		m_heights.resize(num);
	}

	int NvJpegHelper::decodeFrame(int i, const cv::Mat &rawJpeg, Texture2DUchar4 &t_color, cudaStream_t stream)
	{
		vector<char> input_data(rawJpeg.data, rawJpeg.data + rawJpeg.cols);
		size_t input_len = rawJpeg.cols;

		if (EXIT_SUCCESS != 
			prepareBuffers(input_data, input_len, m_widths[i], m_heights[i], m_nvimg[i], m_isz[i])) 
		{
			printf("[NvJpegHelper] get jpeg info failed\n");
			return EXIT_FAILURE;
		}

		if (EXIT_SUCCESS != decodeImage(input_data, input_len, m_nvimg[i], stream)) {
			printf("[NvJpegHelper] decode failed\n");
			return EXIT_FAILURE;
		}
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
		copyNvJpegImage(m_nvimg[i], m_widths[i], m_heights[i], t_color, stream);
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
		return EXIT_SUCCESS;
	}

	int NvJpegHelper::decodeFrame(int i, const cv::Mat &rawJpeg, cv::Mat &color, cudaStream_t stream)
	{
		vector<char> input_data(rawJpeg.data, rawJpeg.data + rawJpeg.cols);
		size_t input_len = rawJpeg.cols;

		if (EXIT_SUCCESS !=
			prepareBuffers(input_data, input_len, m_widths[i], m_heights[i], m_nvimg[i], m_isz[i]))
		{
			printf("[NvJpegHelper] get jpeg info failed\n");
			return EXIT_FAILURE;
		}

		if (EXIT_SUCCESS != decodeImage(input_data, input_len, m_nvimg[i], stream)) {
			printf("[NvJpegHelper] decode failed\n");
			return EXIT_FAILURE;
		}

		downloadNvJpegImage(m_nvimg[i], m_widths[i], m_heights[i], color, stream);
		return EXIT_SUCCESS;
	}

	// encode
	void NvJpegHelper::uploadToEncodeBuffer(const cv::Mat &frame, cudaStream_t stream)
	{
		size_t elements = frame.cols * frame.rows;
		m_handler.pinned_ptr.create(elements);
		m_handler.pinned_ptr.copyFrom((uchar4 *)frame.data, elements);

		//startCudaTimer(upload_texture);
		m_handler.t_color.upload(m_handler.pinned_ptr.Ptr(), frame.cols, frame.rows, stream);
		//m_handler.t_color.upload(frame, stream);
		//stopCudaTimerTag("[NvJpeg]", upload_texture);

		//startCudaTimer(texture_to_buffer);
		surfelwarp::cudaReadTextureToVector4To3(m_handler.t_color.texture(), frame.cols, frame.rows,
			m_handler.d_input, stream);
		//stopCudaTimerTag("[NvJpeg]", texture_to_buffer);
	}

	int NvJpegHelper::encodeFrame(const cv::Mat &frame, std::vector<uchar> &buf, cudaStream_t stream)
	{
		if (!m_handler.nv_handle)
			init();

		startCudaTimer(upload_frame);
		uploadToEncodeBuffer(frame, stream);
		stopCudaTimerTag("[NvJpeg]", upload_frame);

		m_nvimg.resize(1);
		m_nvimg[0].channel[0] = m_handler.d_input.Ptr();
		m_nvimg[0].pitch[0] = frame.cols * 3;

		startCudaTimer(encode_frame);
		cudaSafeCall(nvjpegEncodeImage(m_handler.nv_handle, m_handler.encoder_state, m_handler.encoder_params,
			&m_nvimg[0], NVJPEG_INPUT_BGRI, frame.cols, frame.rows, stream));
		cudaSafeCall(cudaGetLastError());
		stopCudaTimerTag("[NvJpeg]", encode_frame);

		startCudaTimer(retrieve_result);
		size_t length = 0;
		cudaSafeCall(nvjpegEncodeRetrieveBitstream(m_handler.nv_handle, m_handler.encoder_state, NULL, &length, stream));
		buf.resize(length);
		cudaSafeCall(nvjpegEncodeRetrieveBitstream(m_handler.nv_handle, m_handler.encoder_state, buf.data(), &length, stream));
		stopCudaTimerTag("[NvJpeg]", retrieve_result);

		m_nvimg.clear();
		cudaSafeCall(cudaGetLastError());
		return EXIT_SUCCESS;
	}

	int NvJpegHelper::encodeFrame(const pcl::gpu::DeviceArray<uchar> &darr_frame, int width, int height,
		std::vector<uchar> &buf, cudaStream_t stream)
	{
		if (!m_handler.nv_handle)
			init();

		m_nvimg.resize(1);
		m_nvimg[0].channel[0] = (unsigned char *)darr_frame.ptr();
		m_nvimg[0].pitch[0] = width * 3;

		//startCudaTimer(encode_frame);
		cudaSafeCall(nvjpegEncodeImage(m_handler.nv_handle, m_handler.encoder_state, m_handler.encoder_params,
			&m_nvimg[0], NVJPEG_INPUT_BGRI, width, height, stream));
		cudaSafeCall(cudaGetLastError());
		//stopCudaTimerTag("[NvJpeg]", encode_frame);

		//startCudaTimer(retrieve_result);
		size_t length = 0;
		cudaSafeCall(nvjpegEncodeRetrieveBitstream(m_handler.nv_handle, m_handler.encoder_state, NULL, &length, stream));
		buf.resize(length);
		cudaSafeCall(nvjpegEncodeRetrieveBitstream(m_handler.nv_handle, m_handler.encoder_state, buf.data(), &length, stream));
		//stopCudaTimerTag("[NvJpeg]", retrieve_result);

		m_nvimg.clear();
		cudaSafeCall(cudaGetLastError());
		return EXIT_SUCCESS;
	}

	///////////////////////////////////// download /////////////////////////////////////
	// download RGB/BGR in nvjpegImage_t
	void NvJpegHelper::downloadNvJpegImage(const nvjpegImage_t &iout, int width, int height, cv::Mat &frame,
		cudaStream_t stream)
	{
		const unsigned char *d_chanR = iout.channel[0];
		int pitchR = iout.pitch[0];
		const unsigned char *d_chanG = iout.channel[1];
		int pitchG = iout.pitch[1];
		const unsigned char *d_chanB = iout.channel[2];
		int pitchB = iout.pitch[2];

		cv::Mat vchanR(height, width, CV_8U);
		cv::Mat vchanG(height, width, CV_8U);
		cv::Mat vchanB(height, width, CV_8U);
		unsigned char *chanR = vchanR.data;
		unsigned char *chanG = vchanG.data;
		unsigned char *chanB = vchanB.data;
		checkCudaErrors(cudaMemcpy2DAsync(chanR, (size_t)width, d_chanR, (size_t)pitchR,
			width, height, cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaMemcpy2DAsync(chanG, (size_t)width, d_chanG, (size_t)pitchR,
			width, height, cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaMemcpy2DAsync(chanB, (size_t)width, d_chanB, (size_t)pitchR,
			width, height, cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		cv::merge(vector<Mat>({ vchanR, vchanG, vchanB }), frame);
	}

	// copy to texture
	void NvJpegHelper::copyNvJpegImage(const nvjpegImage_t &iout, int width, int height,
		Texture2DUchar4 &t_color, cudaStream_t stream)
	{
		const unsigned char *d_chanR = iout.channel[0];
		int pitchR = iout.pitch[0];
		const unsigned char *d_chanG = iout.channel[1];
		int pitchG = iout.pitch[1];
		const unsigned char *d_chanB = iout.channel[2];
		int pitchB = iout.pitch[2];
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
		t_color.create(height, width);
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
		surfelwarp::cudaMergeColorToTexture(d_chanR, pitchR, d_chanG, pitchG, d_chanB, pitchB, width, height,
			t_color.surface(), stream);
	}

}
