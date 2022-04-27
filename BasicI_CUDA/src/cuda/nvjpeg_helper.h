/* MJpeg decode using NvJpeg after CUDA 10.2.
*  All rights reserved. Prometheus 2020.
*  Contributor(s): Neil Z. Shao
*/
#pragma once
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <opencv2/opencv.hpp>
#include "promdata/texture2d.h"
#include "cuda/DeviceBufferArray.h"
#include "cuda/pinned_pointer.h"

#ifdef _WIN32
#pragma comment(lib, "nvjpeg.lib")
#endif

namespace prometheus
{
	struct NvJpegHelperOption
	{
		bool verbose = false;
		int dev = 0;
		nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_BGR;
	};

	class NvJpegHelper
	{
	public:
		NvJpegHelper();
		~NvJpegHelper();
		void init();

	protected:
		// prepare buffers for RGBi output format
		int prepareBuffers(const std::vector<char> &file_data, size_t file_len,
			int &img_width, int &img_height, 
			nvjpegImage_t &ibuf, nvjpegImage_t &isz);
		int prepareBuffers(const std::vector<std::vector<char>> &file_data, std::vector<size_t> &file_len,
			std::vector<int> &img_width, std::vector<int> &img_height,
			std::vector<nvjpegImage_t> &ibuf,
			std::vector<nvjpegImage_t> &isz);
		int decodeImage(const std::vector<char> &img_data, size_t img_len,
			nvjpegImage_t &out, cudaStream_t stream);
		void release_buffers(std::vector<nvjpegImage_t> &ibuf);

		// routine
		int prepareInputData(const std::vector<cv::Mat> &rawJpegs, const std::vector<cudaStream_t> &streams,
			std::vector<std::vector<char>> &input_data,
			std::vector<size_t> &input_len,
			std::vector<cudaStream_t> &input_stream);
		int prepareInputData(const cv::Mat &rawJpegs, cudaStream_t stream,
			std::vector<char> &input_data,
			size_t &input_len);

	public:
		// decode
		int decodeFrames(const std::vector<cv::Mat> &rawJpegs, const std::vector<cudaStream_t> &streams);
		int decodeFrames(const std::vector<cv::Mat> &rawJpegs, std::vector<cv::Mat> &decodedFrames, 
			const std::vector<cudaStream_t> &streams);
		int decodeFrames(const std::vector<cv::Mat> &rawJpegs, std::vector<Texture2DUchar4 *> &decodeColorTs,
			const std::vector<cudaStream_t> &streams);

		void allocBuffer(int num);
		int decodeFrame(int i, const cv::Mat &rawJpeg, Texture2DUchar4 &t_color, cudaStream_t stream = 0);
		int decodeFrame(int i, const cv::Mat &rawJpeg, cv::Mat &color, cudaStream_t stream = 0);

		// encode
		void uploadToEncodeBuffer(const cv::Mat &frame, cudaStream_t stream = 0);
		int encodeFrame(const cv::Mat &frame, std::vector<uchar> &buf, cudaStream_t stream = 0);
		int encodeFrame(const pcl::gpu::DeviceArray<uchar> &darr_frame, int width, int height,
			std::vector<uchar> &buf, cudaStream_t stream = 0);

		// helper functions
		auto& getOutputBuffer() { return m_nvimg; }

		// download RGB/BGR in nvjpegImage_t
		static void downloadNvJpegImage(const nvjpegImage_t &iout, int width, int height, cv::Mat &frame,
			cudaStream_t stream = 0);
		// copy to texture
		static void copyNvJpegImage(const nvjpegImage_t &iout, int width, int height, 
			Texture2DUchar4 &t_color, cudaStream_t stream = 0);

	private:
		NvJpegHelperOption m_options;

		struct {
			nvjpegJpegState_t nvjpeg_state = 0;
			nvjpegHandle_t nv_handle = 0;
			nvjpegEncoderState_t encoder_state = 0;
			nvjpegEncoderParams_t encoder_params = 0;

			PinnedPointer<uchar4> pinned_ptr;
			Texture2DUchar4 t_color;
			surfelwarp::DeviceBufferArray<uchar> d_input;
		} m_handler;

		std::vector<int> m_widths, m_heights;
		std::vector<nvjpegImage_t> m_nvimg, m_isz;
	};

}
