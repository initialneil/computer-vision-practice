/* Pinned memory for large data transfer.
*  - 3 times faster than normal memory on GTX 2070.
*  All rights reserved. Prometheus 2020.
*  Contributor(s): Neil Z. Shao
*/
#pragma once
#include "cuda_utils.h"
#include "common/macro_utils.h"

namespace prometheus
{	
	template<typename T>
	class PinnedPointer 
	{
	public:
		explicit PinnedPointer() : m_ptr(nullptr), m_size(0) {}
		explicit PinnedPointer(size_t capacity) 
		{
			create(capacity);
		}

		~PinnedPointer()
		{
			//printf("[PinnedPointer] destroy\n");
			release();
			//cudaSafeCall(cudaDeviceSynchronize());
			//cudaSafeCall(cudaGetLastError());
		}

		//No implicit copy/assign/move
		SURFELWARP_NO_COPY_ASSIGN_MOVE(PinnedPointer);

		// handler
		T* Ptr() { return m_ptr; }
		const T* Ptr() const { return m_ptr; }
		
		// alloc N * sizeof(T)
		bool create(size_t N)
		{
			if (m_size >= N)
				return true;
			
			release();
			cudaSafeCall(cudaMallocHost((void**)&m_ptr, N * sizeof(T)));
			m_size = N;
		}

		// copy from
		void copyFrom(const T *data, int N)
		{
			memcpy(m_ptr, data, N * sizeof(T));
		}

		// release
		void release()
		{
			if (m_ptr) {
				cudaFreeHost(m_ptr);
				m_ptr = nullptr;
				m_size = 0;
			}
		}

	private:
		T *m_ptr = nullptr;
		size_t m_size = 0;
	};
	
}
