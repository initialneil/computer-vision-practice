/* Color convert using CUDA.
*  All rights reserved. Prometheus 2020.
*  Contributor(s): Neil Z. Shao
*/
#pragma once
#include <cuda/cuda_utils.h>

namespace prometheus
{
	static __host__ __device__ float3 rgb2hsv(float R, float G, float B)
	{
		float nNormalizedR = (float)R * 0.003921569F; // / 255.0F
		float nNormalizedG = (float)G * 0.003921569F;
		float nNormalizedB = (float)B * 0.003921569F;
		float nS = 0.f;
		float nH = 0.f;

		// Value
		float nV = fmaxf(nNormalizedR, nNormalizedG);
		nV = fmaxf(nV, nNormalizedB);

		// Saturation
		float nTemp = fminf(nNormalizedR, nNormalizedG);
		nTemp = fminf(nTemp, nNormalizedB);
		float nDivisor = nV - nTemp;

		if (nDivisor < 0.00001) {
			// undefined, maybe nan?
			return make_float3(0, 0, nV);
		}

		if (nV == 0.0F) // achromatics case
		{
			nS = 0.0F;
			nH = 0.0F;
		}
		else // chromatics case
			nS = nDivisor / nV;

		// Hue:
		float nCr = (nV - nNormalizedR) / nDivisor;
		float nCg = (nV - nNormalizedG) / nDivisor;
		float nCb = (nV - nNormalizedB) / nDivisor;
		if (nNormalizedR == nV)
			nH = nCb - nCg;
		else if (nNormalizedG == nV)
			nH = 2.0F + nCr - nCb;
		else if (nNormalizedB == nV)
			nH = 4.0F + nCg - nCr;

		//nH = nH * 0.166667F; // / 6.0F       
		//if (nH < 0.0F)
		//	nH = nH + 1.0F;

		// degrees
		nH *= 60.0;
		if (nH < 0.0)
			nH += 360.0;

		return make_float3(nH, nS, nV);
	}

	static __host__ __device__ float3 rgb2hsv_v2(float r, float g, float b)
	{
		float3 hsv;

		double min = r < g ? r : g;
		min = min < b ? min : b;

		double max = r > g ? r : g;
		max = max > b ? max : b;

		// v
		hsv.z = max;

		double delta = max - min;
		if (delta < 0.00001) {
			hsv.y = 0;
			hsv.x = 0; // undefined, maybe nan?
			return hsv;
		}

		// NOTE: if Max is == 0, this divide would cause a crash
		if (max > 0.0) {
			hsv.y = (delta / max);	// s
		}

		else {
			// if max is 0, then r = g = b = 0              
			// s = 0, h is undefined
			hsv.y = 0.0;
			hsv.x = NAN;                            // its now undefined
			return hsv;
		}

		// > is bogus, just keeps compilor happy
		if (r >= max)
			hsv.x = (g - b) / delta;        // between yellow & magenta
		else
			if (g >= max)
				hsv.x = 2.0 + (b - r) / delta;  // between cyan & yellow
			else
				hsv.x = 4.0 + (r - g) / delta;  // between magenta & cyan

		// degrees
		hsv.x *= 60.0;

		if (hsv.x < 0.0)
			hsv.x += 360.0;
		return hsv;
	}


#define nCIE_LAB_D65_xn 0.950455F
#define nCIE_LAB_D65_yn 1.0F
#define nCIE_LAB_D65_zn 1.088753F
	static __host__ __device__ float3 rgb2lab(float R, float G, float B)
	{
		float blue = (float)B / 255.0;
		float green = (float)G / 255.0;
		float red = (float)R / 255.0;

		float x = red * 0.412453 + green * 0.357580 + blue * 0.180423;
		float y = red * 0.212671 + green * 0.715160 + blue * 0.072169;
		float z = red * 0.019334 + green * 0.119193 + blue * 0.950227;

		x = x / 0.950456;
		z = z / 1.088754;

		float l, a, b;

		float fx = x > 0.008856 ? cbrt(x) : (7.787 * x + 16. / 116.);
		float fy = y > 0.008856 ? cbrt(y) : 7.787 * y + 16. / 116.;
		float fz = z > 0.008856 ? cbrt(z) : (7.787 * z + 16. / 116.);

		l = y > 0.008856 ? (116.0 * cbrt(y) - 16.0) : 903.3 * y;
		a = (fx - fy) * 500.0;
		b = (fy - fz) * 200.0;

		return make_float3(l, a, b);
	}

}
