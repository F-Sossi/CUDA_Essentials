﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdio.h>

#define TPB 32

__device__
float distance(float x1, float x2)
{
	return sqrt((x2 - x1) * (x2 - x1));
}

__global__
void distanceKernel(float* d_out, float* d_in, float ref, int len)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const float x = d_in[i];
	d_out[i] = distance(x, ref);

	printf("i=%2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);


}

void distanceArray(float* out, float* in, float ref, int len)
{
	// declare GPU memory pointers
	float* d_in = 0, * d_out = 0;

	// allocate GPU memory
	cudaMalloc(&d_in, len * sizeof(float));
	cudaMalloc(&d_out, len * sizeof(float));

	// transfer the array to the GPU
	cudaMemcpy(d_in, in, len * sizeof(float), cudaMemcpyHostToDevice);

	// launch the kernel
	distanceKernel << <(len + TPB - 1) / TPB, TPB >> > (d_out, d_in, ref, len);

	// copy back the result array to the CPU
	cudaMemcpy(out, d_out, len * sizeof(float), cudaMemcpyDeviceToHost);

	// free the GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);
}