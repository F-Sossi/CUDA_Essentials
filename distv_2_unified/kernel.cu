
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 64
#define TPB 32

float scale(int i, int n)
{
	return ((float)i / (n - 1));
}

__device__ 
float distance(float x1, float x2)
{
	return sqrt((x2 - x1) * (x2 - x1) );
}

__global__
void distanceKernel(float* d_out, float ref, float* d_in)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const float x = d_in[i];
	d_out[i] = distance(x, ref);
	printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

int main()
{
	const float ref = 0.5f;

	float* in = 0;
	float* out = 0;

	// allocate managed memory for input and output
	cudaMallocManaged(&in, N * sizeof(float));
	cudaMallocManaged(&out, N * sizeof(float));

	// Computer scaled input values
	for (int i = 0; i < N; i++)
	{
		in[i] = scale(i, N);
	}

	// launch kernel
	
 	distanceKernel << <N/TPB, TPB >> > (out, ref, in);
	
	// wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	
	// free memory
	cudaFree(in);
	cudaFree(out);
	
	return 0;
}
