#include <stdio.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 64 // Specify a constant value for array length.
#define TPB 32 // Specify a constant value for threads per block.

// A scaling function to convert integers 0,1,...,N-1
// to evenly spaced floats ranging from 0 to 1.
__device__ float scale(int i, int n)
{
  return ((float)i) / (n - 1);
}

// Compute the distance between 2 points on a line.
__device__ float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}

__global__ void distanceKernel(float *d_out, float ref, int len)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	const float x = scale(i, len);

	d_out[i] = distance(x, ref);

	printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);

}

int main()
{

  const float ref = 0.5f;

  float* d_out = 0;
  
  cudaMalloc(&d_out, N * sizeof(float));

  distanceKernel << <N / TPB, TPB >> > (d_out, ref, N);

  cudaFree(d_out);

  return 0;
}