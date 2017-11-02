#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>


// data on the CPU = h_
// data on the GPU = d_

// Kernel program
// __global__ - declaration specifier (declspec) - this is the way
// CUDA knows that this is a kernel
__global__ void square(float * d_out, float * d_in)
{
	int idx = threadIdx.x;
	// threadIdx is a C struct (dim3) with 3 members - .x | .y | .z
	float f = d_in[idx];
	d_out[idx] = f * f;
}
// both the input arguement pointers must be allocated on the GPU

int main(int argc, char ** argv)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	int i, j;

	// generate the input array
	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = float(i);
	}

	// GPU memory pointers
	float *d_in;
	float *d_out;

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	// cudaMemcpy(destination, source, size, options)
	// options - cudaMemcpyHostToDevice | cudaMemcpyDeviceToHost | cudaMemcpyDeviceToDevice


	// Kernel Call
	// cuda launch operator - indicated by <<< >>>(arg1, arg2)
	square<<<1, ARRAY_SIZE>>> (d_out, d_in);
	// launch the kernel square on 1 block with 64 threads
	// launch 64 copies of the kernel on 64 threads


	// transfer the results back to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (i = 0; i < ARRAY_SIZE; i++)
	{
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	// free the GPU memory
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
