#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include "TimingCPU.h"
#include <iostream>
#include "TimingGPU.cuh"
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;


/*Device Function - Offloading task to GPU*/
__global__ void gpu_vectorAddition(const int *a, const int *b, int *c, int size) {
	
	int i = blockIdx.x * blockDim.x +  threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}

void cpu_vectorAddittion(const int* a, const int* b, int* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

void randomIntialize(int * a, int * b, int size) {

	time_t t;
	srand((unsigned)time(&t));

	for (size_t i = 0; i < size; i++) {
		a[i] = (int)(rand() & 0xFF); // Between 0 - 255;
	}

	for (size_t i = 0; i < size; i++) {
		b[i] = (int)(rand() & 0xFF); // Between 0 - 255;
	}

	printf(" Array A : ");
	for (int i = 0; i < size; i++) {
		printf(" %d", a[i]);
	}
	printf(" \n\n........................................................................ \n\n");
	printf(" Array B : ");
	for (int i = 0; i < size; i++) {
		printf(" %d", b[i]);
	}
	printf(" \n\n........................................................................ \n\n");
}

void printResults(int * cpu_result, int * gpu_result, int size) {
	printf(" CPU Result : ");
	for (int i = 0; i < size; i++) {
		printf(" %d", cpu_result[i]);
	}
	printf(" \n\n........................................................................ \n\n");
	printf(" GPU Result : ");
	for (int i = 0; i < size; i++) {
		printf(" %d", gpu_result[i]);
	}
	printf(" \n\n........................................................................ \n\n");

}

/*HOST = CPU functionality , invokes kernel */

int main()
{
	const int arr_Size = 256;
	const int block_size = 32;
	const int bytes = arr_Size * sizeof(int);

	//int a[arr_Size] = {1,2,3,4};
	//int b[arr_Size] = {0,2,3,1};
	//int* gpu_result = (int *)malloc(bytes);

	int* host_a, * host_b, *host_c,  * gpu_result;

	host_a = (int *)malloc(bytes);
	host_b = (int*)malloc(bytes);
	host_c = (int*)malloc(bytes);
	gpu_result = (int*)malloc(bytes);

	randomIntialize(host_a, host_b, arr_Size);

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	auto t1 = Clock::now();
	TimingCPU timer_CPU;
	timer_CPU.StartCounter();
	//memset(gpu_result, 0 , bytes);
	cpu_vectorAddittion(host_a, host_b, host_c , arr_Size);
	gpu_end = clock();
	auto t2 = Clock::now();
	//printf("CPU Execution time = %4.6f \n \n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
	//std::cout << "Delta t2-t1: "
	//	<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
	//	<< " nanoseconds" << std::endl;
	//printf("CPU Execution time = %.20lf \n \n", (gpu_end - gpu_start) / (double)CLOCKS_PER_SEC);
	std::cout << "CPU Timing = " << timer_CPU.GetCounter() << " ms" << std::endl;
	//double cpu_end = timer_CPU.GetCounter();
	/*...................................................... $$ DEVICE $$ .............................................................*/

	int* d_a, * d_b, * d_c;

	cudaMalloc((void**)&d_a, bytes);
	cudaMalloc((void**)&d_b, bytes);
	cudaMalloc((void**)&d_c, bytes);
	
	// Copy data from host to device;
	cudaMemcpy(d_a, host_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, host_b, bytes, cudaMemcpyHostToDevice);

	dim3 block(block_size); // Four threads per block
	dim3 grid((arr_Size / block.x) + 1); // Only 1 block in a grid

	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	TimingGPU timer_GPU;
	timer_GPU.StartCounter();
	
	gpu_vectorAddition << < grid, block>> > (d_a, d_b, d_c, arr_Size);
	
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	//std::cout << "GPU Timing = " << timer_GPU.GetCounter() << " ms" << std::endl;

	cudaMemcpy(gpu_result, d_c, bytes, cudaMemcpyDeviceToHost);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); // returns in the first argument the number of milliseconds time elapsed between the recording of start and stop

	printResults(host_c, gpu_result, arr_Size);

	//printf("GPU Elapsed Execution time = %.20lf\n" , milliseconds);
	//std::cout << "CPU Timing = " << cpu_end << " ms" << std::endl;
	std::cout << "GPU Execution time = " << milliseconds << " ms" << std::endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(host_a);
	free(host_b);
	free(host_c);
	free(gpu_result);

	return 0;
}
