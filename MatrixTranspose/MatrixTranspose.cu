#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <string>

#include "TimingCPU.h"



int generateRandomNumber() {
	//srand(time(0));
	return (rand() % 10) + 1;
}

void populateMatrix(int * matrix, int size) {
	srand(time(0));
	for (int i = 0; i < size; i ++) {
		matrix[i] = generateRandomNumber();
	}
}

void printMatrix(int* matrix,int nx, int ny) {
	printf("\n \n");
	for (int i = 0; i < nx; i++) {

		for (int j = 0; j < ny; j++) {
			int offset = i * ny;
			printf("\t%d ", matrix[offset + j]);
		}

		printf("\n");
	}
}

void cpu_execution(int * result, int * matrix, int nx, int ny) {
	for (int i = 0; i < ny; i++) {
		for (int j = 0; j < nx; j ++) {
			result[j * ny + i] = matrix[i * nx + j];
		}
	}
}

__global__ void gpu_execution(int* d_result, int* d_matrix, int nx, int ny) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if ( ix < nx && iy < ny) {
		d_result[ix * ny + iy] = d_matrix[iy * nx + ix];
	}
}

int main() {
	// Matrix x * y = nx * ny
	const int nx = 4;
	const int ny = 4;
	const int size = nx * ny; // Matrix Size
	const int bytes = size * sizeof(int);

	const int block_x = 2; // x-dimension - 2 threads per block
	const int block_y = 2; // y-dimension - 2 threads per block


	//-------------------- IMPLEMENT HOST LOGIC -----------------------------------------

	double cpu_execution_time;
	TimingCPU cpu_profile;

	int* h_matrix = (int *)malloc(bytes);
	int* cpu_result = (int*)malloc(bytes);
	int* gpu_result = (int*)malloc(bytes);

	populateMatrix(h_matrix,size);
	printf(" \n Input Matrix : ");
	printMatrix(h_matrix, nx, ny);

	cpu_profile.StartCounter();
	cpu_execution(cpu_result, h_matrix, nx, ny);
	cpu_execution_time = cpu_profile.GetCounter();

	//-------------------- IMPLEMENT DEVICE LOGIC -----------------------------------------

	int* d_matrix, * d_result;
	float d_input_copy_time = 0;
	float d_result_copy_time = 0;
	float d_execution_time = 0;

	cudaEvent_t exec_start, exec_stop, inp_start, inp_stop, res_start, res_stop;
	cudaEventCreate(&exec_start);
	cudaEventCreate(&exec_stop);
	cudaEventCreate(&inp_start);
	cudaEventCreate(&inp_stop);
	cudaEventCreate(&res_start);
	cudaEventCreate(&res_stop);
	//cudaEventRecord(start);

	cudaMalloc((void **)&d_matrix, bytes);
	cudaMalloc((void**)&d_result, bytes);

	dim3 block(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);

	cudaEventRecord(inp_start);
	cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);
	cudaEventRecord(inp_stop);
	cudaEventSynchronize(inp_stop);
	cudaEventElapsedTime(&d_input_copy_time, inp_start, inp_stop);

	cudaEventRecord(exec_start);
	gpu_execution <<<grid, block>>>(d_result, d_matrix, nx, ny);
	cudaEventRecord(exec_stop);
	cudaEventSynchronize(exec_stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&d_execution_time, exec_start, exec_stop);
	

	cudaEventRecord(res_start);
	cudaMemcpy(gpu_result, d_result, bytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(res_stop);
	cudaEventSynchronize(res_stop);
	cudaEventElapsedTime(&d_result_copy_time, res_start, res_stop);

	double throughput = 2 * size / (((d_execution_time) / 1000) * 1000000000); //cpu_execution_time / d_execution_time; // Only interms of execution time
	double speedup = cpu_execution_time / (d_input_copy_time + d_execution_time + d_result_copy_time); // interms of entire task computation

	//------------------------------------------------------------------------------------

	printf(" \n CPU Result : ");
	printMatrix(cpu_result, nx, ny);
	printf(" \n GPU Result : ");
	printMatrix(gpu_result, nx, ny);

	//------------------------------ BENCHMARKS -----------------------------------------
	printf(" \n Benchmarks : \n\n");
	std::cout << "\t[1] CPU Execution Time = " << cpu_execution_time << " ms \n" << std::endl;
	std::cout << "\t[2] Time Elapsed For Memory (Input) Transfer from Host to Device = " << d_input_copy_time << " ms \n" << std::endl;
	std::cout << "\t[3] Time Elapsed For Memory (Output) Transfer from Device to Host = " << d_result_copy_time << " ms \n" << std::endl;
	std::cout << "\t[4] GPU Execution Time = " << d_execution_time << " ms \n" << std::endl;
	std::cout << "\t[5] Total GPU Execution Time = " << d_input_copy_time + d_execution_time + d_result_copy_time << " ms \n" << std::endl;
	std::cout << "\t[6] Speed up (Execution time old / Execution time new) = " << speedup << "\n" << std::endl;
	std::cout << "\t[7] Throughput = " << throughput << "  GFLOP/s \n" << std::endl;

	cudaFree(d_matrix);
	cudaFree(d_result);

	free(h_matrix);
	free(cpu_result);
	free(gpu_result);
}

