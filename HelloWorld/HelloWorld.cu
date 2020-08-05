#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



__global__ void helloCuda() {
	printf("Hello Aman and Sharach ..........\n");
}


/*
int main()
{
	dim3 block(4); // 4 threads per block;
	dim3 grid(8); // 8x4 = 32 threads; 1 grid = 8 blocks;

	helloCuda << <grid, block >> > ();
}
*/