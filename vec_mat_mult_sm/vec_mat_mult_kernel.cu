/* Vector-Matrix multiplication: Y = A * X
 * Device code
 */
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
	int threadX = threadIdx.y; 
	  int blockY = blockIdx.y;
	  int yPos = TILE_SIZE * blockY + threadX; // Obtain the corresponding row number

	  float partialSum = 0.0f;
	  for(int i = 0; i < MATRIX_SIZE; i++){
				 partialSum += Ad[MATRIX_SIZE * yPos + i] * Xd[i];
	  }
	  Yd[yPos] = partialSum;
}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
		  // Declare shared memory
	__shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile[TILE_SIZE];

	// Calculate thread index, block index and position in matrix
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int yInMatrix = TILE_SIZE * blockY + threadY;

	// Clear partialSum for thread
	float partialSum = 0.0f;

	for (int i = 0; i < MATRIX_SIZE; i += TILE_SIZE) {
		// Populate shared memory for the current tile if within range
		aTile[threadY][threadX] = Ad[MATRIX_SIZE * yInMatrix + i + threadX]; // Bring TILE_SIZE elements per row of the A matrix into shared memory 
		if(threadY == 0) xTile[threadX] = Xd[i + threadX]; // Bring TILE_SIZE elements of the vector X into shared memory

		__syncthreads();

		// Compute partialSum for the current Tile
		/*
		for (int k = 0; k < TILE_SIZE; ++k)
			partialSum += aTile[threadY][k] * xTile[k];
		*/
		float aElement1;
		float xElement1;
		for (int k = 0; k < TILE_SIZE; k += 1){
			aElement1 = aTile[threadY][k]; xElement1 = xTile[k]; 		
			partialSum += aElement1 * xElement1; 
		}

		__syncthreads();
	}

	// Store partialSum
	if (threadX == 0) {
		Yd[yInMatrix] = (partialSum);
	}
	
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
