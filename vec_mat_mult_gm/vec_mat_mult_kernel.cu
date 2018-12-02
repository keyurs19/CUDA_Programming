/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "vec_mat_mult.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
	int threadX = threadIdx.y; 
	  int blockY = blockIdx.y;
	  int yPos = NUM_THREADS * blockY + threadX; // Obtain the corresponding row number

	  float partialSum = 0.0f;
	  for(int i = 0; i < MATRIX_SIZE; i++){
				 partialSum += Ad[MATRIX_SIZE * yPos + i] * Xd[i];
	  }
	  Yd[yPos] = partialSum;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
