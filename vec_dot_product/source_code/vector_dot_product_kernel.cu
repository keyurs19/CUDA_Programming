#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_
#define NUM_GRID 160
#define TILE_SIZE 1024
/* Edit this function to complete the functionality of dot product on the GPU using atomics. 
	You may add other kernel functions as you deem necessary. 
 */
__global__ void vector_dot_product(float *Ad, float *Bd, float *Cd, int n)
{
	// Declare shared memory
	__shared__ float aTile [TILE_SIZE];
	__shared__ float bTile [TILE_SIZE];
	__shared__ float cTile [TILE_SIZE];
 	
	// Calculate thread index, block index and position in vector
	int tidx = threadIdx.x;
	int threadX = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	// Partial Sum
	float ps = 0.0f;
	
	 
	while(threadX < n){
		// Populate SM
		aTile[tidx] = Ad[threadX];
		bTile[tidx] = Bd[threadX];
		__syncthreads();
		// Calculate Partial Sum
		ps += aTile[tidx] * bTile[tidx];
	cTile[threadX] = ps;
	__syncthreads();
	}
	
	stride = stride / 2;	
	while(stride>0){
		cTile[tidx] += cTile[tidx + stride];
		__syncthreads();
		stride = stride / 2;
		}
	
	if (tidx == 0)
		Cd[tidx] = cTile[0];
	
}

#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
