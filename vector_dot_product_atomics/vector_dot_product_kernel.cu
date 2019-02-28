
#define THREAD_BLOCK_SIZE 512
#define NUM_BLOCKS 320 // Define the size of a tile
/* This function uses a compare and swap technique to acquire a mutex/lock. */
__device__ void lock(int *mutex)
{	  
    while(atomicCAS(mutex, 0, 1) != 0);
}

/* This function uses an atomic exchange operation to release the mutex/lock. */
__device__ void unlock(int *mutex)
{
    atomicExch(mutex, 0);
}

__global__ void vector_dot_product_kernel_atomics(float *A, float *B, float *C, unsigned int num_elements,int *mutex)
{
	__shared__ float sum_per_thread[THREAD_BLOCK_SIZE];	
	unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Obtain the index of the thread
	unsigned int stride = blockDim.x * gridDim.x; 
	float sum = 0.0f; 
	unsigned int i = thread_id; 

	while(i < num_elements){
			  sum += A[i] * B[i];
			  i += stride;
	}

	sum_per_thread[threadIdx.x] = sum; // Copy sum to shared memory
	__syncthreads();

	i = blockDim.x/2;
	while(i != 0){
			  if(threadIdx.x < i) 
						 sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
			  __syncthreads();
			  i /= 2;
	}

	if(threadIdx.x == 0){
			  lock(mutex);	
			  *C += sum_per_thread[0];
			  unlock(mutex);
	}	
}
