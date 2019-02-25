#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// includes, kernels
#include "vector_dot_product_kernel.cu"

void run_test(unsigned int);
void compute_on_device(float *, float *,float *,int);
extern "C" float compute_gold( float *, float *, unsigned int);

int 
main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

void 
run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	float *C = (float *)malloc(NUM_BLOCKS);
	float gpu_result = 0.0f;
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	for(unsigned int i = 0; i < NUM_BLOCKS; i++){
		C[i] = 0.0f;
	}
	printf("Generating dot product on the CPU. \n");
	float reference = compute_gold(A, B, num_elements);
    
	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	compute_on_device(A, B, C, num_elements);
	for(unsigned int i = 0; i<NUM_BLOCKS; i++){
		gpu_result += C[i];
	}

	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    printf("Epsilon: %f. \n", fabsf(reference - gpu_result));

	// cleanup memory
	free(A);
	free(B);
	free(C);
	return;
}

/* Edit this function to compute the dot product on the device using atomic intrinsics. */
void 
compute_on_device(float *A_on_host, float *B_on_host, float *C_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL; 
	
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements * sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate space for the result vector on the GPU
	cudaMalloc((void**)&C_on_device, NUM_BLOCKS * sizeof(float));
	cudaMemcpy(C_on_device, C_on_host, NUM_BLOCKS * sizeof(float), cudaMemcpyHostToDevice);
	// Set up the execution grid on the GPU
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	dim3 grid(NUM_BLOCKS,1);
	
	// Launch the kernel
	vector_dot_product_kernel<<<grid, thread_block>>>(A_on_device, B_on_device, C_on_device, num_elements);

	cudaMemcpy(C_on_host, C_on_device, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(A_on_device);
	cudaFree(B_on_device);
	cudaFree(C_on_device);
}
 

