#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "vec_dot_product.h"
// includes, kernels
#include "vector_dot_product_kernel.cu"

void compute_gold( float *, float *,float *, unsigned int);
Matrix allocate_matrix_on_gpu(const Matrix);
Matrix allocate_matrix(int, int, int);
void copy_matrix_to_device(Matrix, const Matrix);
float get_random_number(int, int);
int checkResults(float *, float *, int, float);
void copy_matrix_from_device(Matrix, const Matrix);
void run_test(unsigned int);
void compute_on_device(const Matrix, const Matrix, Matrix, int);
void check_for_error(char *);

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
	Matrix  A; // N x 1 vector
	Matrix  B; // N x 1 vector
	Matrix  C_cpu; // N x 1 vector
	Matrix  C_gpu; // N x 1 vector
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	A = allocate_matrix(size,1,1); // Create a random N x 1 vector
	B = allocate_matrix(size,1,1); // Create a random N x 1 vector 
	C_cpu = allocate_matrix(size,1,0); // Allocate memory for the output vector
	C_gpu = allocate_matrix(size,1,0); // Allocate memory for the output vector
	printf("Generating random vectors with values between [0, 10]. \n");	
	srand(time(NULL));
	
	printf("Generating dot product on the CPU. \n");
	compute_gold(A.elements, B.elements, C_cpu.elements, num_elements);
    
	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	compute_on_device(A, B, C_gpu, num_elements);

	printf("Result on CPU: %f, result on GPU: %f. \n", C_cpu, C_gpu);
	int res = checkResults(C_cpu.elements, C_gpu.elements, size, 1);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// cleanup memory
	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(B.elements); B.elements = NULL;
	free(C_cpu.elements); C_cpu.elements = NULL;
	free(C_gpu.elements); C_gpu.elements = NULL;
}

/* Edit this function to compute the dot product on the device using atomic intrinsics. */
void 
compute_on_device(const Matrix A_on_host, const Matrix B_on_host, Matrix C_on_host, int num_elements)
{
	Matrix Ad = allocate_matrix_on_gpu(A_on_host);
	Matrix Bd = allocate_matrix_on_gpu(B_on_host);
	Matrix Cd = allocate_matrix_on_gpu(C_on_host);
    	/* Copy matrices to device memory. */
	copy_matrix_to_device(Ad,A_on_host);	
	copy_matrix_to_device(Bd,B_on_host);
	copy_matrix_to_device(Cd,C_on_host);

	dim3 threads(TILE_SIZE,1);
	dim3 grid(NUM_GRID,1);
	struct timeval start, stop;	
		gettimeofday(&start, NULL);
	/* Launch kernel. */
	vector_dot_product<<<grid,threads>>>(Ad.elements, Bd.elements,Cd.elements,num_elements);
	cudaDeviceSynchronize();
	
    	gettimeofday(&stop, NULL);
		printf("Execution time using shared memory = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	copy_matrix_from_device(C_on_host,Cd);

	cudaFree(Ad.elements);
	cudaFree(Bd.elements);
	cudaFree(Cd.elements);
		
}
 
// This function checks for errors returned by the CUDA run time
void 
check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
 
void
copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Allocate a device matrix of same size as M.
Matrix 
allocate_matrix_on_gpu(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

Matrix 
allocate_matrix(int num_rows, int num_columns, int init)
{
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(-0.5, 0.5);
	}
    return M;
}

	
float 
get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
            break;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}

// Copy a device matrix to a host matrix.
void 
copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}
