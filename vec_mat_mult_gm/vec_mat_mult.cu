/* Vector-matrix multiplication: Y = A * X.
 * Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// includes, project


// includes, kernels
#include "vec_mat_mult_kernel.cu"

#define MIN_NUMBER 1
#define MAX_NUMBER 4

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void compute_gold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void vec_mat_mult_on_device(const Matrix M, const Matrix N, Matrix P);
void print_matrix(const Matrix M);
float get_random_number(int, int);
int checkResults(float *, float *, int, float);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  A; // 4096 x 4096 matrix
	Matrix  X; // 4096 x 1 vector
	Matrix  Y; // 4096 x 1 vector
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1); // Create a random 4096 X 4096 matrix
	X  = allocate_matrix(MATRIX_SIZE, 1, 1); // Create a random 4096 x 1 vector 
	Y  = allocate_matrix(MATRIX_SIZE, 1, 0); // Allocate memory for the output vector 
	
	printf("Multiplying matrices on the GPU \n");
		
	// Perform the vector-matrix multiplication on the GPU
    	vec_mat_mult_on_device(A, X, Y);
    	
	printf("Multiplying matrices on the CPU. \n");
    	struct timeval start, stop;	
		gettimeofday(&start, NULL);	

    	// compute the vector-matrix multiplication on the CPU for comparison
    	Matrix reference = allocate_matrix(MATRIX_SIZE, 1, 0);
    	
	//unsigned int timer;
        //cutCreateTimer(&timer);
	//cutStartTimer(timer);

	compute_gold(reference.elements, A.elements, X.elements, A.num_rows, A.num_columns);

    	gettimeofday(&stop, NULL);
		printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	

	//cutStopTimer(timer);
        //printf("Kernel execution time on the CPU: %f seconds. \n", (float)cutGetTimerValue(timer)/1000.0);
	
    	// check if the device result is equivalent to the expected solution
    	//int size_elements = NUM_ROWS;
    	//CUTBoolean res = cutComparefe(reference.elements, Y.elements, size_elements, 0.0001f);
    	//printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    int num_elements = MATRIX_SIZE;
	int status = checkResults(reference.elements, Y.elements, num_elements, 0.1f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");
	// Free host matrices
    	free(A.elements); A.elements = NULL;
    	free(X.elements); X.elements = NULL;
    	free(Y.elements); Y.elements = NULL;

	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void vec_mat_mult_on_device(const Matrix A, const Matrix X, Matrix Y){
	//Interface host call to the device kernel code and invoke the kernel
	/* Allocate device memory. */	
	Matrix Ad = allocate_matrix_on_gpu(A);
	Matrix Xd = allocate_matrix_on_gpu(X);
	Matrix Yd = allocate_matrix_on_gpu(Y);
	
	/* Copy matrices to device memory. */
	copy_matrix_to_device(Ad,A);	
	copy_matrix_to_device(Xd,X);
	copy_matrix_to_device(Yd,Y);
		
	/* Set up execution grid. */
	dim3 threads(1,NUM_THREADS);
	dim3 grid(1,MATRIX_SIZE/NUM_THREADS);
	struct timeval start, stop;	
		gettimeofday(&start, NULL);
	/* Launch kernel. */
	MatrixMulKernel<<<grid,threads>>>(Ad.elements, Xd.elements, Yd.elements);
	cudaThreadSynchronize();
	
    	gettimeofday(&stop, NULL);
		printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));	

	/* Check if kernel execution generated an error. */
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
		exit(-1);
	}

	/* Copy result from device to host. */
	copy_matrix_from_device(Y, Yd);

	/* Clean up memory on the GPU. */
	cudaFree(Ad.elements);
	cudaFree(Xd.elements);
	cudaFree(Yd.elements);
		
}

// Allocate a device matrix of same size as M.
Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init){
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_columns + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float get_random_number(int min, int max){
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
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}
