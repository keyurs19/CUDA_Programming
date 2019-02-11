#include <stdio.h>
#include <math.h>
#include <float.h>

void compute_gold( float *, float *, unsigned int);

void
compute_gold(float*  A, float* B, float* C, unsigned int num_elements){
    unsigned int i;
    double dot_product = 0.0f; 

    for( i = 0; i < num_elements; i++) 
        dot_product += A[i] * B[i];

    	C[i] = (float)dot_product;
}

