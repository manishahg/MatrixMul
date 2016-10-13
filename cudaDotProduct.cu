#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

const int numElements =   1024;
const int threadsPerBlock =   256;
const int blocksPerGrid = (numElements+threadsPerBlock - 1) / threadsPerBlock;

__global__ void cuda_dot_kernel(float *a, float *b, float *c)
{
        int id = (blockIdx.x * blockDim.x) + threadIdx.x;
        int nextid = gridDim.x * blockDim.x;

        __shared__ float shared_cache [threadsPerBlock];
        int cacheIndex = threadIdx.x;
        float temp = 0;

        while(id < numElements)
        {
                temp+= a[id]*b[id];
                id += nextid;
        }

        shared_cache[cacheIndex] = temp;
         __syncthreads();

        //for reduction thread block must be power of 2
        for(i=blockDim.x/2; i!=0; i /= 2)
        {
                if(cacheIndex < i)
                        shared_cache[cacheIndex] += shared_cache[cacheIndex+i];
                __syncthreads();
        }

        if ( cacheIndex == 0)
        {
                c[blockIdx.x] = shared_cache[0];
        }
}

//  Host main routine
int main(void)
{
    int i;
    cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A,B,C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(blocksPerGrid*sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    float serial_product = 0.0;
    // Initialize the host input vectors
    for (i = 0; i < numElements; ++i)
    {
            h_A[i] = rand()/(float)RAND_MAX;
            h_B[i] = rand()/(float)RAND_MAX;
            serial_product += h_A[i]*h_B[i];
    }

    //Allocate the device input vector A
    float *d_A = NULL;
    cudaMalloc((void**)&d_A, size);

    // Allocate the device input vector B
    float *d_B = NULL;
    cudaMalloc((void**)&d_B,size);

    // Allocate the device output vector C
    float *d_C = NULL ;
    cudaMalloc( (void**)&d_C, blocksPerGrid*sizeof(float) );

    // Copy the host input vectors A and B to the device input vectors    
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Dot Vector  CUDA Kernel
    cuda_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // Copy device result  to the host result
    err = cudaMemcpy(h_C, d_C, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy  result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float result = 0;
    for(int k=0; k<blocksPerGrid; k++)
    {
        result += h_C[k];
    }


    // Free device global memory
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("Cuda Dot Product=%f\n", result);
    printf("Serial product  =%f\n", serial_product);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
                                                                  