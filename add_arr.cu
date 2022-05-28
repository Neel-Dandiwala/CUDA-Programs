#include <iostream>
#include <math.h>

//Kernel function
__global__
void add(int n, float *x, float *y){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    printf("Thread Index is %d\n",index);
    printf("Block Dimension is %d\n",stride);
    for(int i = index; i < n; i+=stride){
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1<<10;
    float *x, *y;
    //Alocate Unified Memory
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    //Initialize arrays on host
    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    //Run kernel on GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks,blockSize>>>(N, x, y);

    //Wait for GPU finish before accessing the host
    cudaDeviceSynchronize();

    //Error check
    float maxError = 0.0f;
    for(int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout<<"Max Error: "<<maxError<<std::endl;

    //Free Memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}