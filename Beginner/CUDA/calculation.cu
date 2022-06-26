#include "calculation.h"

/**
 * Perform calculation on d_d where d_v is a constant, and the results are stored in d_i
 **/

__global__ void calculation(int *d_d, int *d_i, int numElements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements) {
        d_i[i] = d_d[i] + d_v; 
    }
}

__host__ int *allocateRandomHostMemory(int numElements) {

    size_t size = numElements * sizeof(int);
    int *h_d = (int *)malloc(size);

    if (h_d == NULL) {
        fprintf(stderr, "Failed to allocate host vectors \n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i) {
        h_d[i] = rand() % 255
    }

    return h_d;
}

__host__ std::tuple<int *, int> readCsv(std::string filename) {

    std::vector<int> tempResult;

    std::ifstream myFile(filename)

    if (!myFile.is_open()) throw std::runtime_error("Could not open file");

    std::string line, colname;
    int val;

    while(std::getline(myFile, line))
    {
        std::stringstream ss(line);

        while(ss >> val){
            tempResult.push_back(val);
            if(ss.peek() == ',') ss.ignore();
        }
    }

    myFile.close();
    int numElements = tempResult.size();
    int result[numElements];
    std::copy(tempResult.begin(), tempResult.end(), result);

    return {result, numElements};
}

__host__ std::tuple<int *, int *> allocateDeviceMemory(int numElements) {
    
    int *d_d = NULL;
    size_t size = numElements * sizeof(int);
    cudaError_t err = cudaMalloc(&d_d, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_d - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_i;
    err = cudaMalloc(&d_i, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector d_i - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return {d_d, d_i};
}

__host__ void copyFromHostToDevice(int h_v, int *h_d, int *d_d, int numElements) {

    size_t size = numElements * sizeof(int);

    cudaError_t err = cudaMemcpy(d_d, h_d, size, copyFromHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpyToSymbol(d_v, &h_v, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy constant int d_v from host to device - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(int *d_d, int *d_i, int numElements, int threadsPerBlock) {

    int blocksPerGrid = (numElements * threadsPerBlock - 1) / threadsPerBlock;
    calculation<<<blocksPerGrid, threadsPerBlock>>>(d_d, d_i, numElements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch kernel function - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(int *d_i, int *h_i, numElements){

    cout << "Copying from Device to Host \n";
    size_t size = numElements * sizeof(int);

    cudaError_t err = cudaMemcpy(h_i, d_i, size, cudaMemcpyDeviceToHost);
    
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy d_i from device to host - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void deallocateMemory(int *d_d, int *d_i){

    cudaError_t err = cudaFree(d_d);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector d_d - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_i);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device int d_i - error code: %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void cleanUpDevice() {

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits

    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to deinitialize")
    }
}

int main(int argc, char *argv[]){ 

    srand(time(0));
    int numElements = 10;
    int h_v = -1;
    int *h_d;
    int threadsPerBlock = 256;

    bool sortInputData = true;
    if(argc > 1)
    {
        std::string sortInputDataStr(argv[1]);
        if(sortInputDataStr == "false")
        {
            sortInputData = false;
        }
    }

    if(argc > 2)
    {
        threadsPerBlock = atoi(argv[2]);
        if(argc > 3)
        {
            numElements = atoi(argv[3]);
        }
    }
    if(argc > 4)
    {
        h_v = atoi(argv[4]);
        std::string inputFilename(argv[5]);
        tuple<int *, int>csvData = readCsv(inputFilename);
        h_d = get<0>(csvData);
        numElements = get<1>(csvData);
    }
    else 
    {
        h_d = allocateRandomHostMemory(numElements);
        // This is a bit hard coded, could consider coming up with another randomization scheme
        h_v = rand() % 255;
    }

    if(sortInputData)
    {
        sort(h_d, h_d + numElements);
    }

    int *h_i = (int *)malloc(numElements * sizeof(int));
    cout << "Data: ";
    for (int i = 0; i < numElements; ++i)
    {
        cout << h_d[i] << " ";
        h_i[i]=0;
    }
    cout << "\n";

    printf("Searching for value: %d \n", h_v);
    auto[d_d, d_i] = allocateDeviceMemory(numElements);
    copyFromHostToDevice(h_v, h_d, d_d, numElements);

    executeKernel(d_d, d_i, numElements, threadsPerBlock);

    copyFromDeviceToHost(d_i, h_i, numElements);

    cout << "Calculation results: ";
    for (int i = 0; i < numElements; ++i)
    {
        cout << h_i[i] << "\n";
    }

    deallocateMemory(d_d, d_i);

    cleanUpDevice();
    return 0;
}