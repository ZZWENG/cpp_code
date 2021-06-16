#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "../utils/types.h"

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

// matrix helpers from cuda tutorial
typedef struct
{
    int width;
    int height;
    int stride;
    real *elements;
} Matrix;

__device__ 
real GetElement(const Matrix A, int row, int col);

__device__ 
void SetElement(Matrix A, int row, int col, real value);

__device__ 
Matrix GetSubMatrix(Matrix B, int row, int col, int height, int width);

__global__
void myGEMMSharedMemoryKernel(Matrix A, Matrix B, Matrix C, real alpha, real beta, int M, int N, int K);
__global__
void myGEMMKernel(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta, int M, int N, int K);
int myGEMM(real* A, real* B, real* C, real* alpha, real* beta, int M, int N, int K);

__global__
void repmatKernel(real* mat1, real* mat2, int M, int N);
void GPUrepmat(real* mat, real* mat2, int M, int N);

__global__
void rowSumKernel(real* mat1, real* mat2, int M, int N);
void GPUrowSum(real* mat, real* output_vec, int M, int N);

__global__
void colSumKernel(real* mat1, real* mat2, int M, int N);
void GPUcolSum(real* mat, real* output_vec, int M, int N);

__global__
void divKernel(real* mat1, real* mat2, int M, int N);
void GPUdiv(real* ddata, real* dresult, int M, int N);

__global__
void sigmoidKernel(real* mat1, real* mat2, int M, int N);
void GPUsigmoid(real* mat, real* mat2, int M, int N);

__global__
void exponentialKernel(real* mat1, real* mat2, int M, int N);
void GPUexp(real* mat, real* mat2, int M, int N);

__global__
void addmat(real* mat1, real* mat2, real* output_mat, int M, int N);
void GPUaddition(real* mat, real* mat2, real* output_mat, real alpha, real beta, int M, int N);

__global__
void transpose(real* mat, real* output_mat, int M, int N);
void GPUtranspose(real* mat, real* output_mat, int M, int N);

__global__ 
void multKernel(real *mat1, real *mat2, real *output_mat, int M, int N);
void GPUmult(real* mat1, real* mat2, real* output_mat, int M, int N);

#endif
