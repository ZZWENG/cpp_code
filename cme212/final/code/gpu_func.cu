#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE 16
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 32
#define NUM_THREADS 1024


__device__ 
real GetElement(const Matrix A, int row, int col)
{
    return A.elements[col * A.stride + row];
}

__device__ 
void SetElement(Matrix A, int row, int col, real value)
{ 
    A.elements[A.stride * col + row] = value;
}

__device__ 
Matrix GetSubMatrix(Matrix B, int row, int col, int height, int width)
{
    Matrix Bsub;
    Bsub.height = height;
    Bsub.width = width;
    Bsub.stride = B.stride;
    Bsub.elements = &B.elements[B.stride * width * col + height * row];
    return Bsub;
}


__global__
void myGEMMSharedMemoryKernel(Matrix A, Matrix B, Matrix C, real alpha, real beta, int M, int N, int K) {
    int block_i = blockIdx.y;
    int block_j = blockIdx.x;
    int row = threadIdx.x; // index in the shared block
    int col = threadIdx.y;
    int global_row = blockIdx.y * blockDim.y + row;  // index in the original matrices
    int global_col = blockIdx.x * blockDim.x + col;

    Matrix Csub = GetSubMatrix(C, block_i, block_j, BLOCK_SIZE, BLOCK_SIZE);
    real Cvalue = 0; 
    
    int bound = BLOCK_SIZE;
    __shared__ real As[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ real Bs[BLOCK_SIZE][BLOCK_SIZE+1]; 

    int num_subs = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int m = 0; m < num_subs; m++){
        Matrix Asub = GetSubMatrix(A, block_i, m, BLOCK_SIZE, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, block_j, BLOCK_SIZE, BLOCK_SIZE);

        if ((m+1) * BLOCK_SIZE > K) bound = K - m * BLOCK_SIZE;
        if (global_row < M && col < bound) As[row][col] = GetElement(Asub, row, col); 
        if (global_col < N && row < bound) Bs[row][col] = GetElement(Bsub, row, col); 

        __syncthreads();
        for (int i = 0; i < bound; i++) Cvalue += alpha * As[row][i]*Bs[i][col];
        
        __syncthreads();
    }

    if (global_row < M && global_col < N)
        SetElement(Csub, row, col,  Cvalue + beta*GetElement(Csub, row, col));

}


__global__
void myGEMMKernel(real* __restrict__ A, real* __restrict__ B,
           real* __restrict__ C, real alpha, real beta, int M, int N, int K) {
    // naive implementation that computes
    // C[i, j] = alpha*sum_{k=0}^{K}(A[i,k]B[k,j]) + beta*C[i, j]
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < M && j < N){ 
        C[j*M + i] = beta*C[j*M + i]; 
        for (int k = 0; k < K; k++) {
            C[j*M + i] += alpha*A[k*M + i]*B[j*K + k];
        }
    }
}

int myGEMM(real* __restrict__ A, real* __restrict__ B, real* __restrict__ C, real* alpha, real* beta,
    int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x;
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y;
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    // myGEMMKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    Matrix MA, MB, MC;
    MA.height = MA.stride = M; MA.width = K; MA.elements = A; 
    MB.height = MB.stride = K; MB.width = N; MB.elements = B; 
    MC.height = MC.stride = M; MC.width = N; MC.elements = C; 
    myGEMMSharedMemoryKernel<<<numBlocks, threadsPerBlock>>>(MA, MB, MC, *alpha, *beta, M, N, K); 
    return 0;
}

__global__
void repmatKernel(real* mat1, real* mat2, int M, int N) {
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (i < M && j < N){ 
        mat2[j*M + i] = mat1[i]; 
    }
}

__global__
void exponentialKernel(real* mat1, real* mat2, int M, int N) {
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < M && j < N) mat2[j*M + i] = exp(mat1[j*M + i]);
}

void GPUexp(real* mat, real* mat2, int M, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int num_blocks_x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int num_blocks_y = (M + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(num_blocks_x, num_blocks_y);
    exponentialKernel<<<numBlocks, threadsPerBlock>>>(mat, mat2, M, N);
}

__global__
void repmat(real* mat1, real* mat2, int M, int N) {
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (j < N){ 
        for (int i = 0; i < M; i++){
                mat2[j*M + i] = mat1[j]; 
        }
    }
}

void GPUrepmat(real* mat1, real* mat2, int M, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y); 
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; 
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y; 
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    repmatKernel<<<numBlocks, threadsPerBlock>>>(mat1, mat2, M, N); 
}

__global__
void colSumKernel(real* mat1, real* mat2, int M, int N) {
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (j < N) { 
        mat2[j] = 0.0; 
        for (int i = 0; i < M; i++) {
            mat2[j] += mat1[j*M + i]; 
        }
    }
}

void GPUcolSum(real* mat, real* output_vec, int M, int N) {
    dim3 threadsPerBlock(NUM_THREADS);
    int num_blocks = (N + threadsPerBlock.x - 1)/threadsPerBlock.x;
    dim3 numBlocks(num_blocks);
    colSumKernel<<<numBlocks, threadsPerBlock>>>(mat, output_vec, M, N);
}

__global__
void rowSumKernel(real* mat1, real* mat2, int M, int N) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < M){ 
        mat2[i] = 0.0; 
        for (int j = 0; j < N; j++) {
            mat2[i] += mat1[j*M + i]; 
        }
    }
}

void GPUrowSum(real* mat, real* output_vec, int M, int N) {
    dim3 threadsPerBlock(NUM_THREADS);
    int num_blocks = (N + threadsPerBlock.x - 1)/threadsPerBlock.x;
    dim3 numBlocks(num_blocks);
    rowSumKernel<<<numBlocks, threadsPerBlock>>>(mat, output_vec, M, N);
}

__global__
void divKernel(real* mat1, real* mat2, int M, int N) {
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < M && j < N) mat2[j*M + i] = mat2[j*M+i] / mat1[j];
}

void GPUdiv(real* mat1, real* mat2, int M, int N)  {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int numBlocks_x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int numBlocks_y = (M + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 numBlocks(numBlocks_x, numBlocks_y);
    divKernel<<<numBlocks, threadsPerBlock>>>(mat1, mat2, M, N);
}
__global__
void sigmoidKernel(real* mat1, real* mat2, int M, int N) {
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; 
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < M && j < N){ 
        mat2[j*M + i] = 1.0/(1.0 + exp(-mat1[j*M + i])); 
    }
}

void GPUsigmoid(real* mat1, real* mat2, int M, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y); 
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; 
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y;
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(mat1, mat2, M, N); 
}

__global__
void addmat(real* mat1, real* mat2, real* output_mat, real alpha, real beta, int M, int N) {

    uint i = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (i < M && j < N){ 
        output_mat[j*M + i] = alpha*mat1[j*M+i] + beta*mat2[j*M+i]; 
    }
}

void GPUaddition(real* mat, real* mat2, real* output_mat, real alpha, real beta, int M, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x; 
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y;
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    addmat<<<numBlocks, threadsPerBlock>>>(mat, mat2, output_mat, alpha, beta, M, N); 
}


__global__
void transpose(real* mat, real* output_mat, int M, int N) {
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; 
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < M && j < N){ 
        output_mat[i*N+j] = mat[j*M + i]; 
    }
}

void GPUtranspose(real* __restrict__ mat, real* __restrict__ output_mat, int M, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y); 
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x;
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y;
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    transpose<<<numBlocks, threadsPerBlock>>>(mat, output_mat, M, N); 
}

__global__
void multKernel(real* mat1, real* mat2, real* output_mat, int M, int N) {
    uint i = (blockIdx.y * blockDim.y) + threadIdx.y; 
    uint j = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if (i < M && j < N){ 
        output_mat[j*M + i] = mat1[j*M + i]*mat2[j*M + i];  
    }
}

void GPUmult(real* mat1, real* mat2, real* output_mat, int M, int N) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y); 
    int num_blocks_x = (N + threadsPerBlock.x - 1)/threadsPerBlock.x;
    int num_blocks_y = (M + threadsPerBlock.y - 1)/threadsPerBlock.y;
    dim3 numBlocks(num_blocks_x, num_blocks_y); 
    multKernel<<<numBlocks, threadsPerBlock>>>(mat1, mat2, output_mat, M, N); 
}

