#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
using namespace std;
using namespace std::chrono;

#define NUM_X_BLOCKS 20'000 // 200'000 // used to put GPU under a tougher parallel load
#define THREAD_1D_LEN 32 // max=32 // one side of a square array
#define TOTAL_SIZE (THREAD_1D_LEN * THREAD_1D_LEN) // 32^2=1024
#define ITER 16'000 // sequential iterations to keep GPU working long enough to get % load


// static unified memory (global) -- (where cudaMallocManaged is dynamic)
// global flat A, B, C used for A @ B + C
__device__ __managed__ float A[TOTAL_SIZE];
__device__ __managed__ float B[TOTAL_SIZE];
__device__ __managed__ float C[TOTAL_SIZE];
/* output sized * num x blocks
 * due to GPU overprocessing these will all be repeated data,
 * but the point is to stress the GPU
 * G takes the majority of setup time when NUM_X_BLOCKS is too high
 */
__device__ __managed__ float G[TOTAL_SIZE * NUM_X_BLOCKS];


// kernels
__device__ int getLocalIdx(int x, int y) {
    return (y * THREAD_1D_LEN) + x;
}
__device__ int getGlobalIdx(int localIdx, int blockIdx) {
    return (TOTAL_SIZE * blockIdx) + localIdx;
}
__global__ void prefillKernel() { // globally fill kernels w/ random floats (untimed)
    int tid = getLocalIdx(threadIdx.x, threadIdx.y);

    curandStatePhilox4_32_10_t state; // philox pseudorandom is fast
    curand_init(50227, tid, 0, &state);
    A[tid] = curand_uniform(&state); // assign float Domain=(0.,1.]
    B[tid] = curand_uniform(&state);
    C[tid] = curand_uniform(&state);
}
__global__ void feedForwardOne() {
    /* note that the number of blocks will just overprocess the GPU in a parallel fashion
     * it will have no effect on the inner-workings
     */
    int tid = getLocalIdx(threadIdx.x, threadIdx.y);
    int gid = getGlobalIdx(tid, blockIdx.x);

    // mat mul
    float res = 0.0;
    for (int i = 0; i < THREAD_1D_LEN; i++) {
        int idxA = getLocalIdx(i, threadIdx.y);
        int idxB = getLocalIdx(threadIdx.x, i);

        res += A[idxA] * B[idxB];
    }

    // set G to matmul + bias
    G[gid] = res + C[tid];
}


void testDisplay() {
    // check all output
    // cout << "~~ A" << endl;
    // for (int i = 0; i < TOTAL_SIZE; i++) {
    //     cout << i << " -- " << A[i] << endl;
    // }
    // cout << "~~ B" << endl;
    // for (int i = 0; i < TOTAL_SIZE; i++) {
    //     cout << i << " -- " << B[i] << endl;
    // }
    // cout << "~~ C" << endl;
    // for (int i = 0; i < TOTAL_SIZE; i++) {
    //     cout << i << " -- " << C[i] << endl;
    // }
    // cout << "~~ G" << endl;
    // for (int i = 0; i < TOTAL_SIZE; i++) {
    //     cout << i << " -- " << G[i] << endl;
    // }

    // check specific index worth
    // cout << A[0] << " * " << B[1] << endl;
    // cout << A[1] << " * " << B[4] << endl;
    // cout << A[2] << " * " << B[7] << endl;
    // cout << "C1 " << C[1] << endl;
    // cout << "G1  " << G[1] << endl;
    // cout << "G10 " << G[10] << endl;
    // cout << "G19 " << G[19] << endl;

    // check thread and block IDs
    cout << "~~" << endl;
    int count = 0;
    for (int i = 0; i < TOTAL_SIZE * NUM_X_BLOCKS; i++) {
        cout << G[i] << "\t"; // G[i]

        count++;
        if (count % THREAD_1D_LEN == 0) {
            cout << endl;
        }
        if (count == TOTAL_SIZE) {
            count = 0;
            cout << endl;
        }
    }
}
int main() {
    dim3 numBlocksPrefill(1, 1); // prefill fills one set of global A,B,C
    dim3 numBlocksProcess(NUM_X_BLOCKS, 1); // process processes global A,B,C several times across multi blocks
    dim3 numThreads(THREAD_1D_LEN, THREAD_1D_LEN);
    cudaDeviceSynchronize(); // needed for timing and debug

    // prefill
    cout << "prefilling... " << endl;
    auto t1 = high_resolution_clock::now(); // start timer
    prefillKernel <<< numBlocksPrefill, numThreads >>> ();
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now(); // end timer
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    cout << "took " << ms_int.count() << "ms\n" << endl;


    //  process
    cout << "processing... " << endl;
    t1 = high_resolution_clock::now(); // start timer
    for (int i = 0; i < ITER; i++) {
        feedForwardOne <<< numBlocksProcess, numThreads >>> ();
    }
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now(); // end timer
    ms_int = duration_cast<milliseconds>(t2 - t1);
    cout << "took " << ms_int.count() << "ms" << endl;


    // DEBUG
    // testDisplay();
    cudaGetLastError(); // check err


    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(G);
    return 0;
}