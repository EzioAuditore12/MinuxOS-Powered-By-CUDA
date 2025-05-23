#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 10000
#define BLOCK_SIZE 256

// Global device counters (for time and completed count)
__device__ int dev_time;
__device__ int dev_completed;


// decrements its remaining time, and advances dev_time, looping until all done.
__global__ void scheduler_kernel(int *arrival, int *remaining, int *priority, int *is_completed) {
    __shared__ unsigned int s_best_packed;  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (true) {
        // reset block’s best
        if (threadIdx.x == 0) s_best_packed = 0xFFFFFFFFu;
        __syncthreads();

        int cur_time = atomicAdd(&dev_time, 0);

        if (tid < N && !is_completed[tid]
            && arrival[tid] <= cur_time
            && remaining[tid] > 0) {
            unsigned int pack = ((unsigned int)priority[tid] << 16) | (unsigned int)tid;
            atomicMin(&s_best_packed, pack);
        }
        __syncthreads();

        // thread-0 of each block merges into global best and updates that proc
        if (threadIdx.x == 0) {
            unsigned int best = s_best_packed;
            if (best != 0xFFFFFFFFu) {
                int idx = best & 0xFFFF;
                // decrement remaining; if we hit zero, mark complete
                int prev = atomicSub(&remaining[idx], 1);
                if (prev == 1) {
                    is_completed[idx] = 1;
                    atomicAdd(&dev_completed, 1);
                }
            }
            // Go to next time unit (like t0 -> t1)
            atomicAdd(&dev_time, 1);
        }
        __syncthreads();

       
        if (atomicAdd(&dev_completed, 0) >= N) break;
    }
}

void initialize_random(int *a, int n, int maxv){
    for(int i=0;i<n;i++) a[i] = (rand() % maxv) + 1;
}

int main(){
    // host arrays
    int *h_arrival = new int[N];
    int *h_burst   = new int[N];
    int *h_prior   = new int[N];

    srand(123);
    initialize_random(h_arrival, N, N);   // arrivals up to N
    initialize_random(h_burst,   N, 100); // burst up to 100
    initialize_random(h_prior,   N, 50);  // priority up to 50
    std::sort(h_arrival, h_arrival+N);

    // device arrays
    int *d_arr, *d_rem, *d_pri, *d_done;
    cudaMalloc(&d_arr, N*sizeof(int));
    cudaMalloc(&d_rem, N*sizeof(int));
    cudaMalloc(&d_pri, N*sizeof(int));
    cudaMalloc(&d_done, N*sizeof(int));

    // copy in
    cudaMemcpy(d_arr, h_arrival, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pri, h_prior,   N*sizeof(int), cudaMemcpyHostToDevice);
    // remaining = burst, is_completed = 0
    cudaMemcpy(d_rem, h_burst,    N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_done, 0,         N*sizeof(int));

    // init dev_time, dev_completed
    int zero=0;
    cudaMemcpyToSymbol(dev_time,      &zero, sizeof(int));
    cudaMemcpyToSymbol(dev_completed, &zero, sizeof(int));

    dim3 block(BLOCK_SIZE), grid((N+BLOCK_SIZE-1)/BLOCK_SIZE);

    std::cout<<"Running GPU scheduler...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    scheduler_kernel<<<grid,block>>>(d_arr, d_rem, d_pri, d_done);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double gput = std::chrono::duration<double>(t1-t0).count();
    std::cout<<"GPU time: "<<gput<<" s\n";

    // (optional) read back final time / completed
    int final_time=0, final_done=0;
    cudaMemcpyFromSymbol(&final_time,      dev_time,      sizeof(int));
    cudaMemcpyFromSymbol(&final_done,      dev_completed, sizeof(int));
    std::cout<<"Simulated time steps: "<<final_time<<", Completed: "<<final_done<<"/"<<N<<"\n";

// CPU scheduling for comparison
    {
        // allocate & init CPU‐side state
        int *cpu_rem   = new int[N];
        int *cpu_done  = new int[N];
        for(int i=0;i<N;i++){
            cpu_rem[i]  = h_burst[i];
            cpu_done[i] = 0;
        }

        int cpu_time = 0, cpu_completed = 0;
        auto c0 = std::chrono::high_resolution_clock::now();
        // time‐stepped priority scheduling on CPU
        while(cpu_completed < N){
            int best_pr = INT_MAX, best_i = -1;
            for(int i=0;i<N;i++){
                if (!cpu_done[i]
                    && h_arrival[i] <= cpu_time
                    && cpu_rem[i] > 0)
                {
                    if (h_prior[i] < best_pr) {
                        best_pr = h_prior[i];
                        best_i = i;
                    }
                }
            }
            if (best_i != -1) {
                if (--cpu_rem[best_i] == 0) {
                    cpu_done[best_i] = 1;
                    cpu_completed++;
                }
            }
            cpu_time++;
        }
        auto c1 = std::chrono::high_resolution_clock::now();
        double cput = std::chrono::duration<double>(c1 - c0).count();
        std::cout<<"CPU time: "<<cput<<" s\n";
        std::cout<<"CPU simulated time steps: "<<cpu_time
                 <<", Completed: "<<cpu_completed<<"/"<<N<<"\n";

        delete[] cpu_rem;
        delete[] cpu_done;
    }

    // cleanup
    cudaFree(d_arr); cudaFree(d_rem);
    cudaFree(d_pri); cudaFree(d_done);
    delete[] h_arrival; delete[] h_burst; delete[] h_prior;
    return 0;
}