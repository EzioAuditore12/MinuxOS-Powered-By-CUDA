#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 100
#define BLOCK_SIZE 32

void initialize_random_values(int *arr, int n){
    for(int i=0;i<n;i++){
        arr[i] = ((int)rand() % 1000) + 1;
    }
}

__global__ void get_highest_priority(int *priority, int *arrival_time, int * remaining_time, int *is_completed, int *selected_process, int time){

    __shared__ int best_priority;
    __shared__ int best_index;
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x == 0){
        // since I am considering 
        // 0 -> hightest priority
        // INT_MAX -> Lowest priority

        best_priority = INT_MAX;
        best_index = -1;
    }

    __syncthreads();

    if(thread_i<N && !is_completed[thread_i] && arrival_time[thread_i]<=time && remaining_time[thread_i]>0){
        atomicMin(&best_priority, priority[thread_i]);
    }

    __syncthreads();

    if(thread_i<N && !is_completed[thread_i] && arrival_time[thread_i]<=time && remaining_time[thread_i]>0 && priority[thread_i]==best_priority){
        atomicExch(&best_index, thread_i);
    }

    __syncthreads();

    // At last I selected this thread to set best process
    if(threadIdx.x == 0){
        *selected_process = best_index;
    }

}

__global__ void update_process(int *remaining_time, int *is_completed, int selected_process) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        remaining_time[selected_process]--;
        
        if (remaining_time[selected_process] == 0) {
            is_completed[selected_process] = 1;
        }
    }
}

void cpu_p_scheduler(int *arrival_time, int *burst_time, int *priority) {
    int remaining_time[N];
    int is_completed[N];
    int time = 0;
    int completed = 0;

    for (int i = 0; i < N; i++) {
        remaining_time[i] = burst_time[i];
        is_completed[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    while (completed < N) {
        int best_priority = INT_MAX;
        int best_index = -1;

        for (int i = 0; i < N; i++) {
            if (!is_completed[i] && arrival_time[i] <= time && remaining_time[i] > 0) {
                if (priority[i] < best_priority) {
                    best_priority = priority[i];
                    best_index = i;
                }
            }
        }

        if (best_index != -1) {
            remaining_time[best_index]--;
            if (remaining_time[best_index] == 0) {
                is_completed[best_index] = 1;
                completed++;
            }
        }
        time++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "CPU priority scheduler time: " << duration.count() << " seconds" << std::endl;
}


int main(){
    int *h_arrival_time, *h_burst_time, *h_priority, *h_remaining_time, *h_is_completed;
    
    int *d_arrival_time, *d_burst_time, *d_priority, *d_remaining_time, *d_is_completed;

    
    h_arrival_time = new int[N];
    h_burst_time = new int[N];
    h_priority = new int[N];
    h_remaining_time = new int[N];
    h_is_completed = new int[N];

    // Randomly initializing values in array for now
    srand(time(NULL));
    initialize_random_values(h_arrival_time, N);
    initialize_random_values(h_burst_time, N);
    initialize_random_values(h_priority, N);
    
    std::sort(h_arrival_time, h_arrival_time + N);
    
    for (int i = 0; i < N; i++) {
        h_remaining_time[i] = h_burst_time[i];
        h_is_completed[i] = 0;
    }

    
    int size = N * sizeof(int);
    cudaMalloc(&d_arrival_time, size);
    cudaMalloc(&d_burst_time, size);
    cudaMalloc(&d_priority, size);
    cudaMalloc(&d_remaining_time, size);
    cudaMalloc(&d_is_completed, size);


    // Now finally copying all these values
    cudaMemcpy(d_arrival_time, h_arrival_time, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_burst_time, h_burst_time, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, h_priority, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_remaining_time, h_remaining_time, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_completed, h_is_completed, size, cudaMemcpyHostToDevice);

    // For now I am dividing it for 32, can reduce it in future
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    int *d_selected_process; // this is not array
    cudaMalloc(&d_selected_process, sizeof(int));

    int time = 0;
    int h_selected_process;
    int completed = 0;

    // Warmup runs
    std::cout<<"Performing warmup runs\n";
    for(int i=0;i<4;i++){
        get_highest_priority<<<gridDim, blockDim>>>(d_priority, d_arrival_time, d_remaining_time, d_is_completed, d_selected_process, time);
    }

    std::cout<<"Starting execution: \n";
    auto start_exec_time = std::chrono::high_resolution_clock::now();
    int completed_temp = completed;

    while(completed < N){
        // Reset selected process to -1 before each kernel launch
        h_selected_process = -1;
        cudaMemcpy(d_selected_process, &h_selected_process, sizeof(int), cudaMemcpyHostToDevice);

        get_highest_priority<<<gridDim, blockDim>>>(d_priority, d_arrival_time, d_remaining_time, d_is_completed, d_selected_process, time);

        cudaMemcpy(&h_selected_process, d_selected_process, sizeof(int), cudaMemcpyDeviceToHost);

        if(h_selected_process != -1){
            // Use a small kernel to update just the selected process
            update_process<<<1, 1>>>(d_remaining_time, d_is_completed, h_selected_process);
            
            // Check if this process completed
            int is_completed_val = 0;
            cudaMemcpy(&is_completed_val, &d_is_completed[h_selected_process], sizeof(int), cudaMemcpyDeviceToHost);
            
            if(is_completed_val) {
                completed++;
            }
        }
        time++;

        if(completed_temp != completed){
            std::cout<<"Completed Process: "<<completed_temp<<std::endl;
            completed_temp = completed;
        }
        
    }
    cudaDeviceSynchronize();

    auto end_exec_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_exec_time - start_exec_time;

    std::cout<<"GPU took: "<<duration.count()<<" seconds\n";

    cpu_p_scheduler(h_arrival_time, h_burst_time, h_priority);

    // Now just cleaning stuff
    cudaFree(d_arrival_time);
    cudaFree(d_burst_time);
    cudaFree(d_priority);
    cudaFree(d_remaining_time);
    cudaFree(d_is_completed);
    cudaFree(d_selected_process);

    delete[] h_arrival_time;
    delete[] h_burst_time;
    delete[] h_priority;
    delete[] h_remaining_time;
    delete[] h_is_completed;

    return 0;
}
