#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <iostream> // Prefer iostream
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define N 10000
#define BLOCK_SIZE 32 // Threads per block

// Forward declaration
void cpu_p_scheduler(int *arrival_time, int *burst_time, int *priority, int n_processes);

void initialize_random_values(int *arr, int n, int max_val = 1000){
    for(int i=0;i<n;i++){
        arr[i] = (rand() % max_val) + 1;
    }
}

// Kernel to select the highest priority process using a two-stage reduction
// Stage 1: Each block finds its best candidate using shared memory.
// Stage 2: One thread from each block atomically updates a global variable.
__global__ void select_best_process_kernel_v2(int *priority_arr, int *arrival_time_arr,
                                           int *remaining_time_arr, int *is_completed_arr,
                                           unsigned int *d_global_selected_packed_value, int time, int n_processes){
    __shared__ unsigned int s_block_best_packed_value;
    
    int thread_idx_in_block = threadIdx.x;
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory for the block
    if (thread_idx_in_block == 0) {
        s_block_best_packed_value = 0xFFFFFFFFU; // Max unsigned int, represents "no process selected"
    }
    __syncthreads(); // Ensure s_block_best_packed_value is initialized

    // Each thread checks one process (if within bounds)
    if(global_thread_idx < n_processes && 
       !is_completed_arr[global_thread_idx] && 
       arrival_time_arr[global_thread_idx] <= time && 
       remaining_time_arr[global_thread_idx] > 0) {
        // Lower priority value is better. Lower index is tie-breaker.
        // Pack priority (higher bits) and index (lower bits)
        unsigned int my_packed_val = ((unsigned int)priority_arr[global_thread_idx] << 16) | (unsigned int)global_thread_idx;
        atomicMin(&s_block_best_packed_value, my_packed_val); // Reduce within the block
    }
    __syncthreads(); // Ensure all threads in block have updated s_block_best_packed_value

    // One thread from the block updates the global minimum
    if (thread_idx_in_block == 0 && s_block_best_packed_value != 0xFFFFFFFFU) {
        atomicMin(d_global_selected_packed_value, s_block_best_packed_value);
    }
}

// Kernel to update the selected process's state and count completions
__global__ void update_selected_process_kernel(int *remaining_time_arr, int *is_completed_arr,
                                             unsigned int* d_selected_packed_value_ptr,
                                             int *d_completed_atomic_counter, int n_processes) {
    // This kernel is launched with <<<1,1>>>
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int packed_value = *d_selected_packed_value_ptr;

        if (packed_value != 0xFFFFFFFFU) { // Check if a process was selected
            int selected_idx = packed_value & 0xFFFFU; // Unpack index

            if (selected_idx >= 0 && selected_idx < n_processes) {
                if (!is_completed_arr[selected_idx] && remaining_time_arr[selected_idx] > 0) {
                    remaining_time_arr[selected_idx]--;
                    
                    if (remaining_time_arr[selected_idx] == 0) {
                        is_completed_arr[selected_idx] = 1;
                        atomicAdd(d_completed_atomic_counter, 1);
                    }
                }
            }
        }
    }
}

void cpu_p_scheduler(int *arrival_time, int *burst_time, int *priority, int n_processes) {
    std::vector<int> remaining_time(n_processes);
    std::vector<int> is_completed(n_processes, 0);
    long long current_time = 0; // Use long long for time to avoid overflow if burst times are large
    int completed_count = 0;
    long long total_cpu_ops = 0; // For rough estimation

    for (int i = 0; i < n_processes; i++) {
        remaining_time[i] = burst_time[i];
    }

    std::cout << "\nStarting CPU Priority Scheduler (N=" << n_processes << ")..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    while (completed_count < n_processes) {
        int best_priority_val = INT_MAX;
        int best_index = -1;

        for (int i = 0; i < n_processes; i++) {
            total_cpu_ops++;
            if (!is_completed[i] && arrival_time[i] <= current_time && remaining_time[i] > 0) {
                if (priority[i] < best_priority_val) {
                    best_priority_val = priority[i];
                    best_index = i;
                } else if (priority[i] == best_priority_val) { // Tie-breaking: lower index wins
                    if (best_index == -1 || i < best_index) { // Ensure consistent tie-breaking
                         best_index = i;
                    }
                }
            }
        }

        if (best_index != -1) {
            remaining_time[best_index]--;
            total_cpu_ops++;
            if (remaining_time[best_index] == 0) {
                is_completed[best_index] = 1;
                completed_count++;
            }
        }
        current_time++;
        if (current_time > n_processes * 2000 && best_index == -1) { // Safety break for CPU if no process gets selected for too long
            std::cout << "CPU Safety break: No process selected for extended time. Completed: " << completed_count << std::endl;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "CPU priority scheduler time: " << duration.count() << " seconds" << std::endl;
    std::cout << "CPU total simulated time steps: " << current_time << std::endl;
    std::cout << "CPU approx operations: " << total_cpu_ops << std::endl;
}


int main(){
    int *h_arrival_time, *h_burst_time, *h_priority;
    // Host arrays for initial data and CPU scheduler
    h_arrival_time = new int[N];
    h_burst_time = new int[N];
    h_priority = new int[N];

    // Device arrays
    int *d_arrival_time, *d_burst_time, *d_priority, *d_remaining_time, *d_is_completed;
    unsigned int *d_selected_packed_value; // Stores (priority << 16 | index)
    int *d_completed_atomic;          // Atomic counter for completed processes on GPU

    srand(123); // Use a fixed seed for reproducible tests
    initialize_random_values(h_arrival_time, N, N); // Arrival times up to N for more spread
    initialize_random_values(h_burst_time, N, 100);  // Max burst time 100 to keep simulation shorter
    initialize_random_values(h_priority, N, 50);    // Max priority value 50

    std::sort(h_arrival_time, h_arrival_time + N);
    
    // Temporary host arrays for initial setup of remaining_time and is_completed
    int* h_temp_remaining_time = new int[N];
    int* h_temp_is_completed = new int[N];
    for (int i = 0; i < N; i++) {
        h_temp_remaining_time[i] = h_burst_time[i];
        h_temp_is_completed[i] = 0;
    }
    
    int data_size_int = N * sizeof(int);
    cudaMalloc(&d_arrival_time, data_size_int);
    cudaMalloc(&d_burst_time, data_size_int);
    cudaMalloc(&d_priority, data_size_int);
    cudaMalloc(&d_remaining_time, data_size_int);
    cudaMalloc(&d_is_completed, data_size_int);
    cudaMalloc(&d_selected_packed_value, sizeof(unsigned int));
    cudaMalloc(&d_completed_atomic, sizeof(int));

    cudaMemcpy(d_arrival_time, h_arrival_time, data_size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_burst_time, h_burst_time, data_size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, h_priority, data_size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_remaining_time, h_temp_remaining_time, data_size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_completed, h_temp_is_completed, data_size_int, cudaMemcpyHostToDevice);

    int h_initial_completed_count = 0;
    cudaMemcpy(d_completed_atomic, &h_initial_completed_count, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE); // Ensure enough blocks

    // Warmup runs (optional, but good practice)
    unsigned int h_dummy_packed_init = 0xFFFFFFFFU; 
    cudaStream_t stream = 0; // Default stream
    std::cout<<"Performing GPU warmup runs..." << std::endl;
    for(int i=0;i<5;i++){
        cudaMemcpyAsync(d_selected_packed_value, &h_dummy_packed_init, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
        select_best_process_kernel_v2<<<gridDim, blockDim, 0, stream>>>(d_priority, d_arrival_time, d_remaining_time, d_is_completed, d_selected_packed_value, 0, N);
        update_selected_process_kernel<<<1, 1, 0, stream>>>(d_remaining_time, d_is_completed, d_selected_packed_value, d_completed_atomic, N);
    }
    cudaDeviceSynchronize(); // Ensure warmups are done

    // Reset state for actual run if warmup modified critical data
    cudaMemcpy(d_remaining_time, h_temp_remaining_time, data_size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_completed, h_temp_is_completed, data_size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_completed_atomic, &h_initial_completed_count, sizeof(int), cudaMemcpyHostToDevice);


    std::cout<<"Starting GPU execution (N=" << N << ")..." << std::endl;
    auto start_exec_time = std::chrono::high_resolution_clock::now();
    
    long long gpu_time_steps = 0;
    int h_completed_gpu = 0;
    int prev_printed_completed_gpu = -1;
    // Check GPU completion status less frequently to reduce D2H copy overhead
    int check_completion_interval = N > 1000 ? N / 10 : 100; // Heuristic for interval
    if (check_completion_interval == 0) check_completion_interval = 1;


    while(h_completed_gpu < N){
        // Reset d_selected_packed_value for the current time step's selection
        cudaMemcpyAsync(d_selected_packed_value, &h_dummy_packed_init, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);

        select_best_process_kernel_v2<<<gridDim, blockDim, 0, stream>>>(d_priority, d_arrival_time, d_remaining_time, d_is_completed, d_selected_packed_value, gpu_time_steps, N);
        
        update_selected_process_kernel<<<1, 1, 0, stream>>>(d_remaining_time, d_is_completed, d_selected_packed_value, d_completed_atomic, N);
        
        gpu_time_steps++;

        // Periodically check completion status from GPU
        if (gpu_time_steps % check_completion_interval == 0 || h_completed_gpu == N -1 ) { // Check more often near end
            cudaMemcpy(&h_completed_gpu, d_completed_atomic, sizeof(int), cudaMemcpyDeviceToHost); // This will sync stream 0
            if (prev_printed_completed_gpu != h_completed_gpu) {
                 // std::cout << "GPU Time: " << gpu_time_steps << ", Processes Completed: " << h_completed_gpu << std::endl;
                 prev_printed_completed_gpu = h_completed_gpu;
            }
        }
        if (gpu_time_steps > N * 2000 && h_completed_gpu < N) { // Safety break for GPU
             std::cout << "GPU Safety break: Simulation running too long. Completed: " << h_completed_gpu << std::endl;
             cudaMemcpy(&h_completed_gpu, d_completed_atomic, sizeof(int), cudaMemcpyDeviceToHost); // Get final count
             break;
        }
    }
    // Final synchronization to ensure all GPU work is done before stopping timer and getting final count
    cudaDeviceSynchronize();
    cudaMemcpy(&h_completed_gpu, d_completed_atomic, sizeof(int), cudaMemcpyDeviceToHost); // Get final count

    auto end_exec_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_exec_time - start_exec_time;
    std::cout << "GPU took: " << duration.count() << " seconds" << std::endl;
    std::cout << "GPU total simulated time steps: " << gpu_time_steps << std::endl;
    std::cout << "GPU total processes completed: " << h_completed_gpu << std::endl;


    // Run CPU scheduler for comparison
    cpu_p_scheduler(h_arrival_time, h_burst_time, h_priority, N);

    // Cleanup
    cudaFree(d_arrival_time);
    cudaFree(d_burst_time);
    cudaFree(d_priority);
    cudaFree(d_remaining_time);
    cudaFree(d_is_completed);
    cudaFree(d_selected_packed_value);
    cudaFree(d_completed_atomic);

    delete[] h_arrival_time;
    delete[] h_burst_time;
    delete[] h_priority;
    delete[] h_temp_remaining_time;
    delete[] h_temp_is_completed;

    return 0;
}
