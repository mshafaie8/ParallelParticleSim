#include "common.h"
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#define NUM_THREADS 1024
#define BLOCKS_PER_SM 32

// Put any static global variables here that you will use throughout the simulation.
int blks;
int devId = 0;
int numSMs;
int global_threads;
double grid_dim;
int bin_grid_dim;
int total_bins;
double bin_size;
using namespace std;
int* bin_starts_gpu;
int* binned_particles_gpu;
int* bins;


__global__ void init_parts(int* part_arr, int num_parts, int global_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_parts) {
        part_arr[tid] = tid;
        tid += global_threads;
    }
}

__global__ void zero_accels(particle_t* particles, int num_parts, int global_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_parts) {
        particles[tid].ax = particles[tid].ay = 0;
        tid += global_threads;
    }
}

__global__ void compute_bin_counts(particle_t* particles, int num_parts, int* bin_counts, int* bins, double bin_size, int bin_grid_dim) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    int binx = particles[tid].x / bin_size;
    int biny = particles[tid].y / bin_size;
    int bin = binx + biny*bin_grid_dim;
    bins[tid] = bin;
    atomicAdd(&bin_counts[bin], 1);
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* binned_particles, int* bin_starts,
                                        double bin_size, int bin_grid_dim) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    double ax = 0;
    double ay = 0;
    
    // Determine bin coordinates once
    double my_x = particles[tid].x;
    double my_y = particles[tid].y;
    int binx = my_x / bin_size;
    int biny = my_y / bin_size;
    
    // Process current bin
    int bin = binx + biny * bin_grid_dim;
    for (int j = bin_starts[bin]; j < bin_starts[bin+1]; j++) {
        int neighbor_ind = binned_particles[j];
        if (tid >= neighbor_ind) continue; // Only apply force once per pair
        double dx = particles[neighbor_ind].x - my_x;
        double dy = particles[neighbor_ind].y - my_y;
        double r2 = dx * dx + dy * dy;
        if (r2 > cutoff * cutoff)
            continue;
        // r2 = fmax( r2, min_r*min_r );
        r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
        double r = sqrt(r2);

        //
        //  very simple short-range repulsive force
        //
        double coef = (1 - cutoff / r) / r2 / mass;

        ax += coef * dx;
        ay += coef * dy;
        atomicAdd(&particles[neighbor_ind].ax, -coef * dx);
        atomicAdd(&particles[neighbor_ind].ay, -coef * dy);
        
        
        //apply_force_gpu(particles[tid], particles[neighbor_ind]);
    }

    // Process all neighboring bins in one loop
    // Loop only through the necessary adjacent bins
    for (int ny = biny; ny <= biny+1; ny++) {
        if (ny >= bin_grid_dim) continue;
        
        for (int nx = binx-1; nx <= binx+1; nx++) {
            if (nx < 0 || nx >= bin_grid_dim || ((nx == binx || nx == binx-1) && ny == biny)) continue;

            int neighbor_bin = nx + ny * bin_grid_dim;
            for (int j = bin_starts[neighbor_bin]; j < bin_starts[neighbor_bin+1]; j++) {
                int neighbor_ind = binned_particles[j];
                double dx = particles[neighbor_ind].x - my_x;
                double dy = particles[neighbor_ind].y - my_y;
                double r2 = dx * dx + dy * dy;
                if (r2 > cutoff * cutoff)
                    continue;
                // r2 = fmax( r2, min_r*min_r );
                r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
                double r = sqrt(r2);

                //
                //  very simple short-range repulsive force
                //
                double coef = (1 - cutoff / r) / r2 / mass;

                ax += coef * dx;
                ay += coef * dy;
                atomicAdd(&particles[neighbor_ind].ax, -coef * dx);
                atomicAdd(&particles[neighbor_ind].ay, -coef * dy);
                //apply_force_gpu(particles[tid], particles[neighbor_ind]);
            }
        }
    }

    atomicAdd(&particles[tid].ax, ax);
    atomicAdd(&particles[tid].ay, ay);

}


__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    bin_size = cutoff;
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    global_threads = NUM_THREADS * BLOCKS_PER_SM * numSMs;

    grid_dim = size;
    bin_size = cutoff;

    bin_grid_dim = ceil(grid_dim/bin_size);
    total_bins = bin_grid_dim * bin_grid_dim;
    cudaMalloc((void**)&bin_starts_gpu, (total_bins + 1) * sizeof(int));
    cudaMalloc((void**)&bins, (num_parts) * sizeof(int));
    cudaMalloc((void**)&binned_particles_gpu, num_parts * sizeof(int));

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // cout << "Total bins: " << total_bins << endl;
    
    //cout << "Simulating Step" << endl;
    
    cudaMemset(bin_starts_gpu, 0, (total_bins + 1) * sizeof(int));


    init_parts<<<numSMs*BLOCKS_PER_SM, NUM_THREADS>>>(binned_particles_gpu, num_parts, global_threads);    
    compute_bin_counts<<<blks, NUM_THREADS>>>(parts, num_parts, bin_starts_gpu, bins, bin_size, bin_grid_dim);

    thrust::exclusive_scan(thrust::device, bin_starts_gpu, bin_starts_gpu + total_bins + 1, bin_starts_gpu, 0, thrust::plus<int>()); // in-place scan
    thrust::sort_by_key(thrust::device, bins, bins + num_parts, binned_particles_gpu);


    zero_accels<<<numSMs*BLOCKS_PER_SM, NUM_THREADS>>>(parts, num_parts, global_threads);
    
    // cudaDeviceSynchronize();
    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, binned_particles_gpu, bin_starts_gpu,
                                             bin_size, bin_grid_dim);

    
    // cudaDeviceSynchronize();
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
