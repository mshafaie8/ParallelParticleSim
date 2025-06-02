#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cstdio>
#include <memory>
#include <ostream>
#include <iostream>

using namespace std;

// Global variables for simulation domain decomposition.
int num_bins;
double bin_size;
bool has_ghost_below;
bool has_ghost_above;
int rows_per_proc;
int start_row;
int end_row;
int local_height;
vector<vector<int>> bins;
vector<particle_t> my_parts;
double dxs[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
double dys[9] = {1, -1, 0, 1, -1, 0, 1, -1, 0};

bool compareParticles(const particle_t &particle1, const particle_t &particle2) {
    return particle1.id < particle2.id;
}

void apply_force(particle_t &particle, particle_t &neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff || r2 == 0) return; // Ignore if too far or same particle

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    double coef = (1 - cutoff / r) / r2 / mass;
    double ax = coef * dx;
    double ay = coef * dy;

    particle.ax += ax;
    particle.ay += ay;
}

/**
 * Updates the position and velocity of a particle using Velocity Verlet.
 */
void move(particle_t &p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    // Bounce from walls:
    while (p.x < 0 || p.x > size) {
        p.x = (p.x < 0) ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = (p.y < 0) ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    bin_size = cutoff;
    num_bins = ceil(size / bin_size);
    
    rows_per_proc = ceil((double)num_bins / num_procs);
    
    start_row = min(rows_per_proc * rank, num_bins-1);
    end_row = min(start_row + rows_per_proc, num_bins);
    local_height = end_row - start_row;
    
    has_ghost_below = (rank > 0);
    has_ghost_above = (rank < num_procs - 1);
    
    // Resize bins and neighbors using local_height.
    bins.resize(num_bins * local_height);
    
    // Assign particles to this process.
    for (int i = 0; i < num_parts; i++) {
        int bin_row = parts[i].y/bin_size;
        if (bin_row >= start_row && bin_row < end_row) {
            my_parts.push_back(parts[i]);
        }
    }
}

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    // cout << "simulating one step" << rank << endl;
    // Clear bins for the new step.
    for (auto &b : bins)
        b.clear();

    // Bin particles and prepare ghost regions.
    vector<particle_t> ghost_above_send;
    vector<particle_t> ghost_below_send;
    // cout << "binning " << rank << endl;

    for (int i = 0; i < my_parts.size(); i++) {
        int binx = my_parts[i].x/bin_size;
        int biny = my_parts[i].y/bin_size;
        int localy = biny - start_row;
        int bin = binx + localy * num_bins;
        // Map into bins using local_height.
        bins[bin].push_back(i);
        if (has_ghost_below && biny == start_row){
            ghost_below_send.push_back(my_parts[i]);
        }
        if (has_ghost_above && biny == (end_row - 1)){
            ghost_above_send.push_back(my_parts[i]);
        }
    }


    // cout << "Binning done" << rank << endl;
    
    // Exchange ghost particles.
    // cout << "ghost particle exchange " << rank << endl;
    vector<particle_t> ghost_above_recv;
    vector<particle_t> ghost_below_recv;
    MPI_Status status;
    
    if (has_ghost_above  && (rank < num_procs - 1) ) {
        int send_count = ghost_above_send.size();
        int recv_count = 0;
        // Send ghost_above to rank+1; receive ghost_below from rank+1.
        MPI_Sendrecv(&send_count, 1, MPI_INT, rank+1, 0,
                     &recv_count, 1, MPI_INT, rank+1, 0,
                     MPI_COMM_WORLD, &status);
        ghost_above_recv.resize(recv_count);
        MPI_Sendrecv(ghost_above_send.data(), send_count, PARTICLE, rank+1, 0,
                     ghost_above_recv.data(), recv_count, PARTICLE, rank+1, 0,
                     MPI_COMM_WORLD, &status);
    }
    
    if (has_ghost_below && (rank > 0)) {
        int send_count = ghost_below_send.size();
        int recv_count = 0;
        // Send ghost_below to rank-1; receive ghost_above from rank-1.
        MPI_Sendrecv(&send_count, 1, MPI_INT, rank-1, 0,
                     &recv_count, 1, MPI_INT, rank-1, 0,
                     MPI_COMM_WORLD, &status);
        ghost_below_recv.resize(recv_count);
        MPI_Sendrecv(ghost_below_send.data(), send_count, PARTICLE, rank-1, 0,
                     ghost_below_recv.data(), recv_count, PARTICLE, rank-1, 0,
                     MPI_COMM_WORLD, &status);
    }
    // cout << "ghost particle exchange done " << rank << endl;
    
    // Bin ghost particles by column.
    // cout << "ghost particle binning " << rank << endl;
    vector<vector<int>> ghost_above_binned(num_bins);
    vector<vector<int>> ghost_below_binned(num_bins);
    
    for (int i = 0; i < ghost_above_recv.size(); i++) {
        int col_ind = ghost_above_recv[i].x / bin_size;
        ghost_above_binned[col_ind].push_back(i);
    }
    for (int i = 0; i < ghost_below_recv.size(); i++) {
        int col_ind = ghost_below_recv[i].x / bin_size;
        ghost_below_binned[col_ind].push_back(i);
    }
    // cout << "ghost particle binning done" << rank << endl;
    
    // Compute forces.
    // cout << "applying forces" << rank << endl;
    for (int bx = 0; bx < num_bins; bx++) {
        for (int by = 0; by < local_height; by++) {
            int bin_index = bx + by * num_bins;
            
            for (int i : bins[bin_index]){
                my_parts[i].ax = my_parts[i].ay = 0;
            }
            
            // For each particle in this bin:
            for (int i : bins[bin_index]){
                for (int k = 0; k < 9; k++) {
                    int nx = bx + dxs[k];
                    int ny = by + dys[k];
                    if (nx >= 0 && nx < num_bins && ny >= 0 && ny < local_height){
                        int neighbor_bin = nx + ny * num_bins;
                        for (int neighbor : bins[neighbor_bin]){
                            if (neighbor != i){
                                apply_force(my_parts[i], my_parts[neighbor]);
                            }
                        }
                    }
                }
                // Forces from ghost_above.
                if (has_ghost_above && (by == local_height-1)) {
                    for (int col = max(0, bx - 1); col <= min(num_bins - 1, bx + 1); col++) {
                        for (int j : ghost_above_binned[col]){
                            apply_force(my_parts[i], ghost_above_recv[j]);
                        }
                    }
                }
                // Forces from ghost_below.
                if (has_ghost_below && by == 0) {
                    for (int col = max(0, bx - 1); col <= min(num_bins - 1, bx + 1); col++) {
                        for (int j : ghost_below_binned[col]){
                            apply_force(my_parts[i], ghost_below_recv[j]);
                        }
                    }
                }
            }
        }
    }
    // cout << "applying forces done" << rank << endl;
    
    // Move particles and mark those that cross processor boundaries.
    vector<particle_t> to_move_above;
    vector<particle_t> to_move_below;

    // cout << "moving particles" << rank << endl;
    for (int i = 0; i < my_parts.size(); ) {
        move(my_parts[i], size);
        int new_row = my_parts[i].y / bin_size;
        if (new_row >= end_row) {
            to_move_above.push_back(my_parts[i]);
            my_parts[i] = my_parts.back();
            my_parts.pop_back();
        } else if (new_row < start_row) {
            to_move_below.push_back(my_parts[i]);
            my_parts[i] = my_parts.back();
            my_parts.pop_back();
        } else {
            ++i;
        }
    }

    // cout << "moving particles done" << rank << endl;
    
    // Exchange migrated particles.
    vector<particle_t> parts_from_above;
    vector<particle_t> parts_from_below;

    // cout << "moving particles communication" << rank << endl;
    
    if (has_ghost_above && (rank < num_procs - 1)) {
        int send_count = to_move_above.size();
        int recv_count = 0;
        MPI_Sendrecv(&send_count, 1, MPI_INT, rank+1, 0,
                     &recv_count, 1, MPI_INT, rank+1, 0,
                     MPI_COMM_WORLD, &status);
        parts_from_above.resize(recv_count);
        MPI_Sendrecv(to_move_above.data(), send_count, PARTICLE, rank+1, 0,
                     parts_from_above.data(), recv_count, PARTICLE, rank+1, 0,
                     MPI_COMM_WORLD, &status);
    }
    
    if (has_ghost_below && (rank > 0)) {
        int send_count = to_move_below.size();
        int recv_count = 0;
        MPI_Sendrecv(&send_count, 1, MPI_INT, rank-1, 0,
                     &recv_count, 1, MPI_INT, rank-1, 0,
                     MPI_COMM_WORLD, &status);
        parts_from_below.resize(recv_count);
        MPI_Sendrecv(to_move_below.data(), send_count, PARTICLE, rank-1, 0,
                     parts_from_below.data(), recv_count, PARTICLE, rank-1, 0,
                     MPI_COMM_WORLD, &status);
    }
    my_parts.insert(my_parts.end(), parts_from_above.begin(), parts_from_above.end());
    my_parts.insert(my_parts.end(), parts_from_below.begin(), parts_from_below.end());
    // cout << "moving particles done" << rank << endl;
}

void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_procs) {
    // cout << "gather for save" << rank << endl;
    int local_count = my_parts.size();
    vector<int> counts(num_procs, 0);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(num_procs, 0);
    int total_count = 0;
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            displs[i] = total_count;
            total_count += counts[i];
        }
    }
    
    vector<particle_t> all_particles(total_count);
    MPI_Gatherv(my_parts.data(), local_count, PARTICLE,
                all_particles.data(), counts.data(), displs.data(), PARTICLE,
                0, MPI_COMM_WORLD);
                
    if (rank == 0) {
        sort(all_particles.begin(), all_particles.end(), compareParticles);
        for (int i = 0; i < total_count; i++) {
            parts[i] = all_particles[i];
        }
    }

    // cout << "gather for save done" << rank << endl;
}