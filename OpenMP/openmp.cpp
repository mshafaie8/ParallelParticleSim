#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>
using namespace std;

// Put any static global variables here that you will use throughout the simulation.
vector<vector<int>> bins;
int num_bins;
vector<int> index_in_bin;
static vector<omp_lock_t> bin_locks;
static vector<omp_lock_t> index_in_bin_locks;
double bin_size;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;

    // Apply the force to both particle and its neighbor
	#pragma omp atomic
    particle.ax += coef * dx;
	#pragma omp atomic
    particle.ay += coef * dy;
	#pragma omp atomic
    neighbor.ax += coef * (-dx);
	#pragma omp atomic
    neighbor.ay += coef * (-dy);
	
}

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
	bin_size = cutoff;
	num_bins = ceil(size / bin_size);
	int total_bins = num_bins * num_bins;
	bins.resize(total_bins); //initialize size of bins to total number of bins
    bin_locks.resize(total_bins);
	index_in_bin.resize(num_parts); // initialize to number of particles
    index_in_bin_locks.resize(num_parts);
    
	for (int i = 0; i < total_bins; ++i){
		omp_init_lock(&bin_locks[i]);
	}

    // bin all of the particle's initial positions and update index_in_bin
    #pragma omp for
    for (int i = 0; i < num_parts; ++i){
		omp_init_lock(&index_in_bin_locks[i]);
        int binx = parts[i].x/bin_size;
        int biny = parts[i].y/bin_size;
        int bin = binx + biny * num_bins;
        omp_set_lock(&bin_locks[bin]);
        bins[bin].push_back(i);
        omp_unset_lock(&bin_locks[bin]);
        index_in_bin[i] = bins[bin].size() - 1;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // can add if statement to check num_parts and run parallizing
	#pragma omp for
    for (int i = 0; i < num_parts; ++i){
		parts[i].ax = 0;
		parts[i].ay = 0;
    }

    // Compute Forces
    #pragma omp for
	for (int i = 0; i < num_parts; ++i){
		// particle_t& particle = parts[i]=
		int x = parts[i].x/bin_size;
		int y = parts[i].y/bin_size;
		int bin = x + y * num_bins;
		
		for(int j: bins[x + y*num_bins]){
			if(i>=j) continue; // Ensures force is only applied once per pair
			apply_force(parts[i], parts[j]);
		}

		if (x-1 >= 0 && y + 1 < num_bins) {
            for(int j: bins[x-1 + (y+1)*num_bins]){
				apply_force(parts[i], parts[j]);
			}
        }
	
		// force of bin (x, y) <-> bin(x+1, y)
		if (x+1 < num_bins){
			for(int j: bins[x+1 + y*num_bins]){
				apply_force(parts[i], parts[j]);
			}
		}
		
		// force of bin (x, y) <-> bin(x, y+1)
		if (y+1 < num_bins){
			for(int j: bins[x+(y+1)*num_bins]){
				apply_force(parts[i], parts[j]);
			}
		}

		// force of bin (x, y) <-> bin(x+1, y+1)
		if (x+1 < num_bins && y+1 < num_bins){
			for(int j: bins[(x+1)+(y+1)*num_bins]){
				apply_force(parts[i], parts[j]);
			}
		}
	}

	// Move Particles
	#pragma omp for
	for (int i = 0; i < num_parts; ++i) {
		// calculate old bin
		int old_binx = parts[i].x/bin_size;
		int old_biny = parts[i].y/bin_size;
		int old_bin = old_binx + old_biny * num_bins;

		move(parts[i], size);
		
		// calculate new bin
		int new_binx = parts[i].x/bin_size;
		int new_biny = parts[i].y/bin_size;
		int new_bin = new_binx + new_biny * num_bins;
		
		// if particle moved bins then rebin
		if (new_bin != old_bin){
			// use the index_in_bin to replace particle with the last particle in its bin
			omp_set_lock(&bin_locks[old_bin]); 
			if(bins[old_bin].size() > 1){
				int last_particle = bins[old_bin].back();
				int old_bin_index = index_in_bin[i];
				bins[old_bin][old_bin_index] = last_particle;
				omp_set_lock(&index_in_bin_locks[last_particle]); 
				index_in_bin[last_particle] = old_bin_index; // update the index_in_bin of last particle
				omp_unset_lock(&index_in_bin_locks[last_particle]); 
			}
			bins[old_bin].pop_back(); // shrink bin size by 1
			omp_unset_lock(&bin_locks[old_bin]); 

			

            omp_set_lock(&bin_locks[new_bin]); 

            omp_set_lock(&index_in_bin_locks[i]); 
			index_in_bin[i] = bins[new_bin].size(); // set particle's new index_in_bin 
            omp_unset_lock(&index_in_bin_locks[i]);
			
			bins[new_bin].push_back(i); // add particle to new bin

			omp_unset_lock(&bin_locks[new_bin]); 
		}
	}
}

