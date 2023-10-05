// Some code in this file, particularly frustum, ray and intersection tests, 
// is adapted from three.js. Three.js is licensed under the MIT license
// This file this follows the three.js licensing
// License: MIT https://github.com/mrdoob/three.js/blob/dev/LICENSE

#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"
#include "structures.cuh"
#include "../CudaPrint/CudaPrint.cuh"

namespace cg = cooperative_groups;

extern "C" __global__
void kernel(
	const Uniforms uniforms,
	uint8_t* buffer_octree,
	Node* nodes,
	Stats* stats,
	CudaPrint* cudaprint,
	uint32_t* _numBatchesUploaded_volatile,
	uint32_t* batchSizes
){
	PRINT("resetting octree \n");

	auto grid = cg::this_grid();

	AllocatorGlobal* allocator_octree = (AllocatorGlobal*)buffer_octree;

	grid.sync();

	Node* root = &nodes[0];
		
	if(grid.thread_rank() == 0){
		
		allocator_octree->buffer = buffer_octree;
		allocator_octree->offset = 16; // 16-aligned, first 8 byte is allocator itself

		*stats = Stats();

		stats->numNodes = 1;
		stats->batchletIndex = 0;
		stats->numPointsProcessed = 0;
		stats->numAllocatedChunks = 0;
		stats->chunkPoolSize = 0;
		stats->frameID = uniforms.frameCounter;
		stats->memCapacityReached = false;

		memset(root->children, 0, 8 * sizeof(Node*));
		root->isFiltered = false;
		root->counter = 0;
		root->numPoints = 0;
		root->level = 0;
		root->X = 0;
		root->Y = 0;
		root->Z = 0;
		root->countIteration = 0;
		memset(root->name, 0, 20);
		root->name[0] = 'r';
		root->numVoxels = 0;
		root->numVoxelsStored = 0;
		root->voxelChunks = nullptr;
		root->grid = (OccupancyGrid*)allocator_octree->alloc(sizeof(OccupancyGrid));

		*_numBatchesUploaded_volatile = 0;
		
		for(int i = 0; i < BATCH_STREAM_SIZE; i++){
			batchSizes[i] = 0;
		}
	}

	grid.sync();

	// clear occupancy grid
	processRange((GRID_NUM_CELLS / 32u), [&](int index){
		root->grid->values[index] = 0;
	});
	
	grid.sync();
}

