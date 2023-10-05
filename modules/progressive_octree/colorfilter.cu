
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.h.cu"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"
// #include "rasterization.cuh"
#include "structures.cuh"

Allocator* allocator = nullptr;
Uniforms uniforms;

constexpr uint64_t sampleGridSize = 64;
constexpr uint64_t sampleGrid_numCells      = sampleGridSize * sampleGridSize * sampleGridSize;
constexpr uint64_t sampleGrid_numBytes      = sampleGrid_numCells * sizeof(uint64_t);
constexpr int acceptedCapacity              = 300'000;

void findBottomMostNodes(Node* nodes, uint32_t numNodes, Node** workload, uint32_t& numBottomUnfilteredNodes){

	auto grid = cg::this_grid();

	if(grid.thread_rank() == 0){ 
		numBottomUnfilteredNodes = 0;
	}
	grid.sync();

	processRange(numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];
		
		if(node->isLeafFn()) return;

		bool allChildrenFiltered = true;
		bool isUnfiltered = !node->isFiltered;

		for(int i = 0; i < 8; i++){
			if(node->children[i] == nullptr) continue;

			bool childFiltered = node->children[i]->isFiltered;

			allChildrenFiltered = allChildrenFiltered && childFiltered;
		}

		if(isUnfiltered && allChildrenFiltered){
			uint32_t bottomNodeIndex = atomicAdd(&numBottomUnfilteredNodes, 1);
			workload[bottomNodeIndex] = node;
		}
	});

	grid.sync();
}

void doSampling(
	Node* node, 
	int childIndex, 
	Chunk* chunk, 
	int numSamples, 
	uint64_t* sampleGrid,
	uint32_t& sh_numAccepted_total,
	uint32_t& sh_numAccepted_child,
	uint32_t* accepted
){

	auto block = cg::this_thread_block();

	if(numSamples == 0) return;
	if(chunk == nullptr) return;

	Node* child = node->children[childIndex];

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 octreeMin = uniforms.boxMin;
	float3 octreeMax = octreeMin + cubeSize;
	float3 octreeSize = octreeMax - octreeMin;

	float3 cubeMin = uniforms.boxMin;
	float nodeSize = cubeSize / pow(2.0f, float(node->level));
	float nodeMin_x = (float(node->X) + 0.0f) * nodeSize + cubeMin.x;
	float nodeMin_y = (float(node->Y) + 0.0f) * nodeSize + cubeMin.y;
	float nodeMin_z = (float(node->Z) + 0.0f) * nodeSize + cubeMin.z;

	float childSize = cubeSize / pow(2.0f, float(child->level));
	float childMin_x = (float(child->X) + 0.0f) * childSize + cubeMin.x;
	float childMin_y = (float(child->Y) + 0.0f) * childSize + cubeMin.y;
	float childMin_z = (float(child->Z) + 0.0f) * childSize + cubeMin.z;

	// __shared__ uint32_t sh_numAccepted_child;
	int chunkIndex = 0;

	// sh_numAccepted_child = 0;

	block.sync();

	for(
		uint32_t pointIndex = block.thread_rank(); 
		pointIndex < numSamples;
		pointIndex += block.num_threads()
	){

		int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

		if(chunkIndex < targetChunkIndex){
			chunk = chunk->next;
			chunkIndex++;
		}

		int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;

		Point point = chunk->points[pointInChunkIndex];

		// integer point coordinate relative to root node
		uint32_t pX_full = 16'777'216.0f * (point.x - octreeMin.x) / octreeSize.x;
		uint32_t pY_full = 16'777'216.0f * (point.y - octreeMin.y) / octreeSize.y;
		uint32_t pZ_full = 16'777'216.0f * (point.z - octreeMin.z) / octreeSize.z;

		uint32_t pX_leveled = pX_full / (1 << (17 - node->level));
		uint32_t pY_leveled = pY_full / (1 << (17 - node->level));
		uint32_t pZ_leveled = pZ_full / (1 << (17 - node->level));

		float pX = pX_leveled % sampleGridSize;
		float pY = pY_leveled % sampleGridSize;
		float pZ = pZ_leveled % sampleGridSize;

		uint32_t voxelIndex = pX + pY * sampleGridSize + pZ * sampleGridSize * sampleGridSize;
		voxelIndex = min(voxelIndex, uint32_t(sampleGrid_numCells - 1));

		uint8_t* rgba = (uint8_t*)&point.color;
		uint64_t R = rgba[0];
		uint64_t G = rgba[1];
		uint64_t B = rgba[2];

		// if(child->numPoints > 0){
		// 	R = 255;
		// 	G = 255;
		// 	B = 0;
		// }else if(child->numVoxels > 0){
		// 	R = 255;
		// 	G = 0;
		// 	B = 255;
		// }

		uint64_t COUNT = 1;
		uint64_t c64 = (R << 46) | (G << 28) | (B << 10) | COUNT;

		uint64_t old = atomicAdd(&sampleGrid[voxelIndex], c64);
		uint64_t oldCount = old & 0b11'1111'1111llu;

		uint32_t encodedVoxelIndex = (childIndex << 24) | voxelIndex;

		if(oldCount == 0){
			atomicAdd(&sh_numAccepted_total, 1);
			uint32_t acceptedIndex = atomicAdd(&sh_numAccepted_child, 1);
			accepted[acceptedIndex] = encodedVoxelIndex;
		}
	}

}



extern "C" __global__
void kernel(
	const Uniforms _uniforms,
	uint32_t* buffer,
	Node* nodes, uint32_t* _numNodes,
	Stats* stats
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;
	uniforms = _uniforms;

	uint32_t numNodes = *_numNodes;

	uint64_t sampleGrids_numCells = grid.num_blocks() * sampleGrid_numCells;
	uint64_t sampleGrids_numBytes = grid.num_blocks() * sampleGrid_numBytes;

	Node** bottomUnfilteredNodes       =  allocator->alloc<Node**>(100'000 * sizeof(Node*));
	uint32_t& numBottomUnfilteredNodes = *allocator->alloc<uint32_t*>(4);
	uint32_t& workIndexCounter         = *allocator->alloc<uint32_t*>(4);
	uint64_t* sampleGrids              =  allocator->alloc<uint64_t*>(sampleGrids_numBytes);

	uint64_t acceptedByteSize          = sizeof(uint32_t) * acceptedCapacity;
	uint32_t* accepteds                = allocator->alloc<uint32_t*>(grid.num_blocks() * acceptedByteSize, "list of accepted indices");
	uint32_t* accepted                 = accepteds + grid.block_rank() * acceptedCapacity;

	uint32_t& dbg                      = *allocator->alloc<uint32_t*>(4);

	uint64_t* sampleGrid = sampleGrids + grid.block_rank() * sampleGrid_numCells;

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 octreeMin = uniforms.boxMin;
	float3 octreeMax = octreeMin + cubeSize;
	float3 octreeSize = octreeMax - octreeMin;

	grid.sync();

	if(grid.thread_rank() == 0){
		numBottomUnfilteredNodes = 0;
		workIndexCounter = 0;
	}

	uint64_t tStart = nanotime();

	grid.sync();

	// clear all sample grids
	processRange(sampleGrids_numCells, [&](uint64_t cellIndex){
		sampleGrids[cellIndex] = 0;
	});

	grid.sync();

	// mark all leaf-nodes as filtered
	processRange(numNodes, [&](int index){
		Node* node = &nodes[index];

		if(node->isLeafFn()){
			node->isFiltered = true;
		}else{
			node->isFiltered = false;
		}
	});

	grid.sync();

	__shared__ uint32_t sh_workIndex;
	__shared__ uint32_t sh_numAccepted_child;
	__shared__ uint32_t sh_numAccepted_total;
	// __shared__ uint32_t sh_numVoxelsReplaced;

	for(int abc = 0; abc < 10; abc++){
		if(grid.thread_rank() == 0){
			workIndexCounter = 0;
			numBottomUnfilteredNodes = 0;
		}

		grid.sync();

		findBottomMostNodes(nodes, numNodes, bottomUnfilteredNodes, numBottomUnfilteredNodes);

		// if(abc == 1){
		// 	numBottomUnfilteredNodes = 1;
		// }

		// PRINT("%i bottom-most nodes found \n", numBottomUnfilteredNodes);

		grid.sync();
		if(numBottomUnfilteredNodes == 0) break;
		grid.sync();

		while(workIndexCounter < numBottomUnfilteredNodes){

			block.sync();

			if(block.thread_rank() == 0){
				sh_workIndex = atomicAdd(&workIndexCounter, 1);
				sh_numAccepted_total = 0;
				// sh_numVoxelsReplaced = 0;
			}

			block.sync();

			if(sh_workIndex >= numBottomUnfilteredNodes) break;

			// retrieve the node that this block should process
			Node* node = bottomUnfilteredNodes[sh_workIndex];

			float3 cubeMin = uniforms.boxMin;
			float nodeSize = cubeSize / pow(2.0f, float(node->level));
			float nodeMin_x = (float(node->X) + 0.0f) * nodeSize + cubeMin.x;
			float nodeMin_y = (float(node->Y) + 0.0f) * nodeSize + cubeMin.y;
			float nodeMin_z = (float(node->Z) + 0.0f) * nodeSize + cubeMin.z;

			Chunk* voxels = node->voxelChunks;
			int voxelChunkIndex = 0;
			int voxelIndexOffset = 0;
			
			for(int childIndex = 0; childIndex < 8; childIndex++){

				Node* child = node->children[childIndex];

				if(child == nullptr) continue;

				float childSize = cubeSize / pow(2.0f, float(child->level));
				float childMin_x = (float(child->X) + 0.0f) * childSize + cubeMin.x;
				float childMin_y = (float(child->Y) + 0.0f) * childSize + cubeMin.y;
				float childMin_z = (float(child->Z) + 0.0f) * childSize + cubeMin.z;

				sh_numAccepted_child = 0;

				block.sync();

				// POINTS
				doSampling(
					node, childIndex, child->points, 
					child->numPoints, sampleGrid, 
					sh_numAccepted_total, sh_numAccepted_child, 
					accepted
				);

				// VOXELS
				doSampling(
					node, childIndex, child->voxelChunks, 
					child->numVoxels, sampleGrid, 
					sh_numAccepted_total, sh_numAccepted_child, 
					accepted
				);

				block.sync();

				// EXTRACT
				for(
					uint32_t acceptedIndex = block.thread_rank();
					acceptedIndex < sh_numAccepted_child;
					acceptedIndex += block.num_threads()
				){

					uint32_t encodedVoxelIndex = accepted[acceptedIndex];
					uint32_t childIndex = (encodedVoxelIndex >> 24) & 0xff;
					uint32_t voxelIndex = encodedVoxelIndex & 0x00ffffff;

					uint64_t c64 = sampleGrid[voxelIndex];
					uint64_t R = (c64 >> 46) & 0b11'11111111'11111111;
					uint64_t G = (c64 >> 28) & 0b11'11111111'11111111;
					uint64_t B = (c64 >> 10) & 0b11'11111111'11111111;
					uint64_t C = (c64 >>  0) & 0b00'00000011'11111111;

					R = (R / C) & 0xff;
					G = (G / C) & 0xff;
					B = (B / C) & 0xff;

					int voxelInParentIndex = voxelIndexOffset + acceptedIndex;
					int targetVoxelChunkIndex = voxelInParentIndex / POINTS_PER_CHUNK;
					if(voxelChunkIndex < targetVoxelChunkIndex){
						voxels = voxels->next;
						voxelChunkIndex++;
					}

					int cx = (childIndex >> 2) & 1;
					int cy = (childIndex >> 1) & 1;
					int cz = (childIndex >> 0) & 1;

					int pX = cx * sampleGridSize + voxelIndex % sampleGridSize;
					int pY = cy * sampleGridSize + (voxelIndex % (sampleGridSize * sampleGridSize)) / sampleGridSize;
					int pZ = cz * sampleGridSize + voxelIndex / (sampleGridSize * sampleGridSize);

					Point voxel;
					voxel.x = nodeMin_x + nodeSize * (float(pX) + 0.5f) / 128.0f;
					voxel.y = nodeMin_y + nodeSize * (float(pY) + 0.5f) / 128.0f;
					voxel.z = nodeMin_z + nodeSize * (float(pZ) + 0.5f) / 128.0f;
					voxel.color = R | (G << 8) | (B << 16);

					// if(abc > 0){
					// 	voxel.color = 0x00ff00ff;
					// }

					// if(abc == 0){
					// 	voxel.color = 0x00ff0000;
					// }
					// if(abc == 1){
					// 	voxel.color = 0x000000ff;
					// }

					int voxelInChunkIndex = voxelInParentIndex % POINTS_PER_CHUNK;
					voxels->points[voxelInChunkIndex] = voxel;

					// clear sample grid
					sampleGrid[voxelIndex] = 0;
				}

				voxelIndexOffset += sh_numAccepted_child;

				block.sync();
			}

			block.sync();

			// assert(sh_numAccepted_total == node->numVoxels);


			if(block.thread_rank() == 0)
			if(sh_numAccepted_total != node->numVoxels)
			{
				printf("node[%i]: sh_numAccepted_total: %i. reference: %i \n", 
					sh_workIndex, sh_numAccepted_total, node->numVoxels);
			}

			// if(sh_workIndex < 10 && block.thread_rank() == 0){
			// 	printf("node[%i]: sh_numAccepted_total: %i. reference: %i \n", 
			// 		sh_workIndex, sh_numAccepted_total, node->numVoxels);
			// }

			node->isFiltered = true;
		}

		grid.sync();
	}

	grid.sync();

	uint64_t tEnd = nanotime();
	float millies = (nanotime() - tStart) / 1'000'000.0;

	PRINT("duration: %f ms\n", millies);
	PRINT("allocated bytes: %i MB\n", allocator->offset / (1024 * 1024));
}