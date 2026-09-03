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
// #include "rasterization.cuh"
#include "structures.cuh"

namespace cg = cooperative_groups;

Allocator* allocator;
Uniforms uniforms;
uint32_t* errorValue;
AllocatorGlobal* allocator_octree;

uint32_t* batchIndex;
Point* backlog_points = nullptr;
Node** backlog_targets = nullptr;
uint32_t* numBacklogPoints = nullptr;

Chunk** chunkQueue = nullptr;
int32_t* numAllocatedChunks = nullptr;
int32_t* chunkPoolSize = nullptr;
uint32_t* dbgCounters = nullptr;

void dbgPrintAllNodes(Node* nodes, uint32_t numNodes){
	processRange(numNodes, [&](int index){
		Node* node = &nodes[index];

		const char* flag = "    ";

		if(node->numPoints > node->counter){
			flag = "!!!!";
		}

		printf("[%8s]: counter: %5i, numPoints: %5i  %s \n", node->name, node->counter, node->numPoints, flag);
	});
}

bool sampleVoxel(Node* node, 
	uint32_t pX_full, uint32_t pY_full, uint32_t pZ_full, 
	Point& point, 
	float3 octreeMin, float3 octreeMax
){

	// if(node->grid == nullptr) return;
	// if(level != 0) return;

	if((point.color >> 24) != 0){

		uint32_t pointLevel = (point.color >> 24) - 1;
		printf("Point has level %u. But sample shouldnt be called for a point that was already accepted \n", pointLevel);
	}

	// node coordinate in current level's voxel precision
	// level 0: [0, 128), 
	// level 1: [0, 256), ...
	// uint32_t nX = node->X * 128 * (1 << node->level);
	// uint32_t nY = node->Y * 128 * (1 << node->level);
	// uint32_t nZ = node->Z * 128 * (1 << node->level);

	// point coordinate in current level's voxel precision
	// from max precision of 2^24 = [0, 16'777'216)
	// for level 0, we need to go from 2^24 to 2^7
	// for level 1, we need to go from 2^24 to 2^8
	// dividing by 2^n is equal to 2^(24 - n)
	// so we need to divide by 2^(17-level)
	uint32_t pX_leveled = pX_full / (1 << (17 - node->level));
	uint32_t pY_leveled = pY_full / (1 << (17 - node->level));
	uint32_t pZ_leveled = pZ_full / (1 << (17 - node->level));

	float pX = pX_leveled % 128;
	float pY = pY_leveled % 128;
	float pZ = pZ_leveled % 128;

	uint32_t voxelIndex = pX + pY * 128 + pZ * 128 * 128;
	uint32_t voxelGridElementIndex = voxelIndex / 32;
	uint32_t voxelGridElementBitIndex = voxelIndex % 32;

	uint32_t bitmask = 1 << voxelGridElementBitIndex;
	uint32_t old = atomicOr(&node->grid->values[voxelGridElementIndex], bitmask);

	if((old & bitmask) == 0){
		// first point!
		// atomicAdd(&node->numVoxels, 1);

		float3 boxSize = uniforms.boxMax - uniforms.boxMin;
		float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
		float3 cubeMin = uniforms.boxMin;
		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		// node-min
		float nodeMin_x = (float(node->X) + 0.0f) * nodeSize + cubeMin.x;
		float nodeMin_y = (float(node->Y) + 0.0f) * nodeSize + cubeMin.y;
		float nodeMin_z = (float(node->Z) + 0.0f) * nodeSize + cubeMin.z;

		// QUANTIZE
		// Point voxel;
		// voxel.x = nodeMin_x + nodeSize * (float(pX) + 0.5f) / 128.0f;
		// voxel.y = nodeMin_y + nodeSize * (float(pY) + 0.5f) / 128.0f;
		// voxel.z = nodeMin_z + nodeSize * (float(pZ) + 0.5f) / 128.0f;
		// voxel.color = point.color;

		uint32_t backlogIndex = atomicAdd(numBacklogPoints, 1);
		backlog_points[backlogIndex] = point;
		backlog_targets[backlogIndex] = node;

		uint32_t ctr = atomicAdd(&node->counter, 1);

		// if(node->level == 0){
		// 	printf("                                          accepted point %i into root! \n", ctr);
		// }

		// printf("accepted point into inner [%8s]! (%f, %f, %f) \n", node->name, point.x, point.y, point.z);

		// flag as "accepted by a node at level <i>"
		uint32_t encLevel = (node->level + 1) << 24;
		point.color = point.color | encLevel;

		return true;
	}else{
		return false;
	}
}


bool doCounting(
	Node* root, Point* points, int numPoints, 
	float3 octreeMin, float3 octreeMax,
	Node* nodes, uint32_t& numNodes,
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	uint32_t countIteration
){
	auto grid = cg::this_grid();

	if(ENABLE_TRACE) PRINT("== DO COUNTING - iteration %i ==\n", countIteration);

	constexpr int MAX_DEPTH = 16;
	float3 octreeSize = octreeMax - octreeMin;
	// quantization grid for coordinates
	// we want octree node coordinates for a max depth of, e.g., 16
	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	*numSpillingNodes = 0;

	grid.sync();

	auto countPoint = [&](Point& point){
		// node coordinate at MAX_DEPTH
		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize.x;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize.y;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize.z;

		// 2^4 = 16'777'216.0f
		// integer point coordinate relative to root node
		uint32_t pX = 16'777'216.0f * (point.x - octreeMin.x) / octreeSize.x;
		uint32_t pY = 16'777'216.0f * (point.y - octreeMin.y) / octreeSize.y;
		uint32_t pZ = 16'777'216.0f * (point.z - octreeMin.z) / octreeSize.z;

		Node* current = root;

		// traverse to leaf node, compute some data about it
		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		// don't count points that were already accepted by an inner node
		// (on purpose overflows to max(uint32) if not accepted)
		// uint32_t pointLevel = (point.color >> 24) - 1;
		if((point.color >> 24) != 0) return;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			// if node is an inner node (has sampling grid), then attempt accepting the point
			// if it is accepted, we stop traversing
			if(!current->isLeaf){
				bool isAcceptedByInner = sampleVoxel(current, pX, pY, pZ, point, octreeMin, octreeMax);
				if(isAcceptedByInner) return;
			}

			if(current->children[childIndex] == nullptr){
				// current == leaf!
				break;
			}else{
				current = current->children[childIndex];
			}
		}

		Node* leaf = current;


		// count points in leaf nodes
		if(leaf->countIteration < countIteration){

			// printf("accepted point into leaf [%8s]! (%f, %f, %f) \n", leaf->name, point.x, point.y, point.z);
			// one atomicAdd per point
			// uint32_t old = atomicAdd(&leaf->numPoints, 1);
			// if(old == MAX_POINTS_PER_NODE){
			// 	// needs splitting
			// 	uint32_t spillIndex = atomicAdd(numSpillingNodes, 1);
			// 	spillingNodes[spillIndex] = leaf;
			// }

			// merge atomicAdds within warps to reduce contention
			uint64_t leafptr = uint64_t(leaf);
			auto warp = cg::coalesced_threads();
			auto group = cg::labeled_partition(warp, leafptr);

			uint32_t old = 0;
			if(group.thread_rank() == 0){
				old = atomicAdd(&leaf->counter, group.num_threads());

				if(old <= MAX_POINTS_PER_NODE)
				if(old + group.num_threads() > MAX_POINTS_PER_NODE)
				{
					// needs splitting
					uint32_t spillIndex = atomicAdd(numSpillingNodes, 1);
					spillingNodes[spillIndex] = leaf;
				}
			}
		}
	};

	// Count points of current batch
	if(ENABLE_TRACE) PRINT("count current batch (%i) \n", numPoints);
	processRange(numPoints, [&](int pointID){
		Point& point = points[pointID];
		
		countPoint(point);
	});

	grid.sync();

	// count spilled points of previous iterations of current batch
	if(ENABLE_TRACE) PRINT("count spilled (%i) \n", *numSpilledPoints);
	processRange(*numSpilledPoints, [&](int pointID){
		Point& point = spilledPoints[pointID];
		
		countPoint(point);
	});

	grid.sync();

	// add points in spilled nodes to list of spilled points
	if(ENABLE_TRACE) PRINT("insert %i spilled nodes into spilled points \n", *numSpillingNodes);
	for(int i = 0; i < *numSpillingNodes; i++){

		grid.sync();

		Node* node = spillingNodes[i];

		if(node->numPoints == 0) continue;

		grid.sync();

		int numChunks = (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
		Chunk* chunk = node->points;

		uint32_t chunkIndex = 0;
		while(chunk != nullptr){
			grid.sync();

			int numPointsInChunk = min(node->numPoints - chunkIndex * POINTS_PER_CHUNK, POINTS_PER_CHUNK);

			processRange(numPointsInChunk, [&](int pointInChunkID){

				assert(chunk != nullptr);
				assert(pointInChunkID < POINTS_PER_CHUNK);

				Point point = chunk->points[pointInChunkID];

				uint32_t spillID = atomicAdd(numSpilledPoints, 1);

				assert(spillID < 3'000'000);
				spilledPoints[spillID] = point;

			});
			
			grid.sync();
			chunk = chunk->next;
			chunkIndex++;

		}
	}

	grid.sync();

	processRange(numNodes, [&](int nodeIndex){
		nodes[nodeIndex].countIteration = countIteration;
		// nodes[nodeIndex].counter = nodes[nodeIndex].numPoints;
	});

	grid.sync();

	// split the spilling nodes
	if(ENABLE_TRACE) PRINT("split %i spilling nodes \n", *numSpillingNodes);
	processRange(*numSpillingNodes, [&](int spillNodeIndex){
		Node* spillingNode = spillingNodes[spillNodeIndex];

		// create child nodes
		uint32_t childOffset = atomicAdd(&numNodes, 8);
		for(int i = 0; i < 8; i++){

			int cx = (i >> 2) & 1;
			int cy = (i >> 1) & 1;
			int cz = (i >> 0) & 1;

			Node child;
			child.counter = 0;
			child.numPoints = 0;
			child.points = nullptr;
			memset(&child.children, 0, 8 * sizeof(Node*));
			child.level = spillingNode->level + 1;
			child.X = 2 * spillingNode->X + cx;
			child.Y = 2 * spillingNode->Y + cy;
			child.Z = 2 * spillingNode->Z + cz;
			child.countIteration = 0;
			memcpy(&child.name[0], &spillingNode->name[0], 20);
			child.name[child.level] = i + '0';
			// child.voxels = Array<Point>();
			child.numVoxels = 0;
			child.numVoxelsStored = 0;
			child.isLeaf = true;
			
			nodes[childOffset + i] = child;

			spillingNode->children[i] = &nodes[childOffset + i];
		}

		// return chunks to chunkQueue
		Chunk* chunk = spillingNode->points;
		while(chunk != nullptr){
			
			Chunk* next = chunk->next;
			
			chunk->next = nullptr;
			int32_t oldIndex = atomicAdd(numAllocatedChunks, -1);
			int32_t newIndex = oldIndex - 1;
			chunkQueue[newIndex] = chunk;

			chunk = next;
		}

		spillingNode->numPoints = 0;
		spillingNode->points = nullptr;
		spillingNode->counter = 0;
		spillingNode->isLeaf = false;

		// allocate occupancy grid for the spilled node
		if(spillingNode->grid == nullptr){
			spillingNode->grid = (OccupancyGrid*)allocator_octree->alloc(sizeof(OccupancyGrid));
		}
	});

	grid.sync();

	// clear the newly allocated occupancy grids
	if(ENABLE_TRACE) PRINT("clear newly created occupancy grids \n", *numSpillingNodes);
	processRange(*numSpillingNodes * 65536, [&](int cellIndex){
		int gridIndex = cellIndex / 65536;
		int localCellIndex = cellIndex % 65536;

		Node* spillingNode = spillingNodes[gridIndex];

		spillingNode->grid->values[localCellIndex] = 0;
	});

	grid.sync();

	if(ENABLE_TRACE){
		dbgPrintAllNodes(nodes, numNodes);
	}

	grid.sync();

	return *numSpillingNodes == 0;
}

void expand(
	Node* root, Point* points, int numPoints, 
	float3 cubeMin, float3 cubeMax,
	Node* nodes, uint32_t& numNodes,
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	uint32_t batchIndex
){

	if(ENABLE_TRACE) PRINT("expand octree \n");

	auto grid = cg::this_grid();

	int numIterations = 0;
	for(int i = 0; i < 20; i++){

		grid.sync();

		bool isFinished = doCounting(root, points, numPoints, 
			cubeMin, cubeMax, 
			nodes, numNodes,
			spillingNodes, numSpillingNodes, 
			spilledPoints, numSpilledPoints,
			batchIndex + 1);

		grid.sync();

		numIterations++;

		if(isFinished){
			// do counting one more time, because we integrated lower LOD sampling
			// into counting, and we need to sample the last nodes we've split.
			doCounting(root, points, numPoints, 
				cubeMin, cubeMax, 
				nodes, numNodes,
				spillingNodes, numSpillingNodes, 
				spilledPoints, numSpilledPoints,
				batchIndex + 1);

				grid.sync();

			break;
		}
	}

	grid.sync();
}


void addBatch(
	Point* points, uint32_t batchSize,
	int batchIndex,
	float3 cubeMin, float3 cubeMax,
	Node* nodes, uint32_t& numNodes,
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(ENABLE_TRACE)
	PRINT(R""""(addBatch(
	points = %llu,
	batchSize = %u,
	cubeMin, cubeMax, 
	nodes = %llu, numNodes = %u,
	...
)
)"""", 
		points, batchSize, nodes, numNodes
	);

	if(grid.thread_rank() == 0){
		*numSpilledPoints = 0;
		*numBacklogPoints = 0;
	}

	grid.sync();

	Node* root = &nodes[0];

	// PRINT("BEFORE EXPAND \n");
	// PRINT("root.counter: %i \n", nodes[0].counter);
	// PRINT("root.numPoints: %i \n", nodes[0].numPoints);

	grid.sync();

	expand(root, points, batchSize, 
		cubeMin, cubeMax, 
		nodes, numNodes,
		spillingNodes, numSpillingNodes, 
		spilledPoints, numSpilledPoints,
		batchIndex
	);

	grid.sync();

	// {// DEBUG: check that number of points with target level 0 matches counter in root

	// 	PRINT("COUNTERS! \n");

	// 	// uint32_t* dbgCounters = allocator->alloc<uint32_t*>(255 * 4);
	// 	processRange(255, [&](int index){
	// 		dbgCounters[index] = 0;
	// 	});

	// 	grid.sync();

	// 	processRange(batchSize, [&](int index){
	// 		Point point = points[index];

	// 		uint32_t targetLevel = (point.color >> 24) - 1;
	// 		targetLevel = min(targetLevel, 255u);

	// 		atomicAdd(&dbgCounters[targetLevel], 1);
	// 	});

	// 	grid.sync();

	// 	PRINT("COUNTERS - BATCH \n");
	// 	processRange(255, [&](int i){
	// 		if(dbgCounters[i] > 0){
	// 			printf("[%3i]: %i \n", i, dbgCounters[i]);
	// 		}
	// 	});

	// 	grid.sync();

	// 	processRange(*numSpilledPoints, [&](int index){
	// 		Point point = spilledPoints[index];

	// 		uint32_t targetLevel = (point.color >> 24) - 1;
	// 		targetLevel = min(targetLevel, 255u);

	// 		if(targetLevel == 0){
	// 			atomicAdd(&dbgCounters[targetLevel], 1);
	// 		}
	// 	});

	// 	grid.sync();

	// 	PRINT("COUNTERS - BATCH + SPILLED \n");
	// 	processRange(255, [&](int i){
	// 		if(dbgCounters[i] > 0){
	// 			printf("[%3i]: %i \n", i, dbgCounters[i]);
	// 		}
	// 	});

	// 	grid.sync();

	// 	PRINT("AFTER EXPAND \n");
	// 	PRINT("root.counter: %i \n", nodes[0].counter);
	// 	PRINT("root.numPoints: %i \n", nodes[0].numPoints);

	// 	grid.sync();

	// 	if(dbgCounters[0] > nodes[0].counter){
	// 		PRINT("too many points for root... \n");
	// 		// assert(false); 
	// 		return;
	// 	}
	// }

	grid.sync();

	// return;

	// ALLOCATE MEMORY FOR POINTS IN NODES
	if(ENABLE_TRACE) PRINT("allocate memory for nodes  \n");
	processRange(numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		// if(nodeIndex == 0){
		// 	printf("root.numPoints = %i; counter = %i \n", node->numPoints, node->counter);
		// }

		// if(node->isLeaf())
		if(node->numPoints < node->counter){

			int numRequiredChunks = (node->counter + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
			int numExistingChunks = (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
			int numAdditionallyRequiredChunks = numRequiredChunks - numExistingChunks;
			
			if(numAdditionallyRequiredChunks > 0){

				if(ENABLE_TRACE) printf("node %10s requires %2i chunks, has %2i. Allocating %2i new chunks. \n", 
					node->name, numRequiredChunks, numExistingChunks, numAdditionallyRequiredChunks
				);

				Chunk* prevChunk = node->points;
				for(int i = 1; i < numExistingChunks; i++){
					prevChunk = prevChunk->next;
				}

				for(int i_chunk = 0; i_chunk < numAdditionallyRequiredChunks; i_chunk++){
					uint32_t chunkIndex = atomicAdd(numAllocatedChunks, 1);
					Chunk* chunk = nullptr;

					if(chunkIndex >= *chunkPoolSize){
						// allocate a new chunk if chunk pool is too small
						chunk = (Chunk*)allocator_octree->alloc(sizeof(Chunk));
					}else{
						// otherwise take from chunk pool
						chunk = chunkQueue[chunkIndex];
					}

					chunk->next = nullptr;

					if(prevChunk == nullptr){
						node->points = chunk;
						prevChunk = chunk;
					}else{
						prevChunk->next = chunk;
						prevChunk = chunk;
					}
				}
				
			}
		}

		// if(nodeIndex == 0){
		// 	printf("root.points = %llu \n", node->points);
		// }
	});
	if(ENABLE_TRACE) {PRINT("allocation done \n"); grid.sync();}

	grid.sync();
	// raise chunk pool size counter if we allocated new chunks
	if(ENABLE_TRACE) {PRINT("raise? \n"); grid.sync();}
	if(grid.thread_rank() == 0){
		*chunkPoolSize = max(*chunkPoolSize, *numAllocatedChunks);
	}
	if(ENABLE_TRACE) {PRINT("raise done? \n"); grid.sync();}
	grid.sync();

	// INSERT POINTS INTO NODES 
	if(ENABLE_TRACE) {PRINT("insert points into nodes \n"); grid.sync();}
	constexpr int MAX_DEPTH = 16;
	float fGridSize = pow(2.0f, float(MAX_DEPTH));
	float3 cubeSize = cubeMax - cubeMin;
	
	auto insertPoint = [&](Point point, bool dbgFromSpilled){
		// node coordinate at MAX_DEPTH
		uint32_t X = fGridSize * (point.x - cubeMin.x) / cubeSize.x;
		uint32_t Y = fGridSize * (point.y - cubeMin.y) / cubeSize.y;
		uint32_t Z = fGridSize * (point.z - cubeMin.z) / cubeSize.z;

		Node* current = root;

		// traverse to leaf node, compute some data about it
		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		uint32_t targetLevel = (point.color >> 24) - 1;
		targetLevel = min(targetLevel, 255u);

		// if(ENABLE_TRACE && dbgFromSpilled) printf("find target node for spilled point\n");

		for(; level < MAX_DEPTH; level++){

			if(current == nullptr){
				printf("error: required chunk does not exist \n");
			}

			// if accepted by inner, point will have a targetLevel.
			// in that case, break earlier at that level.
			if(level == targetLevel) break;

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			if(current->children[childIndex] == nullptr){
				// current == leaf!
				break;
			}else{
				current = current->children[childIndex];
			}
		}

		Node* target = current;

		// if(ENABLE_TRACE && dbgFromSpilled) printf("target: %llu \n", target);

		// printf("adding to [%8s] point(%f, %f, %f) \n", target->name, point.x, point.y, point.z);

		uint32_t pointInNodeIndex = atomicAdd(&target->numPoints, 1);
		uint32_t chunkIndex = pointInNodeIndex / POINTS_PER_CHUNK;
		uint32_t pointInChunkIndex = pointInNodeIndex % POINTS_PER_CHUNK;

		Chunk* chunk = target->points;

		// if(ENABLE_TRACE && dbgFromSpilled) printf("chunk: %llu \n", chunk);

		if(target->numPoints > target->counter){
			printf("pointInNodeIndex(%3i) > counter(%3i); target: %6s; targetLevel: %i \n", 
				pointInNodeIndex, target->counter, target->name, targetLevel);
		}

		// if(chunk == nullptr){
		// 	printf("00 chunk is null. node: %6s \n", target->name);
		// }

		// if(targetLevel == 0){
		// 	printf("point targets level %i. pointInNodeIndex: %i \n", targetLevel, pointInNodeIndex);
		// }

		// printf("targetLevel: %3u; targetNode: %8s; node->numPoints: %6i, node->counter: %6i chunk: %llu \n", 
		// 	targetLevel, target->name, target->numPoints, target->counter, chunk
		// );

		assert(chunk != nullptr);

		for(int i = 0; i < chunkIndex; i++){
			if(chunk == 0) break;

			chunk = chunk->next;
			if(ENABLE_TRACE && dbgFromSpilled) printf("chunk: %llu \n", chunk);

			if(chunk == nullptr){
				printf("10 chunk is null. node: %6s; node->counter: %i, node->numPoints: %i, targetLevel: %i; pointIndex: %i; pInChunkIndex: %i; chunkIndex: %i \n", 
					target->name, target->counter, target->numPoints, targetLevel, pointInNodeIndex, pointInChunkIndex, chunkIndex);
			}
		}

		if(chunk == nullptr){
			// printf("20 chunk is null. node: %6s \n", target->name);
			printf("chunk is null targetLevel: %3u; targetNode: %8s; node->numPoints: %6i, node->counter: %6i chunk: %llu \n", 
				targetLevel, target->name, target->numPoints, target->counter, chunk
			);
		}

		chunk->points[pointInChunkIndex] = point;
	};

	// INSERT POINTS FROM CURRENT BATCH
	if(ENABLE_TRACE) {PRINT("insert %i points from current batch \n", batchSize); grid.sync();}
	processRange(batchSize, [&](int pointID){
		Point point = points[pointID];
		
		insertPoint(point, false);
	});

	grid.sync();

	// PRINT("AFTER INSERT BATCH \n");
	// PRINT("root.counter: %i \n", nodes[0].counter);
	// PRINT("root.numPoints: %i \n", nodes[0].numPoints);

	// grid.sync();

	// INSERT POINTS FROM SPILLED NODES
	// (essentially redistributing from spilled to new leaves)
	if(ENABLE_TRACE) {PRINT("insert %i points from spilled nodes \n", *numSpilledPoints); grid.sync();}
	processRange(*numSpilledPoints, [&](int pointID){
		assert(pointID < 3'000'000);

		Point point = spilledPoints[pointID];

		insertPoint(point, true);
	});

	grid.sync();

	if(ENABLE_TRACE) {PRINT("insertion done \n"); grid.sync();}

	// // ALLOCATE VOXEL MEMORY FOR EACH NODE
	// processRange(numNodes, [&](int nodeIndex){

	// 	Node* node = &nodes[nodeIndex];

	// 	int requiredChunks = (node->numVoxels + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;

	// 	if(requiredChunks == 0) return;

	// 	if(node->voxelChunks == nullptr){
	// 		node->voxelChunks = (Chunk*)allocator_octree->alloc(sizeof(Chunk));
	// 		node->voxelChunks->next = nullptr;
	// 	}

	// 	Chunk* chunk = node->voxelChunks;

	// 	for(int chunkIndex = 1; chunkIndex < requiredChunks; chunkIndex++){

	// 		if(chunk->next == nullptr){
	// 			Chunk* newChunk = (Chunk*)allocator_octree->alloc(sizeof(Chunk));
	// 			newChunk->next = nullptr;
	// 			chunk->next = newChunk;
	// 		}

	// 		chunk = chunk->next;

	// 	}

	// });

	// grid.sync();

	// // INSERT VOXELS
	// processRange(*numBacklogPoints, [&](int index){
	// 	Point point = backlog_points[index];
	// 	Node* target = backlog_targets[index];

	// 	uint32_t voxelIndex = atomicAdd(&target->numVoxelsStored, 1);
	// 	uint32_t chunkIndex = voxelIndex / POINTS_PER_CHUNK;

	// 	Chunk* chunk = target->voxelChunks;

	// 	for(int i = 0; i < chunkIndex; i++){
	// 		chunk = chunk->next;
	// 	}

	// 	uint32_t chunkLocalVoxelIndex = voxelIndex % POINTS_PER_CHUNK;

	// 	chunk->points[chunkLocalVoxelIndex] = point;
	// });

	grid.sync();
}

extern "C" __global__
void kernel_construct(
	const Uniforms _uniforms,
	Point* points,
	uint32_t batchSize,
	uint32_t* buffer,
	uint8_t* buffer_octree,
	Node* nodes, uint32_t* _numNodes,
	Stats* stats
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uniforms = _uniforms;

	// if(uniforms.numBatchesProcessed > 1000) return;

	// if(ENABLE_TRACE) 
	// {
	// 	PRINT("======================================================================\n") 
	// 	PRINT("=== launching kernel_construct! ======================================\n") 
	// 	PRINT("======================================================================\n") 
	// 	PRINT("&points:             %llu \n", points);
	// 	PRINT("batchSize:           %i \n", batchSize);
	// 	PRINT("&buffer:             %llu \n", buffer);
	// 	PRINT("buffer_octree:       %llu \n", buffer_octree);
	// 	PRINT("nodes:               %llu \n", nodes);
	// 	PRINT("_numNodes:           %u \n", *_numNodes);
	// 	PRINT("#processedBatchs:    %u \n", uniforms.numBatchesProcessed);
	// 	PRINT("======================================================================\n") 
	// }

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	allocator_octree = (AllocatorGlobal*)buffer_octree;

	batchIndex = allocator->alloc<uint32_t*>(4);
	grid.sync();

	// ALLOCATE STUFF
	uint32_t& numNodes           = *_numNodes;
	uint32_t& numPointsAdded     = *allocator->alloc<uint32_t*>(4);

	if(ENABLE_TRACE){
		dbgPrintAllNodes(nodes, numNodes);
	}

	// memory for "backlog"
	backlog_points     = allocator->alloc<Point*>(10'000'000 * sizeof(Point));
	backlog_targets    = allocator->alloc<Node**>(10'000'000 * sizeof(Node*));
	numBacklogPoints   = allocator->alloc<uint32_t*>(4);

	dbgCounters = allocator->alloc<uint32_t*>(255 * 4);

	// List of nodes that received too many points and ned to be split
	Node** spillingNodes         =  allocator->alloc<Node**>(10'000 * sizeof(Node*));
	uint32_t* numSpillingNodes   =  allocator->alloc<uint32_t*>(4);

	// Points from spilled&split nodes that need to be redistributed to the new leaves
	// It's quite large because in the worst case, a single point can trigger 
	// <MAX_POINTS_PER_NODE> points to be spilled
	Point* spilledPoints = allocator->alloc<Point*>(10'000'000 * sizeof(Point));
	uint32_t* numSpilledPoints = allocator->alloc<uint32_t*>(4);

	numAllocatedChunks = allocator->alloc<int32_t*>(4);
	chunkPoolSize = allocator->alloc<int32_t*>(4);
	chunkQueue = allocator->alloc<Chunk**>(sizeof(Chunk*) * 1'000'000);

	grid.sync();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;
	float3 cubeMax = cubeMin + cubeSize;
	float3 cubePosition = uniforms.boxMin + cubeSize * 0.5f;

	auto tStartExpand = nanotime();

	{
		Node* root = &nodes[0];

		// RESET OCTREE 
		if(uniforms.frameCounter == 0 || uniforms.requestReset){

			if(ENABLE_TRACE) PRINT("resetting octree \n");
			
			if(grid.thread_rank() == 0){
				
				allocator_octree->buffer = buffer_octree;
				allocator_octree->offset = 16; // 16-aligned, first 8 byte is allocator itself

				numNodes = 1;
				numPointsAdded = 0;
				*batchIndex = 0;
				*numAllocatedChunks = 0;
				*chunkPoolSize = 0;

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
				root->points = nullptr;
				root->isLeaf = true;
			}

			grid.sync();

			// clear occupancy grid
			processRange(65536, [&](int index){
				root->grid->values[index] = 0;
			});

			grid.sync();
		}

		grid.sync();

		addBatch(
			points, batchSize,
			*batchIndex,
			cubeMin, cubeMax,
			nodes, numNodes,
			spillingNodes, numSpillingNodes,
			spilledPoints, numSpilledPoints
		);
		
		grid.sync();

		if(grid.thread_rank() == 0){
			atomicAdd(batchIndex, 1);
		}

	}

	auto tEndExpand = nanotime();
	float durationExpandMS = double(tEndExpand - tStartExpand) / 1'000'000.0;

	grid.sync();
	
	// if(grid.thread_rank() == 0){
	// 	if(durationExpandMS > 0.1){
	// 		printf("expand: %4.1f  \n", durationExpandMS);
	// 	}else{
	// 		// printf("expand:    -   draw: %4.1f \n", durationMS);
	// 	}
	// }

	grid.sync();

	uint32_t* counter_inner = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_leaves = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_nonempty_leaves = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_points = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_voxels = allocator->alloc<uint32_t*>(4);

	if(grid.thread_rank() == 0){
		*counter_inner = 0;
		*counter_leaves = 0;
		*counter_nonempty_leaves = 0;
		*counter_points = 0;
		*counter_voxels = 0;
	}
	grid.sync();

	processRange(numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		if(node->isLeaf){
			atomicAdd(counter_leaves, 1);

			if(node->numPoints > 0){
				atomicAdd(counter_nonempty_leaves, 1);
			}
		}else{
			atomicAdd(counter_inner, 1);
		}

		atomicAdd(counter_points, node->numPoints);
		atomicAdd(counter_voxels, node->numVoxels);
	});

	grid.sync();

	if(grid.thread_rank() == 0){

		// printf("=======\n");
		// printf("#nodes: %i\n", numNodes);
		// printf("    #inner: %i\n", *counter_inner);
		// printf("    #leaves: %i\n", *counter_leaves);
		// printf("allocated bytes (momentary buffer):  %llu MB \n", allocator->offset / (1024 * 1024));
		// printf("allocated bytes (persistent buffer): %llu MB \n", 
		// 	allocator_octree->offset / (1024 * 1024)
		// );

		stats->numNodes = numNodes;
		stats->numInner = *counter_inner;
		stats->numLeaves = *counter_leaves;
		stats->numNonemptyLeaves = *counter_nonempty_leaves;
		stats->numPoints = *counter_points;
		stats->numVoxels = *counter_voxels;
		stats->allocatedBytes_momentary = allocator->offset;
		stats->allocatedBytes_persistent = allocator_octree->offset;
	}
}


