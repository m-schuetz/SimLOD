
#pragma once

#include "builtin_types.h"

constexpr int MAX_POINT_BATCHES = 10'000;

struct mat4{
	float4 rows[4];
};

struct Point{
	float x;
	float y;
	float z;
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
};

struct Uniforms{
	float width;
	float height;
	float time;
	float fovy_rad;
	mat4 world;
	mat4 view;
	mat4 proj;
	mat4 transform;
	mat4 transform_updateBound;
	mat4 transformInv_updateBound;

	uint64_t persistentBufferCapacity;
	uint64_t momentaryBufferCapacity;

	uint64_t frameCounter;
	
	float3 boxMin;
	float3 boxMax;

	bool showBoundingBox;
	bool showPoints;
	bool colorByNode;
	bool colorByLOD;
	bool doUpdateVisibility;
	bool doProgressive;
	float LOD;
	bool useHighQualityShading;
	float minNodeSize;
	int pointSize;
	bool updateStats;

	uint32_t numTiles;
	uint32_t numChunks;
};

struct Stats{
	uint32_t frameID                    = 0;
	uint32_t numNodes                   = 0;
	uint32_t numInner                   = 0;
	uint32_t numLeaves                  = 0;
	uint32_t numNonemptyLeaves          = 0;
	uint32_t numPoints                  = 0;
	uint32_t numVoxels                  = 0;
	uint64_t allocatedBytes_momentary   = 0;
	uint64_t allocatedBytes_persistent  = 0;
	uint32_t numVisibleNodes            = 0;
	uint32_t numVisibleInner            = 0;
	uint32_t numVisibleLeaves           = 0;
	uint32_t numVisiblePoints           = 0;
	uint32_t numVisibleVoxels           = 0;

	uint32_t batchletIndex              = 0;
	uint64_t numPointsProcessed         = 0;
	uint64_t numAllocatedChunks         = 0;
	uint64_t chunkPoolSize              = 0;
	uint32_t dbg                        = 0;

	bool memCapacityReached             = false;
};

constexpr int STATE_EMPTY               = 0;
constexpr int STATE_LOADING             = 1;
constexpr int STATE_LOADED              = 2;
constexpr int STATE_UNLOADING           = 3;

// Corresponds to a las or laz file
struct Tile{
	float3 min;
	float3 max;
	uint32_t color;
	uint32_t numPoints;
	uint32_t numPointsLoaded;
	uint32_t state;
	bool isHighlyVisible;
};

// Corresponds to a chunk of points within a tile.
// - in laz files, equivalent to a single compressed chunk. Typically 50k, but can differ.
// - las and others don't have inherent chunks, so we create chunks of about 50k points ourselves
struct Chunk{
	float3 min; 
	float3 max; 
	uint32_t tileID;
	uint32_t chunkIndex; // within tile
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
	uint32_t numPoints; 
	uint32_t numPointsLoaded;
	uint32_t state;
	Point* points;
};

constexpr int COMMAND_QUEUE_CAPACITY = 100'000;
constexpr int CMD_READ_FILE = 0;
constexpr int CMD_READ_CHUNK = 1;
constexpr int CMD_UNLOAD_CHUNK = 2;
constexpr int CMD_DBG = 1234;

struct CommandReadChunkData{
	uint32_t tileID;
	uint32_t chunkIndex;
	uint32_t chunkID;
	uint64_t cptr_pointBatch; // Host should allocate and store chunk's points here
};

struct CommandUnloadChunkData{
	uint32_t tileID;
	uint32_t chunkIndex;
	uint32_t chunkID;
	uint64_t cptr_pointBatch; // Device notifies host that this memory can be deallocated
};

struct Command{
	int command;
	uint8_t data[124];
};