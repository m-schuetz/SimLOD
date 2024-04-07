
#pragma once

#include "builtin_types.h"

struct mat4{
	float4 rows[4];
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
	bool colorWhite;
	bool doUpdateVisibility;
	bool doProgressive;
	float LOD;
	bool useHighQualityShading;
	float minNodeSize;
	int pointSize;
	bool updateStats;
	bool enableEDL;
	float edlStrength;
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
	uint32_t numChunksPoints            = 0;
	uint32_t numChunksVoxels            = 0;

	uint32_t batchletIndex              = 0;
	uint64_t numPointsProcessed         = 0;
	uint64_t numAllocatedChunks         = 0;
	uint64_t chunkPoolSize              = 0;
	uint32_t dbg                        = 0;

	bool memCapacityReached             = false;
};