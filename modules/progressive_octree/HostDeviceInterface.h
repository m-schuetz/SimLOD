
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

// ============================================================================
// 宿主(kernel 参数按值传递、Stats DtoH 回读)与设备侧共享布局契约锁。
// 本头文件被 MSVC(宿主) 与 NVRTC(kernel) 双侧编译，断言因此被双侧校验：
// 任一侧布局理解不一致时，对应编译立即失败，而非运行期数据错位。
// 有意修改以下结构时必须同步更新断言。
// 注意：NVRTC 的词法器不接受字符串字面量中的非 ASCII 字符，断言消息只能用英文。
// ============================================================================
static_assert(sizeof(mat4) == 64,     "mat4: 4 rows of float4");
static_assert(sizeof(Uniforms) == 480, "Uniforms is passed to kernels by value as one block; host and device must agree");
static_assert(sizeof(Stats) == 112,   "Stats must match cuMemAlloc(&cptr_stats, 112) at startup and the DtoH readback");