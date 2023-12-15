// Some code in this file, particularly frustum, ray and intersection tests, 
// is adapted from three.js. Three.js is licensed under the MIT license
// This file this follows the three.js licensing
// License: MIT https://github.com/mrdoob/three.js/blob/dev/LICENSE

#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "utils.cuh"
#include "builtin_types.h"
#include "helper_math.h"
#include "HostDeviceInterface.h"

#include "math.cuh"
#include "rasterization.cuh"

namespace cg = cooperative_groups;

Allocator* allocator;
Uniforms uniforms;

constexpr int SPLAT_SIZE = 1;
constexpr uint32_t BACKGROUND_COLOR = 0x00332211;
// constexpr uint32_t BACKGROUND_COLOR = 0x00ffff00;
// constexpr uint32_t BACKGROUND_COLOR = 0x00ffffff;
// constexpr uint32_t BACKGROUND_COLOR = 0x000000ff;

// https://colorbrewer2.org/
uint32_t SPECTRAL[8] = {
	0x4f3ed5,
	0x436df4,
	0x61aefd,
	0x8be0fe,
	0x98f5e6,
	0xa4ddab,
	0xa5c266,
	0xbd8832,
};

struct PointBatch{
	uint32_t numPoints;
	Point points[50'000];
};

template<typename T>
struct Pool{

	T** items;
	int32_t numItems;

};



struct Data{
	uint32_t* buffer;
	Pool<PointBatch>* pointBatchPool;
	uint64_t* framebuffer;
	Lines* lines;
	Triangles* triangles;
	uint64_t* dbg_0;
	uint64_t* dbg_1;
};

Data* data = nullptr;

Data* allocateStuff(uint32_t* buffer){

	Data* data = allocator->alloc2<Data>(1);
	data->buffer = buffer;
	data->pointBatchPool = allocator->alloc2<Pool<PointBatch>>(1);
	data->pointBatchPool->numItems = MAX_POINT_BATCHES;
	data->pointBatchPool->items = allocator->alloc2<PointBatch*>(MAX_POINT_BATCHES);

	// int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	// int numPixels = int(uniforms.width) * int(uniforms.height);
	int numPixels = 4000 * 4000;
	data->framebuffer = allocator->alloc2<uint64_t>(numPixels);

	Lines* lines = allocator->alloc2<Lines>(1); 
	lines->count = 0;
	lines->vertices = allocator->alloc2<Vertex>(1'000'000);
	data->lines = lines;

	Triangles* triangles = allocator->alloc2<Triangles>(1);
	triangles->positions = allocator->alloc2<float3  >(1'000'000);
	triangles->uvs       = allocator->alloc2<float2  >(1'000'000);
	triangles->colors    = allocator->alloc2<uint32_t>(1'000'000);
	data->triangles = triangles;

	data->dbg_0 = allocator->alloc2<uint64_t>(1);
	data->dbg_1 = allocator->alloc2<uint64_t>(1);

	return data;
}

void toScreen(float3 boxMin, float3 boxMax, float2& screen_min, float2& screen_max){
	auto min8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = min(f0, f1);
		float m1 = min(f2, f3);
		float m2 = min(f4, f5);
		float m3 = min(f6, f7);

		float n0 = min(m0, m1);
		float n1 = min(m2, m3);

		return min(n0, n1);
	};

	auto max8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = max(f0, f1);
		float m1 = max(f2, f3);
		float m2 = max(f4, f5);
		float m3 = max(f6, f7);

		float n0 = max(m0, m1);
		float n1 = max(m2, m3);

		return max(n0, n1);
	};

	// compute node boundaries in screen space
	float4 p000 = {boxMin.x, boxMin.y, boxMin.z, 1.0f};
	float4 p001 = {boxMin.x, boxMin.y, boxMax.z, 1.0f};
	float4 p010 = {boxMin.x, boxMax.y, boxMin.z, 1.0f};
	float4 p011 = {boxMin.x, boxMax.y, boxMax.z, 1.0f};
	float4 p100 = {boxMax.x, boxMin.y, boxMin.z, 1.0f};
	float4 p101 = {boxMax.x, boxMin.y, boxMax.z, 1.0f};
	float4 p110 = {boxMax.x, boxMax.y, boxMin.z, 1.0f};
	float4 p111 = {boxMax.x, boxMax.y, boxMax.z, 1.0f};

	float4 ndc000 = uniforms.transform_updateBound * p000;
	float4 ndc001 = uniforms.transform_updateBound * p001;
	float4 ndc010 = uniforms.transform_updateBound * p010;
	float4 ndc011 = uniforms.transform_updateBound * p011;
	float4 ndc100 = uniforms.transform_updateBound * p100;
	float4 ndc101 = uniforms.transform_updateBound * p101;
	float4 ndc110 = uniforms.transform_updateBound * p110;
	float4 ndc111 = uniforms.transform_updateBound * p111;

	float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};
	float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{uniforms.width, uniforms.height, 1.0f, 1.0f};

	float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
	float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

	float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
	float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

	screen_min.x = smin_x;
	screen_min.y = smin_y;
	screen_max.x = smax_x;
	screen_max.y = smax_y;
}

extern "C" __global__
void kernel_init(
	uint32_t* buffer,
	Stats* stats,
	Tile* tiles, Chunk* chunks,
	Command* commandQueue, uint64_t* commandQueueCounter
){
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	Data* data = allocateStuff(buffer);

	// processRange(MAX_POINT_BATCHES, [&](int index){

	// 	PointBatch* batch = new PointBatch();
	// 	batch->numPoints = 0;

	// 	data->pointBatchPool->items[index] = batch;
	// });
}


extern "C" __global__
void kernel_chunkLoaded(
	const uint32_t chunkIndex,
	uint64_t ptr_points,
	Tile* tiles, Chunk* chunks
){
	// Allocator _allocator(buffer, 0);
	// allocator = &_allocator;

	// Data* data = allocateStuff(buffer);

	Chunk& chunk = chunks[chunkIndex];
	chunk.state = STATE_LOADED;
	memcpy(&chunk.points, &ptr_points, 8);

	printf("[kernel] chunk loaded!\n");
	printf("[kernel] chunkIndex: %u \n", chunkIndex);
	printf("[kernel] ptr_points: %llu \n", ptr_points);

	Point point = chunk.points[0];

	printf("[kernel] xyz: %.1f, %.1f, %.1f \n", point.x, point.y, point.z);
	printf("huh?\n");
}


extern "C" __global__
void kernel_render(
	uint32_t* buffer,
	const Uniforms _uniforms,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats,
	Tile* tiles, Chunk* chunks,
	Command* commandQueue, uint64_t* commandQueueCounter
){

	auto tStart = nanotime();

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();


	uniforms = _uniforms;
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	data = allocateStuff(buffer);

	uint64_t* framebuffer = data->framebuffer;

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// depth:            7f800000 (Infinity)
		// background color: 00332211 (aabbggrr)
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (0x7f800000ull << 32) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	rasterizeLines(data->lines, framebuffer, uniforms.width, uniforms.height, uniforms.transform);

	grid.sync();

	uint32_t* numProcessedTriangles = allocator->alloc2<uint32_t>(1);
	Triangles* triangles = data->triangles;

	if(grid.thread_rank() == 0){
		triangles->numTriangles = 0;
	}

	grid.sync();

	*data->dbg_0 = 0;

	grid.sync();

	if(grid.thread_rank() < uniforms.numTiles){
		Tile& tile = tiles[grid.thread_rank()];

		float3 pos = {
			tile.min.x * 0.5 + tile.max.x * 0.5,
			tile.min.y * 0.5 + tile.max.y * 0.5,
			tile.min.z * 0.5 + tile.max.z * 0.5
		};
		float3 size = float3{
			tile.max.x - tile.min.x,
			tile.max.y - tile.min.y,
			tile.max.z - tile.min.z
		} * 0.5f;


		bool isIntersectingFrustum = intersectsFrustum(uniforms.transform_updateBound, tile.min, tile.max);

		if(isIntersectingFrustum){

			float2 smin, smax;
			toScreen(tile.min, tile.max, smin, smax);

			// screen-space size
			float dx = smax.x - smin.x;
			float dy = smax.y - smin.y;

			float screen_center_x = ((smin.x + smax.x) * 0.5f - uniforms.width * 0.5f) / uniforms.width;
			float screen_center_y = ((smin.y + smax.y) * 0.5f - uniforms.height * 0.5f) / uniforms.height;
			float d = sqrt(screen_center_x * screen_center_x + screen_center_y * screen_center_y);

			// float w = clamp(1.0f - d, 0.0f, 1.0f);
			float w = exp(-d * d / 0.20f);
			float w2 = w * dx * dy;

			tile.isHighlyVisible = w2 > 20000;

			// drawBox(triangles, pos, size, tile.color);
		}
	}

	grid.sync();

	processRange(uniforms.numChunks, [&](int chunkIndex){

		Chunk& chunk = chunks[chunkIndex];
		Tile& tile = tiles[chunk.tileID];

		bool isIntersectingFrustum = intersectsFrustum(uniforms.transform_updateBound, chunk.min, chunk.max);
		if(!isIntersectingFrustum) return;

		float2 smin, smax;
		toScreen(chunk.min, chunk.max, smin, smax);

		// screen-space size
		float dx = smax.x - smin.x;
		float dy = smax.y - smin.y;

		float screen_center_x = ((smin.x + smax.x) * 0.5f - uniforms.width * 0.5f) / uniforms.width;
		float screen_center_y = ((smin.y + smax.y) * 0.5f - uniforms.height * 0.5f) / uniforms.height;
		float d = sqrt(screen_center_x * screen_center_x + screen_center_y * screen_center_y);

		// float w = clamp(powf(clamp(1.0f - d, 0.0f, 1.0f), 4.0f), 0.5f, 1.0f);
		// float w = 1.0f;
		float w = clamp(1.0f - d, 0.0f, 1.0f);
		w = exp(-d * d / 0.20f);
		float w2 = w * dx * dy;

		bool isHighlyVisible = w2 > 20000;

		float3 pos = {
			tile.min.x * 0.5 + tile.max.x * 0.5,
			tile.min.y * 0.5 + tile.max.y * 0.5,
			tile.min.z * 0.5 + tile.max.z * 0.5
		};
		float3 size = float3{
			tile.max.x - tile.min.x,
			tile.max.y - tile.min.y,
			tile.max.z - tile.min.z
		} * 0.5f;

		if(chunk.state == STATE_LOADED)
		{
			uint32_t boxColor = 0x0000ff00;
			drawBoundingBox(data->lines, pos, 2.05f * size, boxColor);
		}

		if(tile.isHighlyVisible){

			uint32_t boxColor = 0x000000ff;
			if(chunk.state == STATE_LOADED){
				boxColor = 0x0000ff00;
			}
			drawBoundingBox(data->lines, pos, 2.02f * size, boxColor);

			if(chunk.state == STATE_EMPTY){
				// LOAD CHUNK!

				uint32_t index_batchInPool = atomicAdd(&data->pointBatchPool->numItems, -1) - 1;

				if(index_batchInPool < 0){
					// not enough point batches in pool, revert and skip loading
					atomicAdd(&data->pointBatchPool->numItems, 1);
				}else{

					// PointBatch* batch = data->pointBatchPool->items[index_batchInPool];

					Command command;
					command.command = CMD_READ_CHUNK;
					
					CommandReadChunkData cmddata;
					cmddata.tileID = chunk.tileID;
					cmddata.chunkIndex = chunk.chunkIndex;
					cmddata.chunkID = chunkIndex;
					// cmddata.cptr_pointBatch = (uint64_t)batch;
					cmddata.cptr_pointBatch = 0;

					memcpy(command.data, &cmddata, sizeof(cmddata));
					
					uint32_t index = atomicAdd(commandQueueCounter, 1llu) % COMMAND_QUEUE_CAPACITY;
					commandQueue[index] = command;

					chunk.state = STATE_LOADING;
				}


				
			}
		}

	});

	grid.sync();

	// processRange(uniforms.numChunks, [&](int chunkIndex){

	// 	Chunk& chunk = chunks[chunkIndex];
	// 	Tile& tile = tiles[chunk.tileID];

	// 	if(chunk.state == STATE_LOADED){
	// 		rasterizePoints(chunk.points, chunk.numPoints, framebuffer, uniforms.width, uniforms.height, uniforms.transform);
	// 	}

	// });

	grid.sync();

	// if(grid.thread_rank() == 0){
	// 	// printf("data->dbg_0: %llu M \n", (*data->dbg_0) / 1'000'000llu);
	// 	printf("%i \n", uniforms.numChunks);
	// }

	// create some test commands
	// if(grid.thread_rank() < 7){

	// 	Command command;
	// 	command.command = CMD_READ_FILE;
	// 	uint32_t* data_u32 = (uint32_t*)command.data;
	// 	data_u32[0] = 13;

	// 	uint32_t index = atomicAdd(commandQueueCounter, 1llu) % COMMAND_QUEUE_CAPACITY;

	// 	commandQueue[index] = command;
	// }

	grid.sync();

	rasterizeTriangles(triangles, numProcessedTriangles, framebuffer, uniforms);
	rasterizeLines(data->lines, data->framebuffer, uniforms.width, uniforms.height, uniforms.transform);

	grid.sync();

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;
		// color = 0x000000ff;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});

}

