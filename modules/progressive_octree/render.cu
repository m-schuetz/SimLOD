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

void rasterizeChunk(
	Chunk* chunk, 
	Tile* tile,
	uint64_t* target, 
	float width, float height, 
	mat4 transform
){

	processRange(chunk->numPoints, [&](int index){

		Point point = chunk->points[index];

		point.x += tile->min.x;
		point.y += tile->min.y;
		point.z += tile->min.z;

		float4 ndc = transform * float4{point.x, point.y, point.z, 1.0f};

		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;
		float depth = ndc.w;

		int x = (ndc.x * 0.5 + 0.5) * width;
		int y = (ndc.y * 0.5 + 0.5) * height;

		if(x > 1 && x < width  - 2.0)
		if(y > 1 && y < height - 2.0){

			// SINGLE PIXEL
			uint32_t pixelID = x + int(width) * y;
			uint64_t udepth = *((uint32_t*)&depth);
			uint64_t encoded = (udepth << 32) | point.color;

			atomicMin(&target[pixelID], encoded);

			// POINT SPRITE
			// for(int ox : {-2, -1, 0, 1, 2})
			// for(int oy : {-2, -1, 0, 1, 2}){
			// 	uint32_t pixelID = (x + ox) + int(uniforms.width) * (y + oy);
			// 	uint64_t udepth = *((uint32_t*)&depth);
			// 	uint64_t encoded = (udepth << 32) | point.color;

			// 	atomicMin(&target[pixelID], encoded);
			// }
		}

	});

}

// from: https://stackoverflow.com/a/51549250
// TODO: License
__forceinline__ float atomicMinFloat(float * addr, float value) {
	float old;
	old = (value >= 0) ? 
		__int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

// from: https://stackoverflow.com/a/51549250
// TODO: License
__forceinline__ float atomicMaxFloat(float * addr, float value) {
	float old;
	old = (value >= 0) ?
		__int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
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
}

extern "C" __global__
void kernel_chunkLoaded(
	uint32_t* buffer,
	const uint32_t chunkIndex,
	const uint32_t chunkID,
	uint64_t ptr_points,
	Tile* tiles, Chunk* chunks
){

	auto grid = cg::this_grid();
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	Data* data = allocateStuff(buffer);

	uint32_t* sumColors = allocator->alloc2<uint32_t>(4);

	Chunk& chunk = chunks[chunkID];
	chunk.state = STATE_LOADED;

	Tile& tile = tiles[chunk.tileID];

	memcpy(&chunk.points, &ptr_points, 8);

	float3 min = tile.min;
	
	// compute new bounding box and sum of colors
	grid.sync();
	if(grid.thread_rank() == 0){
		chunk.min = {Infinity, Infinity, Infinity};
		chunk.max = {-Infinity, -Infinity, -Infinity};
		sumColors[0] = 0;
		sumColors[1] = 0;
		sumColors[2] = 0;
		sumColors[3] = 0;
	}

	grid.sync();

	processRange(chunk.numPoints, [&](int index){
		Point point = chunk.points[index];

		atomicMinFloat(&chunk.min.x, point.x + min.x);
		atomicMinFloat(&chunk.min.y, point.y + min.y);
		atomicMinFloat(&chunk.min.z, point.z + min.z);
		atomicMaxFloat(&chunk.max.x, point.x + min.x);
		atomicMaxFloat(&chunk.max.y, point.y + min.y);
		atomicMaxFloat(&chunk.max.z, point.z + min.z);

		atomicAdd(&sumColors[0], uint32_t(point.rgba[0]));
		atomicAdd(&sumColors[1], uint32_t(point.rgba[1]));
		atomicAdd(&sumColors[2], uint32_t(point.rgba[2]));
		atomicAdd(&sumColors[3], 1u);
	});

	grid.sync();

	// set chunk's color to average of point
	if(grid.thread_rank() == 0){
		chunk.rgba[0] = sumColors[0] / sumColors[3];
		chunk.rgba[1] = sumColors[1] / sumColors[3];
		chunk.rgba[2] = sumColors[2] / sumColors[3];
	}

	// realign points relative to chunk's new bounding box.
	// processRange(chunk.numPoints, [&](int index){
	// 	Point point = chunk.points[index];

	// 	point.x = point.x + min.x - chunk.min.x;
	// 	point.y = point.y + min.y - chunk.min.y;
	// 	point.z = point.z + min.z - chunk.min.z;

	// 	chunk.points[index] = point;

	// });
}

extern "C" __global__
void kernel_chunkUnloaded(
	const uint32_t chunkID,
	Chunk* chunks
){
	Chunk& chunk = chunks[chunkID];
	chunk.state = STATE_EMPTY;
	chunk.points = nullptr;
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
			float w = exp(-d * d / 0.020f);
			float w2 = w * dx * dy;

			tile.isHighlyVisible = w2 > 20000;
			// tile.isHighlyVisible = false;

			if(!tile.isHighlyVisible){
				drawBox(triangles, pos, size, tile.color);
			}else{
				drawBoundingBox(data->lines, pos, 2.0f * size, 0x0000ffff);
			}
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
		w = exp(-d * d / 0.0020f);
		float w2 = w * dx * dy;

		bool isHighlyVisible = w2 > 20000;

		float3 pos = {
			chunk.min.x * 0.5 + chunk.max.x * 0.5,
			chunk.min.y * 0.5 + chunk.max.y * 0.5,
			chunk.min.z * 0.5 + chunk.max.z * 0.5
		};
		float3 size = float3{
			chunk.max.x - chunk.min.x,
			chunk.max.y - chunk.min.y,
			chunk.max.z - chunk.min.z
		} * 0.5f;

		if(chunk.state == STATE_LOADED && chunk.chunkIndex % 1 == 0)
		{
			uint32_t boxColor = 0x0000ff00;
			// drawBoundingBox(data->lines, pos, 2.05f * size, boxColor);
			// drawBox(data->triangles, pos, size, chunk.color);
		}

		if(tile.isHighlyVisible){

			uint32_t boxColor = 0x000000ff;
			if(chunk.state == STATE_LOADED){
				boxColor = 0x0000ff00;
			}
			// drawBoundingBox(data->lines, pos, 2.02f * size, boxColor);

			if(chunk.state == STATE_EMPTY){
				// LOAD CHUNK!

				uint32_t index_batchInPool = atomicAdd(&data->pointBatchPool->numItems, -1) - 1;

				if(index_batchInPool < 0){
					// not enough point batches in pool, revert and skip loading
					atomicAdd(&data->pointBatchPool->numItems, 1);
				}else{
					Command command;
					command.command = CMD_READ_CHUNK;
					
					CommandReadChunkData cmddata;
					cmddata.tileID = chunk.tileID;
					cmddata.chunkIndex = chunk.chunkIndex;
					cmddata.chunkID = chunkIndex;
					cmddata.cptr_pointBatch = 0;

					memcpy(command.data, &cmddata, sizeof(cmddata));
					
					uint32_t index = atomicAdd(commandQueueCounter, 1llu) % COMMAND_QUEUE_CAPACITY;
					commandQueue[index] = command;

					chunk.state = STATE_LOADING;
				}
			}
		}else if(chunk.state == STATE_LOADED && !tile.isHighlyVisible){

			// chunk is loaded but not high priority -> unload points

			Command command;
			command.command = CMD_UNLOAD_CHUNK;

			CommandUnloadChunkData cmddata;
			cmddata.tileID = chunk.tileID;
			cmddata.chunkIndex = chunk.chunkIndex;
			cmddata.chunkID = chunkIndex;
			cmddata.cptr_pointBatch = (uint64_t)chunk.points;

			memcpy(command.data, &cmddata, sizeof(cmddata));
					
			uint32_t index = atomicAdd(commandQueueCounter, 1llu) % COMMAND_QUEUE_CAPACITY;
			commandQueue[index] = command;

			chunk.state = STATE_UNLOADING;

			// if(chunk.chunkIndex == 0)
			// {
				drawBoundingBox(data->lines, pos, 2.02f * size, 0x000000ff);
			// 	drawBox(data->triangles, pos, size, tile.color);
			// }
		}

	});

	grid.sync();

	uint32_t* numChunksVisible = allocator->alloc2<uint32_t>(1);
	Chunk* visibleChunks = allocator->alloc2<Chunk>(1'000'000);

	if(grid.thread_rank() == 0){
		*numChunksVisible = 0;
	}

	grid.sync();

	processRange(uniforms.numChunks, [&](int chunkIndex){

		Chunk& chunk = chunks[chunkIndex];
		Tile& tile = tiles[chunk.tileID];

		if(chunk.state == STATE_LOADED){
			uint32_t index = atomicAdd(numChunksVisible, 1);
			visibleChunks[index] = chunk;
			// rasterizePoints(chunk.points, chunk.numPoints, framebuffer, uniforms.width, uniforms.height, uniforms.transform);
		}

	});

	grid.sync();

	// if(grid.thread_rank() == 0){
	// 	printf("*numChunksVisible: %i\n", *numChunksVisible);
	// }

	for(int i = 0; i < *numChunksVisible; i++){
		Chunk chunk = visibleChunks[i];
		Tile* tile = &tiles[chunk.tileID];
		rasterizeChunk(&chunk, tile, framebuffer, uniforms.width, uniforms.height, uniforms.transform);
	}

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

