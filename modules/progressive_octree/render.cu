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
#include "rasterization.cuh"
#include "structures.cuh"

#include "../CudaPrint/CudaPrint.cuh"

namespace cg = cooperative_groups;

Allocator* allocator;
Uniforms uniforms;
Lines* _lines = nullptr;

constexpr int VISIBLITY_DISJUNCT = 0;
constexpr int VISIBLITY_ADDING = 1;
constexpr int VISIBLITY_SKIP = 2;
constexpr int visiblityMethod = VISIBLITY_DISJUNCT;

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

uint32_t getLodColor(int level){
	int index = float(8 - level) * 1.8f;
	index = clamp(index, 0, 7);

	uint32_t color = SPECTRAL[index];

	// if(level == 4) return 0x000000ff;
	// if(level == 8) return 0x0000ffff;
	
	return color;
}

void drawPoint(Point point, Node* node, uint64_t* framebuffer){
	float4 ndc = uniforms.transform * float4{point.x, point.y, point.z, 1.0f};
	float depth = ndc.w;
	ndc = ndc / ndc.w;

	int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
	int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

	if(x > 1 && x < uniforms.width  - 2.0)
	if(y > 1 && y < uniforms.height - 2.0)
	{

		uint32_t color = point.color;
		if(uniforms.colorByNode){
			color = (node->getID() % 127) * 123456789ull;
		}else if(uniforms.colorByLOD){
			color = getLodColor(node->level);
		}

		// float w = y - 3.0 * x;
		// if(w > -3800.0){
		// 	color = point.color;
		// }else{
		// 	color = (node->getID() % 127) * 123456789ull;
		// 	color = getLodColor(node->level);
		// }

		for(int ox = 0; ox < uniforms.pointSize; ox++)
		for(int oy = 0; oy < uniforms.pointSize; oy++)
		{
			uint32_t px = clamp(x + ox, 0, int(uniforms.width));
			uint32_t py = clamp(y + oy, 0, int(uniforms.height));

			uint32_t pixelID = px + int(uniforms.width) * py;
			uint64_t udepth = *((uint32_t*)&depth);
			uint64_t encoded = (udepth << 32) | color;

			if(encoded < framebuffer[pixelID]){
				atomicMin(&framebuffer[pixelID], encoded);
			}
		}

	}
}

void drawNode(Chunk* chunk, Node* node, uint32_t numElements, uint64_t* framebuffer){
	auto block = cg::this_thread_block();
	
	int chunkIndex = 0;

	for(
		int pointIndex = block.thread_rank(); 
		pointIndex < numElements; 
		pointIndex += block.num_threads()
	){
		int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

		if(chunkIndex < targetChunkIndex){
			chunk = chunk->next;
			chunkIndex++;
		}

		int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
		Point point = chunk->points[pointInChunkIndex];

		drawPoint(point, node, framebuffer);

		// debug pseudo-voxel
		// if(node->numPoints == 0)
		// { 
		// 	float s = nodeSize / 256.0;
		// 	float numSteps = 5;

		// 	for(float sx = -numSteps; sx <= numSteps; sx += 1.0f)
		// 	for(float sy = -numSteps; sy <= numSteps; sy += 1.0f)
		// 	{
		// 		float ox = s * sx / numSteps;
		// 		float oy = s * sy / numSteps;
		// 		Point p;
		// 		p.x = point.x + ox;
		// 		p.y = point.y + oy;
		// 		p.z = point.z + s;
		// 		p.color = point.color;

		// 		drawPoint(point);

		// 		p.x = point.x + ox;
		// 		p.y = point.y + s;
		// 		p.z = point.z + oy;
		// 		drawPoint(point);

		// 		p.x = point.x + s;
		// 		p.y = point.y + ox;
		// 		p.z = point.z + oy;
		// 		drawPoint(p);
		// 	}
		// }
	}
};

void drawNodes(Node* nodes, uint32_t numNodes, uint64_t* framebuffer){

	if(!uniforms.showPoints) return;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;

	uint32_t& numNodesDrawn = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	grid.sync();

	__shared__ int sh_nodeIndex;

	while(true){

		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(&numNodesDrawn, 1);
		}

		block.sync();

		if(sh_nodeIndex >= numNodes) break;

		Node* node = &nodes[sh_nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};

		drawNode(node->points, node, node->numPoints, framebuffer);
		drawNode(node->voxelChunks, node, node->numVoxels, framebuffer);

		block.sync();
	}

	grid.sync();
}

void drawNodesHQS(Node* nodes, uint32_t numNodes, uint64_t* framebuffer){

	if(!uniforms.showPoints) return;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;


	uint32_t& numNodesDrawn = *allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	int numPixels = int(uniforms.width) * int(uniforms.height);
	uint32_t* fb_depth = allocator->alloc<uint32_t*>(1 * numPixels * sizeof(uint32_t));
	uint32_t* fb_color = allocator->alloc<uint32_t*>(4 * numPixels * sizeof(uint32_t));

	processRange(numPixels, [&](int pixelID){
		fb_depth[pixelID] = 0x7f800000;
	});
	processRange(numPixels, [&](int pixelID){
		fb_color[4 * pixelID + 0] = 0;
		fb_color[4 * pixelID + 1] = 0;
		fb_color[4 * pixelID + 2] = 0;
		fb_color[4 * pixelID + 3] = 0;
	});

	grid.sync();

	__shared__ int sh_nodeIndex;

	// DEPTH
	while(true){

		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(&numNodesDrawn, 1);
		}

		block.sync();

		if(sh_nodeIndex >= numNodes) break;

		Node* node = &nodes[sh_nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};

		{ // draw points
			Chunk* chunk = node->points;
			int chunkIndex = 0;

			for(int pointIndex = block.thread_rank(); pointIndex < node->numPoints; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				float4 ndc = uniforms.transform * float4{point.x, point.y, point.z, 1.0f};
				float depth = ndc.w;
				ndc = ndc / ndc.w;

				int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
				int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{
					for(int ox = 0; ox < uniforms.pointSize; ox++)
					for(int oy = 0; oy < uniforms.pointSize; oy++)
					{
						uint32_t px = clamp(x + ox, 0, int(uniforms.width));
						uint32_t py = clamp(y + oy, 0, int(uniforms.height));

						uint32_t pixelID = px + int(uniforms.width) * py;
						uint32_t udepth = *((uint32_t*)&depth);

						if(udepth < fb_depth[pixelID]){
							atomicMin(&fb_depth[pixelID], udepth);
						}
					}
				}
			}
		}

		block.sync();

		uint32_t childVisibleMask = 0;
		for(int i = 0; i < 8; i++){
			Node* child = node->children[i];
			uint32_t childExists = (child != nullptr) ? 1 : 0;
			uint32_t childIsVisible = 0;

			if(childExists){
				childIsVisible = child->visible ? 1 : 0;
			}

			childVisibleMask = childVisibleMask | (childIsVisible << i);
		}

		if(node->numVoxels > 0)
		{// draw voxels
			Chunk* chunk = node->voxelChunks;
			int chunkIndex = 0;

			for(int pointIndex = block.thread_rank(); pointIndex < node->numVoxels; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				// cull voxels in areas where child nodes are visible
				float3 posuv = (float3{point.x, point.y, point.z} - nodeMin) / nodeSize;
				uint32_t cx = (posuv.x < 0.5f) ? 0 : 1;
				uint32_t cy = (posuv.y < 0.5f) ? 0 : 1;
				uint32_t cz = (posuv.z < 0.5f) ? 0 : 1;
				uint32_t childIndex = (cx << 2) | (cy << 1) | cz;

				bool childIsVisible = (childVisibleMask & (1 << childIndex)) != 0;

				if(visiblityMethod == VISIBLITY_ADDING){
					if(childIsVisible) continue;
				}
				// if(node->children[childIndex]){
				// 	if(node->children[childIndex]->visible) continue;
				// }

				float4 ndc = uniforms.transform * float4{point.x, point.y, point.z, 1.0f};
				float depth = ndc.w;
				ndc = ndc / ndc.w;

				int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
				int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{

					for(int ox = 0; ox < uniforms.pointSize; ox++)
					for(int oy = 0; oy < uniforms.pointSize; oy++)
					{
						uint32_t px = clamp(x + ox, 0, int(uniforms.width));
						uint32_t py = clamp(y + oy, 0, int(uniforms.height));

						uint32_t pixelID = px + int(uniforms.width) * py;
						uint64_t udepth = *((uint32_t*)&depth);

						if(udepth < fb_depth[pixelID]){
							atomicMin(&fb_depth[pixelID], udepth);
						}
					}

				}
			}
		}
	}

	grid.sync();

	if(grid.thread_rank() == 0){
		numNodesDrawn = 0;
	}

	grid.sync();

	auto inDebugZone = [&](int x, int y){
		if(x > uniforms.width / 2)  return true;
		else                        return false;
	};

	// COLOR
	while(true){

		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(&numNodesDrawn, 1);
		}

		block.sync();

		if(sh_nodeIndex >= numNodes) break;

		Node* node = &nodes[sh_nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};

		{ // draw points
			Chunk* chunk = node->points;
			int chunkIndex = 0;

			for(int pointIndex = block.thread_rank(); pointIndex < node->numPoints; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				// if(chunkIndex > (int(10 * uniforms.time) % 50)) break;

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				float4 ndc = uniforms.transform * float4{point.x, point.y, point.z, 1.0f};
				float depth = ndc.w;
				ndc = ndc / ndc.w;

				int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
				int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{
					uint32_t color = point.color;
					if(uniforms.colorByNode){
						color = (node->getID() % 127) * 123456789;
					}else if(uniforms.colorByLOD){
						color = getLodColor(node->level);
					}
					// if(RIGHTSIDE_NODECOLORS && x > uniforms.width / 2){
					// 	color = (node->getID() % 127) * 123456789;
					// }

					if(RIGHTSIDE_NODECOLORS && inDebugZone(x, y)){
						color = (node->getID() % 127) * 123456789;
					}

					// color = chunkIndex * 1234567;
					
					uint8_t* rgba = (uint8_t*)&color;

					for(int ox = 0; ox < uniforms.pointSize; ox++)
					for(int oy = 0; oy < uniforms.pointSize; oy++)
					{
						uint32_t px = clamp(x + ox, 0, int(uniforms.width));
						uint32_t py = clamp(y + oy, 0, int(uniforms.height));

						uint32_t pixelID = px + int(uniforms.width) * py;
						uint32_t udepth = *((uint32_t*)&depth);

						float fbDepth = *((float*)&fb_depth[pixelID]);

						if(depth < fbDepth * 1.01f)
						{
							atomicAdd(&fb_color[4 * pixelID + 0], rgba[0]);
							atomicAdd(&fb_color[4 * pixelID + 1], rgba[1]);
							atomicAdd(&fb_color[4 * pixelID + 2], rgba[2]);
							atomicAdd(&fb_color[4 * pixelID + 3], 1);
						}
					}
				}
			}
		}

		block.sync();

		uint32_t childVisibleMask = 0;
		for(int i = 0; i < 8; i++){
			Node* child = node->children[i];
			uint32_t childExists = (child != nullptr) ? 1 : 0;
			uint32_t childIsVisible = 0;

			if(childExists){
				childIsVisible = child->visible ? 1 : 0;
			}

			childVisibleMask = childVisibleMask | (childIsVisible << i);
		}

		if(node->numVoxels > 0)
		{// draw voxels
			Chunk* chunk = node->voxelChunks;
			int chunkIndex = 0;

			for(int pointIndex = block.thread_rank(); pointIndex < node->numVoxels; pointIndex += block.num_threads()){

				int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

				if(chunkIndex < targetChunkIndex){
					chunk = chunk->next;
					chunkIndex++;
				}

				// if(chunkIndex > (int(10 * uniforms.time) % 30)) break;

				int pointInChunkIndex = pointIndex % POINTS_PER_CHUNK;
				Point point = chunk->points[pointInChunkIndex];

				// cull voxels in areas where child nodes are visible
				float3 posuv = (float3{point.x, point.y, point.z} - nodeMin) / nodeSize;
				uint32_t cx = (posuv.x < 0.5f) ? 0 : 1;
				uint32_t cy = (posuv.y < 0.5f) ? 0 : 1;
				uint32_t cz = (posuv.z < 0.5f) ? 0 : 1;
				uint32_t childIndex = (cx << 2) | (cy << 1) | cz;
				// if(node->children[childIndex]){
				// 	if(node->children[childIndex]->visible) continue;
				// }

				bool childIsVisible = (childVisibleMask & (1 << childIndex)) != 0;

				if(visiblityMethod == VISIBLITY_ADDING){
					if(childIsVisible) continue;
				}

				float4 ndc = uniforms.transform * float4{point.x, point.y, point.z, 1.0f};
				float depth = ndc.w;
				ndc = ndc / ndc.w;

				int x = (ndc.x * 0.5 + 0.5) * uniforms.width;
				int y = (ndc.y * 0.5 + 0.5) * uniforms.height;

				if(x > 1 && x < uniforms.width  - 2.0)
				if(y > 1 && y < uniforms.height - 2.0)
				if(depth > 0.0f)
				{

					uint32_t color = point.color;
					if(uniforms.colorByNode){
						color = (node->getID() % 127) * 123456789;
					}else if(uniforms.colorByLOD){
						color = getLodColor(node->level);
					}
					// if(RIGHTSIDE_NODECOLORS && x > uniforms.width / 2){
					// 	color = (node->getID() % 127) * 123456789;
					// }

					if(RIGHTSIDE_NODECOLORS && inDebugZone(x, y)){
						color = (node->getID() % 127) * 123456789;
					}

					// color = chunkIndex * 1234567;

					uint8_t* rgba = (uint8_t*)&color;

					for(int ox = 0; ox < uniforms.pointSize; ox++)
					for(int oy = 0; oy < uniforms.pointSize; oy++)
					{
						uint32_t px = clamp(x + ox, 0, int(uniforms.width));
						uint32_t py = clamp(y + oy, 0, int(uniforms.height));

						uint32_t pixelID = px + int(uniforms.width) * py;
						uint64_t udepth = *((uint32_t*)&depth);

						float fbDepth = *((float*)&fb_depth[pixelID]);
						
						if(depth < fbDepth * 1.01f)
						{
							atomicAdd(&fb_color[4 * pixelID + 0], rgba[0]);
							atomicAdd(&fb_color[4 * pixelID + 1], rgba[1]);
							atomicAdd(&fb_color[4 * pixelID + 2], rgba[2]);
							atomicAdd(&fb_color[4 * pixelID + 3], 1);
						}
					}

				}
			}
		}
	}

	grid.sync();

	// RESOLVE
	processRange(numPixels, [&](int pixelID){
		
		uint32_t R = fb_color[4 * pixelID + 0];
		uint32_t G = fb_color[4 * pixelID + 1];
		uint32_t B = fb_color[4 * pixelID + 2];
		uint32_t C = fb_color[4 * pixelID + 3];

		if(C == 0) return;

		float depth = *((float*)&fb_depth[pixelID]);

		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;
		
		rgba[0] = R / C;
		rgba[1] = G / C;
		rgba[2] = B / C;
		rgba[3] = 255;

		uint64_t udepth = *((uint32_t*)&depth);
		uint64_t encoded = (udepth << 32) | color;
		

		framebuffer[pixelID] = encoded;

	});

	grid.sync();
}

void drawNodesBoundingBoxes(
	Node* nodes, uint32_t numNodes, 
	uint64_t* framebuffer,
	float3 cubeMin, float cubeSize,
	Lines* lines
){
	processRange(numNodes, [&](int index){
		Node* node = &nodes[index];

		if(node->numPoints == 0 && node->numVoxels == 0) return;

		float scale = cubeSize / pow(2.0f, float(node->level));

		float3 pos = {
			cubeMin.x + float(node->X + 0.5f) * scale,
			cubeMin.y + float(node->Y + 0.5f) * scale,
			cubeMin.z + float(node->Z + 0.5f) * scale,
		};

		uint32_t color = 0;
		if(node->isLeafFn()){
			color = 0x0000ff00;
		}else{
			color = 0x0000ffff;
		}

		// color = 0x00ffffff;
		// color = 0x00000000;
		color = 0x0000ffff;
		// color = SPECTRAL[(4 - node->level) % 8];
		if(node->level == 2){
			color = SPECTRAL[6];
		} else if(node->level == 3){
			color = SPECTRAL[4];
		}else if(node->level == 4){
			color = SPECTRAL[0];
		}else if(node->level == 5){
			color = SPECTRAL[6];
		}

		color = 0x0000ff00;

		drawBoundingBox(lines, pos, {scale, scale, scale}, color);
		int steps = 1;
		for(int si = 0; si < steps; si++){
			float s = 1.0 + 0.02 * float(si) / 5.0f;
			drawBoundingBox(lines, pos, float3{s * scale, scale, scale}, color);
			drawBoundingBox(lines, pos, float3{scale, s * scale, scale}, color);
			drawBoundingBox(lines, pos, float3{scale, s * scale, s * scale}, color);
		}
	});
}

void compute_visibility_disjunct(
	uint32_t& numVisibleNodes,
	uint32_t& numVisiblePoints,
	uint32_t& numVisibleVoxels,
	uint32_t& numVisibleInner,
	uint32_t& numVisibleLeaves,
	uint32_t& dbg_numSmallNodes,
	uint32_t& dbg_numSmallNodePoints,
	Stats* stats, Node* nodes, Node* visibleNodes,
	float cubeSize, float3 cubeMin
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

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

	float fwidth = uniforms.width;
	float fheight = uniforms.height;

	grid.sync();
	if(grid.thread_rank() == 0){
		numVisibleNodes  = 0;
		numVisiblePoints = 0;
		numVisibleVoxels = 0;
		numVisibleInner  = 0;
		numVisibleLeaves = 0;
		dbg_numSmallNodes = 0;
		dbg_numSmallNodePoints = 0;
	}
	grid.sync();

	auto makeVisible = [&](Node* node){
		uint32_t visibleNodeIndex = atomicAdd(&numVisibleNodes, 1);
		visibleNodes[visibleNodeIndex] = *node;

		if(node->numPoints > 0){
			atomicAdd(&numVisibleLeaves, 1);
			atomicAdd(&numVisiblePoints, node->numPoints);
		}else if(node->numVoxels > 0){
			atomicAdd(&numVisibleInner, 1);
			atomicAdd(&numVisibleVoxels, node->numVoxels);
		}

		// const char* name = (const char*)node->name;
		// printf("node: %s \n", name);
	};

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * nodeSize,
			cubeMin.y + float(node->Y + 0.0f) * nodeSize,
			cubeMin.z + float(node->Z + 0.0f) * nodeSize,
		};
		float3 nodeMax = {
			cubeMin.x + float(node->X + 1.0f) * nodeSize,
			cubeMin.y + float(node->Y + 1.0f) * nodeSize,
			cubeMin.z + float(node->Z + 1.0f) * nodeSize,
		};
		float3 nodeCenter = {
			cubeMin.x + float(node->X + 0.5f) * nodeSize,
			cubeMin.y + float(node->Y + 0.5f) * nodeSize,
			cubeMin.z + float(node->Z + 0.5f) * nodeSize,
		};

		// compute node boundaries in screen space
		float4 p000 = {nodeMin.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p001 = {nodeMin.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p010 = {nodeMin.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p011 = {nodeMin.x, nodeMax.y, nodeMax.z, 1.0f};
		float4 p100 = {nodeMax.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p101 = {nodeMax.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p110 = {nodeMax.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p111 = {nodeMax.x, nodeMax.y, nodeMax.z, 1.0f};

		float4 ndc000 = uniforms.transform_updateBound * p000;
		float4 ndc001 = uniforms.transform_updateBound * p001;
		float4 ndc010 = uniforms.transform_updateBound * p010;
		float4 ndc011 = uniforms.transform_updateBound * p011;
		float4 ndc100 = uniforms.transform_updateBound * p100;
		float4 ndc101 = uniforms.transform_updateBound * p101;
		float4 ndc110 = uniforms.transform_updateBound * p110;
		float4 ndc111 = uniforms.transform_updateBound * p111;

		float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};

		float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		// screen-space size
		float dx = smax_x - smin_x;
		float dy = smax_y - smin_y;

		// {
		// 	float sx_x = abs(s100.x - s000.x);
		// 	float sx_y = abs(s100.y - s000.y);
		// 	float sy_x = abs(s010.x - s000.x);
		// 	float sy_y = abs(s010.y - s000.y);
		// 	float sz_x = abs(s001.x - s000.x);
		// 	float sz_y = abs(s001.y - s000.y);

		// 	smax_x = max(max(sx_x, sy_x), sz_x);
		// 	smax_y = max(max(sx_y, sy_y), sz_y);

		// 	dx = 1.0 * smax_x;
		// 	dy = 1.0 * smax_y;
		// }

		// {
		// 	float4 proj = uniforms.transform_updateBound * float4{nodeCenter.x, nodeCenter.y, nodeCenter.z, 1.0};
		// 	float d = proj.w;
		// 	float r = nodeSize;
		// 	float aspect = fwidth / fheight;
		// 	float pr = 1.0 / tan(uniforms.fovy_rad) * r / sqrt(d * d - r * r);
		// 	float pixels = pr * fheight * 2.0;
		// 	float projSize = aspect * fheight * nodeSize / proj.w;
		// 	if(proj.w > 0.0){
		// 		dx = projSize;
		// 		dy = projSize;

		// 		dx = pixels;
		// 		dy = pixels;
		// 	}
		// }

		bool isIntersectingFrustum = intersectsFrustum(uniforms.transform_updateBound, nodeMin, nodeMax);
		bool hasSamples = node->numPoints > 0 || node->numVoxels > 0;

		bool visible = true;

		if(!isIntersectingFrustum) visible = false;
		if(!hasSamples) visible = false;

		node->visible = visible;
		node->isLarge = dx > 2.0 * uniforms.minNodeSize || dy > 2.0 * uniforms.minNodeSize;

		// if(strequal(node->name, "r")){

		// 	bool isLarge = false;
		// 	{
		// 		float sx_x = abs(s100.x - s000.x);
		// 		float sx_y = abs(s100.y - s000.y);
		// 		float sy_x = abs(s010.x - s000.x);
		// 		float sy_y = abs(s010.y - s000.y);
		// 		float sz_x = abs(s001.x - s000.x);
		// 		float sz_y = abs(s001.y - s000.y);

		// 		smax_x = max(max(sx_x, sy_x), sz_x);
		// 		smax_y = max(max(sx_y, sy_y), sz_y);

		// 		dx = 1.0 * smax_x;
		// 		dy = 1.0 * smax_y;

		// 		isLarge = dx > 2.0 * uniforms.minNodeSize || dy > 2.0 * uniforms.minNodeSize;
		// 	}

		// 	uint32_t color = 0x0000ffff;
		// 	if(isLarge) color = 0x00ffffff;

		// 	drawLine(_lines, make_float3(p000), make_float3(p100), color);
		// 	drawLine(_lines, make_float3(p000), make_float3(p010), color);
		// 	drawLine(_lines, make_float3(p000), make_float3(p001), color);

		// 	drawLine(_lines, make_float3(p111), make_float3(p011), 0xffffffff - color);
		// 	drawLine(_lines, make_float3(p111), make_float3(p101), 0xffffffff - color);
		// 	drawLine(_lines, make_float3(p111), make_float3(p110), 0xffffffff - color);

		// 	makeVisible(node);


		// }else{
		// 	node->visible = false;
		// }

	});

	grid.sync();

	// if(false)
	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		// Banyunibo 7 & 10
		// if(strequal(node->name, "r06226")){
		// if(strequal(node->name, "r062263")){
		// 	makeVisible(node);
		// 	printf("#samples: %i \n", node->numPoints);
		// }
		// return;


		if(node->isLarge && !node->isLeafFn()){

			for(int i = 0; i < 8; i++){
				Node* child = node->children[i];

				if(child == nullptr) continue;
				if(child->isLarge) continue;
				if(!child->visible) continue;

				makeVisible(child);
				// printf("test \n");
			}
		}else if(node->isLarge && node->isLeafFn() && node->visible){
			makeVisible(node);
		}
	});
}

void compute_visibility_add(
	uint32_t& numVisibleNodes,
	uint32_t& numVisiblePoints,
	uint32_t& numVisibleVoxels,
	uint32_t& numVisibleInner,
	uint32_t& numVisibleLeaves,
	uint32_t& dbg_numSmallNodes,
	uint32_t& dbg_numSmallNodePoints,
	Stats* stats, Node* nodes, Node* visibleNodes,
	float cubeSize, float3 cubeMin
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

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

	float fwidth = uniforms.width;
	float fheight = uniforms.height;

	grid.sync();
	if(grid.thread_rank() == 0){
		numVisibleNodes  = 0;
		numVisiblePoints = 0;
		numVisibleVoxels = 0;
		numVisibleInner  = 0;
		numVisibleLeaves = 0;
		dbg_numSmallNodes = 0;
		dbg_numSmallNodePoints = 0;
	}
	grid.sync();

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		float nodeSize = cubeSize / pow(2.0f, float(node->level));
		// node-center
		float ncx = (float(node->X) + 0.5f) * nodeSize + cubeMin.x;
		float ncy = (float(node->Y) + 0.5f) * nodeSize + cubeMin.y;
		float ncz = (float(node->Z) + 0.5f) * nodeSize + cubeMin.z;

		float scale = cubeSize / pow(2.0f, float(node->level));
		float3 nodeMin = {
			cubeMin.x + float(node->X + 0.0f) * scale,
			cubeMin.y + float(node->Y + 0.0f) * scale,
			cubeMin.z + float(node->Z + 0.0f) * scale,
		};
		float3 nodeMax = {
			cubeMin.x + float(node->X + 1.0f) * scale,
			cubeMin.y + float(node->Y + 1.0f) * scale,
			cubeMin.z + float(node->Z + 1.0f) * scale,
		};

		bool visible = true;

		// compute node boundaries in screen space
		float4 p000 = {nodeMin.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p001 = {nodeMin.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p010 = {nodeMin.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p011 = {nodeMin.x, nodeMax.y, nodeMax.z, 1.0f};
		float4 p100 = {nodeMax.x, nodeMin.y, nodeMin.z, 1.0f};
		float4 p101 = {nodeMax.x, nodeMin.y, nodeMax.z, 1.0f};
		float4 p110 = {nodeMax.x, nodeMax.y, nodeMin.z, 1.0f};
		float4 p111 = {nodeMax.x, nodeMax.y, nodeMax.z, 1.0f};

		float4 ndc000 = uniforms.transform_updateBound * p000;
		float4 ndc001 = uniforms.transform_updateBound * p001;
		float4 ndc010 = uniforms.transform_updateBound * p010;
		float4 ndc011 = uniforms.transform_updateBound * p011;
		float4 ndc100 = uniforms.transform_updateBound * p100;
		float4 ndc101 = uniforms.transform_updateBound * p101;
		float4 ndc110 = uniforms.transform_updateBound * p110;
		float4 ndc111 = uniforms.transform_updateBound * p111;

		float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};

		float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		// screen-space size
		float dx = smax_x - smin_x;
		float dy = smax_y - smin_y;

		if(!intersectsFrustum(uniforms.transform_updateBound, nodeMin, nodeMax)){
			visible = false;
		}

		if(dx < uniforms.minNodeSize || dy < uniforms.minNodeSize){
			visible = false;
		}

		if(node->numPoints == 0 && node->numVoxels == 0){
			visible = false;
		}

		node->visible = visible;
		
		if(visible){
			uint32_t visibleNodeIndex = atomicAdd(&numVisibleNodes, 1);
			visibleNodes[visibleNodeIndex] = nodes[nodeIndex];

			if(node->numPoints > 0){
				atomicAdd(&numVisibleLeaves, 1);
				atomicAdd(&numVisiblePoints, node->numPoints);
			}else if(node->numVoxels > 0){
				atomicAdd(&numVisibleInner, 1);
				atomicAdd(&numVisibleVoxels, node->numVoxels);
			}
		}
	});

	grid.sync();
}

extern "C" __global__
void kernel_render(
	uint32_t* buffer,
	const Uniforms _uniforms,
	Node* nodes,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats,
	uint64_t* frameStartTimestamp,
	CudaPrint* cudaprint
){

	auto tStart = nanotime();

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0){
		*frameStartTimestamp = nanotime();
	}

	uniforms = _uniforms;
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

	Node* visibleNodes                 = allocator->alloc<Node*>(100'000 * sizeof(Node));
	uint32_t& numVisibleNodes          = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisiblePoints         = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleVoxels         = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleInner          = *allocator->alloc<uint32_t*>(4);
	uint32_t& numVisibleLeaves         = *allocator->alloc<uint32_t*>(4);
	uint32_t& dbg_numSmallNodes        = *allocator->alloc<uint32_t*>(4);
	uint32_t& dbg_numSmallNodePoints   = *allocator->alloc<uint32_t*>(4);

	Lines* lines = allocator->alloc<Lines*>(sizeof(Lines));
	lines->count = 0;
	lines->vertices = allocator->alloc<Point*>(1'000'000 * sizeof(Point));
	_lines = lines;

	int framebufferSize = int(uniforms.width) * int(uniforms.height) * sizeof(uint64_t);
	uint64_t* framebuffer = allocator->alloc<uint64_t*>(framebufferSize);

	// clear framebuffer
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){
		// depth:            7f800000 (Infinity)
		// background color: 00332211 (aabbggrr)
		// framebuffer[pixelIndex] = 0x7f800000'00332211ull;
		framebuffer[pixelIndex] = (0x7f800000ull << 32) | uint64_t(BACKGROUND_COLOR);
	});

	grid.sync();

	float3 boxSize = uniforms.boxMax - uniforms.boxMin;
	float cubeSize = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 cubeMin = uniforms.boxMin;
	float3 cubeMax = cubeMin + cubeSize;
	float3 cubePosition = uniforms.boxMin + cubeSize * 0.5f;

	grid.sync();
	
	// if(uniforms.doUpdateVisibility)
	{
		if(visiblityMethod == VISIBLITY_DISJUNCT){
			compute_visibility_disjunct(
				numVisibleNodes,
				numVisiblePoints,
				numVisibleVoxels,
				numVisibleInner,
				numVisibleLeaves,
				dbg_numSmallNodes,
				dbg_numSmallNodePoints,
				stats, nodes, visibleNodes,
				cubeSize, cubeMin
			);
		}else if(visiblityMethod == VISIBLITY_ADDING){
			compute_visibility_add(
				numVisibleNodes,
				numVisiblePoints,
				numVisibleVoxels,
				numVisibleInner,
				numVisibleLeaves,
				dbg_numSmallNodes,
				dbg_numSmallNodePoints,
				stats, nodes, visibleNodes,
				cubeSize, cubeMin
			);
		}
	}

	// draw all nodes
	// visibleNodes = nodes;
	// numVisibleNodes = numNodes;

	grid.sync();

	{ // draw nodes, boxes, lines, ...
		if(uniforms.useHighQualityShading){
			drawNodesHQS(visibleNodes, numVisibleNodes, framebuffer);
		}else{
			drawNodes(visibleNodes, numVisibleNodes, framebuffer);
		}

		grid.sync();

		// draw point cloud cube
		// if(uniforms.showBoundingBox)
		// if(grid.thread_rank() == 0)
		// {
		// 	float3 size = {cubeSize, cubeSize, cubeSize};
		// 	drawBoundingBox(lines, cubePosition, size, 0x000000ff);
		// }

		grid.sync();

		if(uniforms.showBoundingBox)
		if(grid.thread_rank() == 0)
		{// draw frustum

			auto project = [](float3 coordinate){
				
				float4 pos = uniforms.transformInv_updateBound * float4{coordinate.x, coordinate.y, coordinate.z, 1.0};

				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;
				pos.z = pos.z / pos.w;

				return make_float3(pos);
			};

			float fend = 0.99995;

			drawLine(lines, project({ 1.0,  1.0, -1.0}), project({ 1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({ 1.0, -1.0, -1.0}), project({ 1.0, -1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0,  1.0, -1.0}), project({-1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0, -1.0, -1.0}), project({-1.0, -1.0, fend}), 0x000000ff);

			drawLine(lines, project({-1.0, -1.0, fend}), project({ 1.0, -1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0,  1.0, fend}), project({ 1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({-1.0, -1.0, fend}), project({-1.0,  1.0, fend}), 0x000000ff);
			drawLine(lines, project({ 1.0, -1.0, fend}), project({ 1.0,  1.0, fend}), 0x000000ff);
		}

		grid.sync();

		if(uniforms.showBoundingBox){ 
			drawNodesBoundingBoxes(visibleNodes, numVisibleNodes, framebuffer, cubeMin, cubeSize, lines);
		}

		grid.sync();

		rasterizeLines(lines, framebuffer, uniforms.width, uniforms.height, uniforms.transform);

		grid.sync();
	}

	// if(grid.thread_rank() == 0){
	// 	double duration_ms = double(nanotime() - tStart) / 1'000'000.0;
	// 	printf("ms: %f \n", duration_ms);
	// }

	// UPDATE STATS
	if(grid.thread_rank() == 0){
		// printf("update stats! \n");
		stats->numVisibleNodes  = numVisibleNodes;
		stats->numVisibleInner  = numVisibleInner;
		stats->numVisibleLeaves = numVisibleLeaves;
		stats->numVisiblePoints = numVisiblePoints;
		stats->numVisibleVoxels = numVisibleVoxels;
		stats->frameID          = uniforms.frameCounter;
	}

	// if(false)
	{ // tile-based EDL

		struct Pixel{
			uint32_t color;
			float depth;
		};

		float* framebuffer_f = (float*)framebuffer;
		uint32_t* framebuffer_u32 = (uint32_t*)framebuffer;
		Pixel* framebuffer_p = (Pixel*)framebuffer;

		int width = uniforms.width;
		int height = uniforms.height;

		uint32_t tileSize = 16;
		uint32_t numTiles_x = width / tileSize;
		uint32_t numTiles_y = height / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;
		uint32_t tilesPerBlock = numTiles / grid.num_blocks();

		for(int i = 0; i < tilesPerBlock; i++){

			block.sync();

			int tileID = i * grid.num_blocks() + block.group_index().x;
			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			int tileStart = tileX * tileSize + tileY * width * tileSize;

			int pixelID = tileStart + block.thread_rank() % tileSize + (block.thread_rank() / tileSize) * width;

			Pixel& pixel = framebuffer_p[pixelID];

			float PI = 3.1415;
			float numSamples = 50;
			float stepsize = 2.0 * PI / numSamples;
			float r = 1.5;
			float edlStrength = 0.4;

			float sum = 0.0;
			// for(float u = 0.0; u < 2.0 * PI; u += stepsize)
			for(float u : {0.0f, PI / 2.0f, PI, 3.0f * PI / 2.0f})
			{
				// r = u;
				int dx = r * sin(u);
				int dy = r * cos(u);

				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height);

				Pixel neighbour = framebuffer_p[index];
				
				sum = sum + max(log2(pixel.depth) - log2(neighbour.depth), 0.0);
			}
			
			float response = sum / numSamples;
			float shade = __expf(-response * 300.0 * edlStrength);
			// shade = 1.0;

			uint32_t R = shade * ((pixel.color >>  0) & 0xff);
			uint32_t G = shade * ((pixel.color >>  8) & 0xff);
			uint32_t B = shade * ((pixel.color >> 16) & 0xff);
			uint32_t color = R | (G << 8) | (B << 16) | (255u << 24);

			pixel.color = color;

			block.sync();
		}
		grid.sync();
	}

	grid.sync();

	// auto tEnd = nanotime();
	// double millies = double(tEnd - tStart) / 1'000'000.0;
	// PRINT("duration: %.1f ms \n", millies);

	// transfer framebuffer to opengl texture
	processRange(0, uniforms.width * uniforms.height, [&](int pixelIndex){

		int x = pixelIndex % int(uniforms.width);
		int y = pixelIndex / int(uniforms.width);

		uint64_t encoded = framebuffer[pixelIndex];
		uint32_t color = encoded & 0xffffffffull;

		surf2Dwrite(color, gl_colorbuffer, x * 4, y);
	});

	// if(grid.thread_rank() == 0){
	// 	printf("allocator.offset: %llu \n", allocator->offset / 1'000'000);
	// }


	// if(grid.thread_rank() == 0){
	// 	auto tEnd = nanotime();
	// 	float ms = double(tEnd - tStart) / 1'000'000.0;
	// 	printf("%f ms\n", ms);
	// }
}

