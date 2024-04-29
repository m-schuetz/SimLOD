#pragma once

#include "structures.cuh"

void drawLine(Lines* lines, float3 start, float3 end, uint32_t color){

	Vertex pStart;
	pStart.x = start.x;
	pStart.y = start.y;
	pStart.z = start.z;
	pStart.color = color;

	Vertex pEnd;
	pEnd.x = end.x;
	pEnd.y = end.y;
	pEnd.z = end.z;
	pEnd.color = color;

	int index = atomicAdd(&lines->count, 2);
	lines->vertices[index + 0] = pStart;
	lines->vertices[index + 1] = pEnd;

}

void drawBoundingBox(Lines* lines, float3 pos, float3 size, uint32_t color){

	float3 min = pos - size / 2.0;
	float3 max = pos + size / 2.0;

	// BOTTOM
	drawLine(lines, float3{min.x, min.y, min.z}, float3{max.x, min.y, min.z}, color);
	drawLine(lines, float3{max.x, min.y, min.z}, float3{max.x, max.y, min.z}, color);
	drawLine(lines, float3{max.x, max.y, min.z}, float3{min.x, max.y, min.z}, color);
	drawLine(lines, float3{min.x, max.y, min.z}, float3{min.x, min.y, min.z}, color);

	// TOP
	drawLine(lines, float3{min.x, min.y, max.z}, float3{max.x, min.y, max.z}, color);
	drawLine(lines, float3{max.x, min.y, max.z}, float3{max.x, max.y, max.z}, color);
	drawLine(lines, float3{max.x, max.y, max.z}, float3{min.x, max.y, max.z}, color);
	drawLine(lines, float3{min.x, max.y, max.z}, float3{min.x, min.y, max.z}, color);

	// BOTTOM TO TOP
	drawLine(lines, float3{max.x, min.y, min.z}, float3{max.x, min.y, max.z}, color);
	drawLine(lines, float3{max.x, max.y, min.z}, float3{max.x, max.y, max.z}, color);
	drawLine(lines, float3{min.x, max.y, min.z}, float3{min.x, max.y, max.z}, color);
	drawLine(lines, float3{min.x, min.y, min.z}, float3{min.x, min.y, max.z}, color);
}

void drawBox(Triangles* triangles, float3 pos, float3 size, uint32_t color){

	uint32_t index = atomicAdd(&triangles->numTriangles, 12);

	float sx = size.x;
	float sy = size.y;
	float sz = size.z;

	// TOP
	triangles->positions[3 * index + 0 +  0] = float3{-sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 1 +  0] = float3{ sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 2 +  0] = float3{ sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 3 +  0] = float3{-sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 4 +  0] = float3{ sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 5 +  0] = float3{-sx,  sy,  sz} + pos;

	// BOTTOM
	triangles->positions[3 * index + 0 +  6] = float3{-sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 1 +  6] = float3{ sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 2 +  6] = float3{ sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 3 +  6] = float3{ sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 4 +  6] = float3{-sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 5 +  6] = float3{-sx,  sy, -sz} + pos;

	// SIDE 1
	triangles->positions[3 * index + 0 + 12] = float3{-sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 1 + 12] = float3{-sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 2 + 12] = float3{-sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 3 + 12] = float3{-sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 4 + 12] = float3{-sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 5 + 12] = float3{-sx, -sy,  sz} + pos;

	// SIDE 2
	triangles->positions[3 * index + 0 + 18] = float3{ sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 1 + 18] = float3{ sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 2 + 18] = float3{ sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 3 + 18] = float3{ sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 4 + 18] = float3{ sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 5 + 18] = float3{ sx, -sy, -sz} + pos;

	// SIDE 3
	triangles->positions[3 * index + 0 + 24] = float3{-sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 1 + 24] = float3{ sx, -sy, -sz} + pos;
	triangles->positions[3 * index + 2 + 24] = float3{ sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 3 + 24] = float3{ sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 4 + 24] = float3{-sx, -sy,  sz} + pos;
	triangles->positions[3 * index + 5 + 24] = float3{-sx, -sy, -sz} + pos;

	// SIDE 3
	triangles->positions[3 * index + 0 + 30] = float3{-sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 1 + 30] = float3{ sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 2 + 30] = float3{ sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 3 + 30] = float3{ sx,  sy,  sz} + pos;
	triangles->positions[3 * index + 4 + 30] = float3{-sx,  sy, -sz} + pos;
	triangles->positions[3 * index + 5 + 30] = float3{-sx,  sy,  sz} + pos;

	for(int i = 0; i < 36; i++){
		triangles->colors[3 * index + i] = color;
	}
}


void rasterizePoints(
	Point* points, uint32_t numPoints, 
	uint64_t* target, 
	float width, float height, 
	mat4 transform
){

	processRange(numPoints, [&](int index){

		Point point = points[index];

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
			// 	uint64_t encoded = (udepth << 32) | color;

			// 	atomicMin(&framebuffer[pixelID], encoded);
			// }
		}

	});

}

void rasterizeLines(Lines* lines, uint64_t* target, int width, int height, mat4 transform){

	auto grid = cg::this_grid();
	grid.sync();
 
	Frustum frustum = Frustum::fromWorldViewProj(transform);
	
	int numLines = lines->count / 2;
	processRange(0, numLines, [&](int lineIndex){

		Vertex start = lines->vertices[2 * lineIndex + 0];
		Vertex end = lines->vertices[2 * lineIndex + 1];

		float3 dir = float3{
			end.x - start.x,
			end.y - start.y,
			end.z - start.z
		};
		dir = normalize(dir);

		if(!frustum.contains({start.x, start.y, start.z})){
			float3 I = frustum.intersectRay({start.x, start.y, start.z}, dir);

			start.x = I.x;
			start.y = I.y;
			start.z = I.z;
		}

		if(!frustum.contains({end.x, end.y, end.z})){
			float3 I = frustum.intersectRay({end.x, end.y, end.z}, dir * -1.0f);

			end.x = I.x;
			end.y = I.y;
			end.z = I.z;
		}

		float4 ndc_start = transform * float4{start.x, start.y, start.z, 1.0f};
		ndc_start.x = ndc_start.x / ndc_start.w;
		ndc_start.y = ndc_start.y / ndc_start.w;
		ndc_start.z = ndc_start.z / ndc_start.w;

		float4 ndc_end = transform * float4{end.x, end.y, end.z, 1.0f};
		ndc_end.x = ndc_end.x / ndc_end.w;
		ndc_end.y = ndc_end.y / ndc_end.w;
		ndc_end.z = ndc_end.z / ndc_end.w;

		float3 screen_start = {
			(ndc_start.x * 0.5f + 0.5f) * width,
			(ndc_start.y * 0.5f + 0.5f) * height,
			1.0f
		};
		float3 screen_end = {
			(ndc_end.x * 0.5f + 0.5f) * width,
			(ndc_end.y * 0.5f + 0.5f) * height,
			1.0f
		};

		float steps = length(screen_end - screen_start);
		// prevent long lines, to be safe
		steps = clamp(steps, 0.0f, 400.0f); 
		float stepSize = 1.0 / steps;

		float start_depth_linear = ndc_start.w;
		float end_depth_linear = ndc_end.w;

		for(float u = 0; u <= 1.0; u += stepSize){
			float ndc_x = (1.0 - u) * ndc_start.x + u * ndc_end.x;
			float ndc_y = (1.0 - u) * ndc_start.y + u * ndc_end.y;
			float depth = (1.0 - u) * start_depth_linear + u * end_depth_linear;

			if(ndc_x < -1.0 || ndc_x > 1.0) continue;
			if(ndc_y < -1.0 || ndc_y > 1.0) continue;

			int x = (ndc_x * 0.5 + 0.5) * width;
			int y = (ndc_y * 0.5 + 0.5) * height;

			x = clamp(x, 0, width - 1);
			y = clamp(y, 0, height - 1);

			int pixelID = x + width * y;

			uint64_t idepth = *((uint32_t*)&depth);
			uint64_t encoded = (idepth << 32) | start.color;

			atomicMin(&target[pixelID], encoded);
		}
	});

	grid.sync();
}

void rasterizePlane(uint64_t* target, int width, int height, mat4 transform){

	int cells = 2000;
	processRange(0, cells * cells, [&](int index){
		int ux = index % cells;
		int uy = index / cells;

		float u = float(ux) / float(cells - 1);
		float v = float(uy) / float(cells - 1);

		float4 pos = {
			20.0 * (u - 0.5), 
			20.0 * (v - 0.5), 
			0.0f, 
			1.0f};

		float4 ndc = transform * pos;
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;
		ndc.z = ndc.z / ndc.w;
		float depth = ndc.w;

		int x = (ndc.x * 0.5 + 0.5) * width;
		int y = (ndc.y * 0.5 + 0.5) * height;

		uint32_t R = 255.0f * u;
		uint32_t G = 255.0f * v;
		uint32_t B = 0;
		uint64_t color = R | (G << 8) | (B << 16);

		if(x > 1 && x < width  - 2.0)
		if(y > 1 && y < height - 2.0){

			// SINGLE PIXEL
			uint32_t pixelID = x + int(width) * y;
			uint64_t udepth = *((uint32_t*)&depth);
			uint64_t encoded = (udepth << 32) | color;

			atomicMin(&target[pixelID], encoded);

			// POINT SPRITE
			// for(int ox : {-2, -1, 0, 1, 2})
			// for(int oy : {-2, -1, 0, 1, 2}){
			// 	uint32_t pixelID = (x + ox) + int(uniforms.width) * (y + oy);
			// 	uint64_t udepth = *((uint32_t*)&depth);
			// 	uint64_t encoded = (udepth << 32) | color;

			// 	atomicMin(&framebuffer[pixelID], encoded);
			// }
		}
	});

}

void rasterizeTriangles(
	Triangles* triangles, 
	uint32_t* processedTriangles, 
	uint64_t* framebuffer, 
	Uniforms& uniforms
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// return;
	
	// uint32_t* processedTriangles = allocator->alloc<uint32_t*>(4);
	if(grid.thread_rank() == 0){
		*processedTriangles = 0;
	}
	grid.sync();

	{
		__shared__ int sh_triangleIndex;

		block.sync();

		// safety mechanism: each block draws at most <loop_max> triangles
		int loop_max = 10'000;
		for(int loop_i = 0; loop_i < loop_max; loop_i++){
			
			// grab the index of the next unprocessed triangle
			block.sync();
			if(block.thread_rank() == 0){
				sh_triangleIndex = atomicAdd(processedTriangles, 1);
			}
			block.sync();

			if(sh_triangleIndex >= triangles->numTriangles) break;

			// project x/y to pixel coords
			// z: whatever 
			// w: linear depth
			auto toScreenCoord = [&](float3 p){
				float4 pos = uniforms.transform * float4{p.x, p.y, p.z, 1.0f};

				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * uniforms.width, 
					(pos.y * 0.5f + 0.5f) * uniforms.height,
					pos.z, 
					pos.w
				};

				return imgPos;
			};

			int i0 = 3 * sh_triangleIndex + 0;
			int i1 = 3 * sh_triangleIndex + 1;
			int i2 = 3 * sh_triangleIndex + 2;
			
			float3 v0 = triangles->positions[i0];
			float3 v1 = triangles->positions[i1];
			float3 v2 = triangles->positions[i2];

			float4 p0 = toScreenCoord(v0);
			float4 p1 = toScreenCoord(v1);
			float4 p2 = toScreenCoord(v2);

			// cull a triangle if one of its vertices is closer than depth 0
			if(p0.w < 0.0 || p1.w < 0.0 || p2.w < 0.0) continue;

			float2 v01 = {p1.x - p0.x, p1.y - p0.y};
			float2 v02 = {p2.x - p0.x, p2.y - p0.y};

			auto cross = [](float2 a, float2 b){ return a.x * b.y - a.y * b.x; };

			{// backface culling
				float w = cross(v01, v02);
				if(w < 0.0) continue;
			}

			// compute screen-space bounding rectangle
			float min_x = min(min(p0.x, p1.x), p2.x);
			float min_y = min(min(p0.y, p1.y), p2.y);
			float max_x = max(max(p0.x, p1.x), p2.x);
			float max_y = max(max(p0.y, p1.y), p2.y);

			// clamp to screen
			min_x = clamp(min_x, 0.0f, uniforms.width);
			min_y = clamp(min_y, 0.0f, uniforms.height);
			max_x = clamp(max_x, 0.0f, uniforms.width);
			max_y = clamp(max_y, 0.0f, uniforms.height);

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			// iterate through fragments in bounding rectangle and draw if within triangle
			int numProcessedSamples = 0;
			for(int fragOffset = 0; fragOffset < numFragments; fragOffset += block.num_threads()){

				// safety mechanism: don't draw more than <x> pixels per thread
				if(numProcessedSamples > 5'000) break;

				int fragID = fragOffset + block.thread_rank();
				int fragX = fragID % size_x;
				int fragY = fragID / size_x;

				float2 pFrag = {
					floor(min_x) + float(fragX), 
					floor(min_y) + float(fragY)
				};
				float2 sample = {pFrag.x - p0.x, pFrag.y - p0.y};

				// v: vertex[0], s: vertex[1], t: vertex[2]
				float s = cross(sample, v02) / cross(v01, v02);
				float t = cross(v01, sample) / cross(v01, v02);
				float v = 1.0 - (s + t);

				int2 pixelCoords = make_int2(pFrag.x, pFrag.y);
				int pixelID = pixelCoords.x + pixelCoords.y * uniforms.width;
				pixelID = clamp(pixelID, 0, int(uniforms.width * uniforms.height) - 1);

				if(s >= 0.0)
				if(t >= 0.0)
				if(s + t <= 1.0)
				{
					uint8_t* v0_rgba = (uint8_t*)&triangles->colors[i0];
					uint8_t* v1_rgba = (uint8_t*)&triangles->colors[i1];
					uint8_t* v2_rgba = (uint8_t*)&triangles->colors[i2];

					float2 v0_uv = triangles->uvs[i0];
					float2 v1_uv = triangles->uvs[i1];
					float2 v2_uv = triangles->uvs[i2];
					float2 uv = {
						v * v0_uv.x + s * v1_uv.x + t * v2_uv.x,
						v * v0_uv.y + s * v1_uv.y + t * v2_uv.y
					};

					uint32_t color;
					uint8_t* rgb = (uint8_t*)&color;

					// by triangle id
					color = sh_triangleIndex * 123456;

					// by vertex color
					rgb[0] = v * v0_rgba[0] + s * v1_rgba[0] + t * v2_rgba[0];
					rgb[1] = v * v0_rgba[1] + s * v1_rgba[1] + t * v2_rgba[1];
					rgb[2] = v * v0_rgba[2] + s * v1_rgba[2] + t * v2_rgba[2];

					float depth = v * p0.w + s * p1.w + t * p2.w;
					uint64_t udepth = *((uint32_t*)&depth);
					uint64_t pixel = (udepth << 32ull) | color;

					atomicMin(&framebuffer[pixelID], pixel);
				}

				numProcessedSamples++;
			}


		}
	}
}