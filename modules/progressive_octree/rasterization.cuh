#pragma once

#include "structures.cuh"

void drawLine(Lines* lines, float3 start, float3 end, uint32_t color){

	Point pStart;
	pStart.x = start.x;
	pStart.y = start.y;
	pStart.z = start.z;
	pStart.color = color;

	Point pEnd;
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

void rasterizePoints(Point* points, uint32_t numPoints, uint64_t* target, float width, float height, mat4 transform){

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

		Point start = lines->vertices[2 * lineIndex + 0];
		Point end = lines->vertices[2 * lineIndex + 1];

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

			if(RIGHTSIDE_BOXES){
				if(x < width / 2) continue;
			}

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