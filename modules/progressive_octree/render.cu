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
Lines* _lines = nullptr;

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

extern "C" __global__
void kernel_render(
	uint32_t* buffer,
	const Uniforms _uniforms,
	cudaSurfaceObject_t gl_colorbuffer,
	Stats* stats
){

	auto tStart = nanotime();

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();


	uniforms = _uniforms;
	Allocator _allocator(buffer, 0);
	allocator = &_allocator;

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

	if(grid.thread_rank() == 0){
		drawBoundingBox(lines, float3{0.0f, 0.0f, 0.0f}, float3{2.0f, 2.0f, 2.0f}, 0x000000ff);
	}

	grid.sync();

	rasterizeLines(lines, framebuffer, uniforms.width, uniforms.height, uniforms.transform);

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

