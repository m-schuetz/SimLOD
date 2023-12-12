

#include <iostream>
#include <filesystem>
#include <locale.h>
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <deque>
#include <atomic>
#include <optional>
#ifdef __cpp_lib_format
#include <format>
#else
#include "fmt/core.h"
using namespace fmt;
#endif

#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "cudaGL.h"

#include "unsuck.hpp"
#include "laszip_api.h"

#include "HostDeviceInterface.h"
#include "SimlodLoader.h"
#include "LasLoader.h"

using namespace std;

vector<string> paths = {
	"NONE",
	// "d:/dev/pointclouds/riegl/retz.las",
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

CUdevice device;
CUcontext context;
int numSMs;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_renderbuffer;
CUdeviceptr cptr_stats;
CUdeviceptr cptr_frameStart;
CUgraphicsResource cugl_colorbuffer;
CUevent ce_render_start, ce_render_end;
CUevent ce_update_start, ce_update_end;
cudaStream_t stream_upload, stream_download;

CudaModularProgram* cuda_program = nullptr;

glm::mat4 transform;
glm::mat4 transform_updatebound;

Stats stats;
void* h_stats_pinned = nullptr;

struct {
	bool useHighQualityShading       = false;
	bool showBoundingBox             = false;
	bool doUpdateVisibility          = true;
	bool showPoints                  = true;
	bool colorByNode                 = false;
	bool colorByLOD                  = false;
	bool autoFocusOnLoad             = true;
	bool benchmarkRendering          = false;
	float LOD                        = 0.2f;
	float minNodeSize                = 64.0f;
	int pointSize                    = 1;
	float fovy                       = 60.0f;
} settings;

float renderingDuration            = 0.0f;
uint64_t momentaryBufferCapacity   = 0;
vector<double> processFrameTimes; 

float toggle = 1.0;
float lastFrameTime = static_cast<float>(now());
float timeSinceLastFrame = 0.0;

float3 boxMin  = float3{InfinityF, InfinityF, InfinityF};
float3 boxMax  = float3{-InfinityF, -InfinityF, -InfinityF};
float3 boxSize = float3{0.0, 0.0, 0.0};

uint64_t frameCounter = 0;

void initCuda(){
	cuInit(0);
	cuDeviceGet(&device, 0);
	cuCtxCreate(&context, 0, device);
	cuStreamCreate(&stream_upload, CU_STREAM_NON_BLOCKING);
	cuStreamCreate(&stream_download, CU_STREAM_NON_BLOCKING);

	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
}

Uniforms getUniforms(shared_ptr<GLRenderer> renderer){
	Uniforms uniforms;

	glm::mat4 world;
	glm::mat4 view = renderer->camera->view;
	glm::mat4 proj = renderer->camera->proj;
	glm::mat4 worldViewProj = proj * view * world;
	world = glm::transpose(world);
	view = glm::transpose(view);
	proj = glm::transpose(proj);
	worldViewProj = glm::transpose(worldViewProj);

	memcpy(&uniforms.world, &world, sizeof(world));
	memcpy(&uniforms.view, &view, sizeof(view));
	memcpy(&uniforms.proj, &proj, sizeof(proj));
	memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

	if(settings.doUpdateVisibility){
		transform_updatebound = worldViewProj;
	}

	glm::mat4 transform_inv_updatebound = glm::inverse(transform_updatebound);
	memcpy(&uniforms.transform_updateBound, &transform_updatebound, sizeof(transform_updatebound));
	memcpy(&uniforms.transformInv_updateBound, &transform_inv_updatebound, sizeof(transform_inv_updatebound));
	
	uniforms.width                  = static_cast<float>(renderer->width);
	uniforms.height                 = static_cast<float>(renderer->height);
	uniforms.fovy_rad               = 3.1415f * renderer->camera->fovy / 180.0;
	uniforms.time                   = static_cast<float>(now());
	uniforms.boxMin                 = float3{0.0f, 0.0f, 0.0f};
	uniforms.boxMax                 = boxSize;
	uniforms.frameCounter           = frameCounter;
	uniforms.showBoundingBox        = settings.showBoundingBox;
	uniforms.doUpdateVisibility     = settings.doUpdateVisibility;
	uniforms.showPoints             = settings.showPoints;
	uniforms.colorByNode            = settings.colorByNode;
	uniforms.colorByLOD             = settings.colorByLOD;
	uniforms.LOD                    = settings.LOD;
	uniforms.minNodeSize            = settings.minNodeSize;
	uniforms.pointSize              = settings.pointSize;
	uniforms.useHighQualityShading  = settings.useHighQualityShading;

	return uniforms;
}

// draw the octree with a CUDA kernel
void renderCUDA(shared_ptr<GLRenderer> renderer){

	Uniforms uniforms = getUniforms(renderer);

	static bool registered = false;
	static GLuint registeredHandle = -1;

	cuGraphicsGLRegisterImage(
		&cugl_colorbuffer, 
		renderer->view.framebuffer->colorAttachments[0]->handle, 
		GL_TEXTURE_2D, 
		CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// map OpenGL resources to CUDA
	vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
	cuGraphicsMapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	CUDA_RESOURCE_DESC res_desc = {};
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
	CUsurfObject output_surf;
	cuSurfObjectCreate(&output_surf, &res_desc);

	cuEventRecord(ce_render_start, 0);

	float time = static_cast<float>(now());
	int workgroupSize = 256;
	
	auto& kernel_render = cuda_program->kernels["kernel_render"];
	int maxActiveBlocksPerSM;
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, 
		kernel_render, workgroupSize, 0);
	
	int numGroups = maxActiveBlocksPerSM * numSMs;
	
	void* args[] = {
		&cptr_renderbuffer,
		&uniforms, 
		&output_surf,
		&cptr_stats
	};

	auto res_launch = cuLaunchCooperativeKernel(kernel_render,
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_render_end, 0);

	if(settings.benchmarkRendering){
		cuCtxSynchronize();
		cuEventElapsedTime(&renderingDuration, ce_render_start, ce_render_end);
	}

	cuSurfObjectDestroy(output_surf);
	cuGraphicsUnmapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	cuGraphicsUnregisterResource(cugl_colorbuffer);
}

// compile kernels and allocate buffers
void initCudaProgram(shared_ptr<GLRenderer> renderer){

	uint64_t cptr_buffer_bytes       = 300'000'000;
	uint64_t cptr_renderbuffer_bytes = 200'000'000;

	cuMemAlloc(&cptr_buffer                , cptr_buffer_bytes);
	cuMemAlloc(&cptr_renderbuffer          , cptr_renderbuffer_bytes);
	cuMemAlloc(&cptr_stats                 , sizeof(Stats));
	cuMemAllocHost((void**)&h_stats_pinned , sizeof(Stats));

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/render.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel_render"}
	});

	cuEventCreate(&ce_render_start, 0);
	cuEventCreate(&ce_render_end, 0);
	cuEventCreate(&ce_update_start, 0);
	cuEventCreate(&ce_update_end, 0);
	
	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

int main(){

	auto renderer = make_shared<GLRenderer>();
	auto cpu = getCpuData();
	// int numThreads = 2 * static_cast<int>(cpu.numProcessors);
	int numThreads = 2 * static_cast<int>(cpu.numProcessors);
	// numThreads = 16;
	// int numThreads = max(2 * cpu.numProcessors - 10, 2ull);
	printfmt("cpu.numProcessors: {} \n", cpu.numProcessors);
	printfmt("launching {} loader threads \n", numThreads);


	renderer->controls->yaw    = -1.15;
	renderer->controls->pitch  = -0.57;
	renderer->controls->radius = 10.0f;
	renderer->controls->target = {0.0f, 0.0f, 0.0f};


	initCuda();
	initCudaProgram(renderer);

	renderer->onFileDrop([&](vector<string> files){
		//vector<string> pointCloudFiles;

		//for(auto file : files){
		//	printfmt("dropped: {} \n", file);

		//	if(iEndsWith(file, "las") || iEndsWith(file, "laz")){
		//		pointCloudFiles.push_back(file);
		//	}else if(iEndsWith(file, "simlod")){
		//		pointCloudFiles.push_back(file);
		//	}
		//}

		//paths = pointCloudFiles;

		//if(settings.autoFocusOnLoad){
		//	renderer->controls->yaw = -1.15;
		//	renderer->controls->pitch = -0.57;
		//	renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
		//	renderer->controls->target = {
		//		boxSize.x * 0.5f,
		//		boxSize.y * 0.5f,
		//		boxSize.z * 0.1f
		//	};
		//}
	});

	auto update = [&](){
		renderer->camera->fovy = settings.fovy;
		renderer->camera->update();
	};

	auto render = [&](){

		timeSinceLastFrame = static_cast<float>(now()) - lastFrameTime;
		lastFrameTime = static_cast<float>(now());

		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		renderCUDA(renderer);

		static int statsAge = 0;
		{
			// copy stats from gpu to cpu.
			// actually laggs behind because we do async copy.
			// lacks sync, but as long as bytes are updated atomically in multiples of 4 or 8 bytes, 
			// results should be fine.

			// seems to be fine to add the async copy to the main stream?
			cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
			memcpy(&stats, h_stats_pinned, sizeof(Stats));

			statsAge = static_cast<int>(renderer->frameCount) - stats.frameID;
		}

		frameCounter++;
	};

	renderer->loop(update, render);

	return 0;
}