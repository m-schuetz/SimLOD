

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
#include "../CudaPrint/CudaPrint.h"

#include "unsuck.hpp"
#include "laszip_api.h"

#include "HostDeviceInterface.h"
#include "SimlodLoader.h"
#include "LasLoader.h"

using namespace std;

constexpr uint64_t PINNED_MEM_POOL_SIZE = 200;         // pool of pointers to batches of pinned memory
constexpr uint64_t BATCH_STREAM_SIZE    = 50;          // ring buffer of batches that are async streamed to GPU
constexpr uint64_t MAX_BATCH_SIZE       = 1'000'000;   // each loaded batch comprises <size> points
constexpr int MAX_LOADQUEUE_SIZE        = 300;         // stop loading from disk if processing lags behind

CudaPrint cudaprint;


vector<string> paths = {
	"NONE",
	// "d:/dev/pointclouds/riegl/retz.las",
};

struct PinnedMemorySlot {
	void* memLocation;
	CUevent uploadEnd;

	// ReadFileEx with unbuffered flag has special alignment requirements.
	// memLocation + memOffset gives the correct beginning of the data
	uint64_t memOffset;
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

struct PointBatch{
	string file = "";
	int first = 0;
	int count = 0;
	shared_ptr<vector<Point>> points;
	PinnedMemorySlot pinnedMem;

	LasHeader lasHeader;
};

bool lazInBatches = false;
deque<PointBatch> batchesToProcess;
deque<PointBatch> batchesInPinnedMemory;
deque<PointBatch> batchesInPageableMemory;
deque<PinnedMemorySlot> pinnedMemoryInUpload;
atomic_bool resetInProgress;
mutex mtx_uploader;
vector<unique_ptr<mutex>> mtx_loader;
mutex mtx_batchesToProcess;
mutex mtx_batchesInPinnedMemory;
mutex mtx_batchesInPageableMemory;
mutex mtx_pinnedMemoryInUpload;

int batchStreamUploadIndex = 0;
CUdeviceptr cptr_points_ring[BATCH_STREAM_SIZE];

CUdevice device;
CUcontext context;
int numSMs;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_buffer_persistent;
CUdeviceptr cptr_nodes;
CUdeviceptr cptr_renderbuffer;
CUdeviceptr cptr_stats;
CUdeviceptr cptr_numBatchesUploaded;
CUdeviceptr cptr_batchSizes;
CUdeviceptr cptr_frameStart;
CUgraphicsResource cugl_colorbuffer;
CUevent ce_render_start, ce_render_end;
CUevent ce_update_start, ce_update_end;
cudaStream_t stream_upload, stream_download;

CudaModularProgram* cuda_program_update = nullptr;
CudaModularProgram* cuda_program_render = nullptr;
// CudaModularProgram* cuda_program_filter = nullptr;
CudaModularProgram* cuda_program_reset  = nullptr;

glm::mat4 transform;
glm::mat4 transform_updatebound;

Stats stats;
void* h_stats_pinned = nullptr;

double t_drop_start = 0.0;

struct {
	bool useHighQualityShading       = true;
	bool showBoundingBox             = false;
	bool doUpdateVisibility          = true;
	bool showPoints                  = true;
	bool colorByNode                 = false;
	bool colorByLOD                  = false;
	bool colorWhite                  = false;
	bool autoFocusOnLoad             = true;
	bool benchmarkRendering          = false;
	float LOD                        = 0.2f;
	float minNodeSize                = 64.0f;
	int pointSize                    = 1;
	float fovy                       = 60.0f;
	bool enableEDL                   = true;
	float edlStrength                = 0.8f;
} settings;

struct PinnedMemPool{
	mutex mtx_pool;
	mutex mtx_register;

	deque<PinnedMemorySlot> pool;
	vector<PinnedMemorySlot> registered;

	PinnedMemPool(){
		
	}

	// put all registered batches back into the pool
	void refill(){
		pool.clear();
		pool.insert(
			pool.end(),
			make_move_iterator(registered.begin()),
			make_move_iterator(registered.end()));
	}

	void reserveSlots(int numSlots){
		for(int i = 0; i < numSlots; i++){
			this->reserveSlot();
		}
	}

	void reserveSlot(){
		uint64_t bytesPerSlot = MAX_BATCH_SIZE * sizeof(Point);
		// reserve some extra bytes in case of IO with special alignment
		uint64_t alignmentPadding = 1'048'576;

		void* pinnedMem;
		CUevent slotEvent;

		cuMemAllocHost((void**)&pinnedMem, bytesPerSlot + alignmentPadding);
		cuEventCreate(&slotEvent, 0);

		PinnedMemorySlot slot = {
			.memLocation = pinnedMem, 
			.uploadEnd   = slotEvent,
		};
		pool.push_back(slot);

		lock_guard<mutex> lock(mtx_register);
		registered.push_back(slot);
	}

	PinnedMemorySlot acquire(){
		lock_guard<mutex> lock(mtx_pool);

		if(pool.size() == 0){
			printfmt("pool is empty, allocating additional pinned memory slots \n");
			reserveSlot();
		}

		PinnedMemorySlot slot = pool.front();
		slot.memOffset = 0;

		pool.pop_front();

		return slot;
	}

	void release(PinnedMemorySlot slot){
		lock_guard<mutex> lock(mtx_pool);

		pool.push_back(slot);
	}

	void release(vector<PinnedMemorySlot> slots){

		if (slots.size() == 0) return;

		lock_guard<mutex> lock(mtx_pool);

		pool.insert(
			pool.end(),
			make_move_iterator(slots.begin()),
			make_move_iterator(slots.end()));
	}

};


bool requestReset                  = false;
bool requestBenchmark              = false;
atomic_bool requestStepthrough     = false;
atomic_bool requestStep            = false;
bool requestColorFiltering         = false;
float renderingDuration            = 0.0f;
uint32_t numPointsUploaded         = 0;
float loadStart                    = 0.0f;

float kernelUpdateDuration         = 0.0f;
float totalUpdateDuration          = 0.0f;
double minKernelUpdateDuration     = Infinity;
double maxKernelUpdateDuration     = 0.0;
double avgKernelUpdateDuration     = 0.0;
double cntKernelUpdateDuration     = 0.0;

float kernelRenderDuration         = 0.0f;
float totalRenderDuration          = 0.0f;
double minKernelRenderDuration     = Infinity;
double maxKernelRenderDuration     = 0.0;
double avgKernelRenderDuration     = 0.0;
double cntKernelRenderDuration     = 0.0;

atomic_uint64_t numPointsTotal     = 0;
atomic_uint64_t numPointsLoaded    = 0;
atomic_uint64_t numBytesTotal      = 0;
atomic_uint64_t numBytesLoaded     = 0;
atomic_uint64_t numThreadsLoading  = 0;
int numBatchesTotal                = 0;
int numBatchesProcessed            = 0;
bool lastBatchFinishedDevice       = false;
uint64_t momentaryBufferCapacity   = 0;
uint64_t persistentBufferCapacity  = 0;
vector<double> processFrameTimes; 

PinnedMemPool pinnedMemPool;

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
	
	uniforms.width                    = static_cast<float>(renderer->width);
	uniforms.height                   = static_cast<float>(renderer->height);
	uniforms.fovy_rad                 = 3.1415f * renderer->camera->fovy / 180.0;
	uniforms.time                     = static_cast<float>(now());
	uniforms.boxMin                   = float3{0.0f, 0.0f, 0.0f};
	uniforms.boxMax                   = boxSize;
	uniforms.frameCounter             = frameCounter;
	uniforms.showBoundingBox          = settings.showBoundingBox;
	uniforms.doUpdateVisibility       = settings.doUpdateVisibility;
	uniforms.showPoints               = settings.showPoints;
	uniforms.colorByNode              = settings.colorByNode;
	uniforms.colorByLOD               = settings.colorByLOD;
	uniforms.colorWhite               = settings.colorWhite;
	uniforms.LOD                      = settings.LOD;
	uniforms.minNodeSize              = settings.minNodeSize;
	uniforms.pointSize                = settings.pointSize;
	uniforms.useHighQualityShading    = settings.useHighQualityShading;
	uniforms.persistentBufferCapacity = persistentBufferCapacity;
	uniforms.momentaryBufferCapacity  = momentaryBufferCapacity;
	uniforms.enableEDL                = settings.enableEDL;
	uniforms.edlStrength              = settings.edlStrength;

	return uniforms;
}

void resetCUDA(shared_ptr<GLRenderer> renderer){

	Uniforms uniforms = getUniforms(renderer);

	void* args[] = {
		&uniforms, 
		&cptr_buffer_persistent,
		&cptr_nodes,
		&cptr_stats,
		&cudaprint.cptr,
		&cptr_numBatchesUploaded,
		&cptr_batchSizes,
	};

	// we only need one single thread to do the resetting on device
	uint32_t numGroups = 1;
	uint32_t workgroupSize = 1;

	auto res_launch = cuLaunchCooperativeKernel(cuda_program_reset->kernels["kernel"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		printfmt("CUDA kernel 'reset' failed.\n");
	}

	cuCtxSynchronize();
}

// incrementally updates the octree on the GPU
void updateOctree(shared_ptr<GLRenderer> renderer){

	// cuCtxSynchronize();
	
	Uniforms uniforms = getUniforms(renderer);

	int workgroupSize = 256;
	int numGroups     = 1 * numSMs;
	auto ptrPoints    = cptr_points_ring[0];

	void* args[] = {
		&uniforms, &ptrPoints, 
		&cptr_buffer, &cptr_buffer_persistent,
		&cptr_nodes,
		&cptr_stats, &cptr_frameStart,
		&cudaprint.cptr,
		&cptr_numBatchesUploaded,
		&cptr_batchSizes,
	};

	// printfmt("launching update!\n");
	// printfmt("    frame:                   {} \n", renderer->frameCount);
	// printfmt("    points:                  {} \n", batchStreamPointSize[processRingIndex]);
	// printfmt("    uploadIndex:             {} \n", batchStreamUploadIndex);
	// printfmt("    uploadRingIndex:         {} \n", batchStreamUploadIndex % BATCH_STREAM_SIZE);
	// printfmt("    processIndex:            {} \n", batchStreamProcessIndex);
	// printfmt("    processRingIndex:        {} \n", processRingIndex);
	// printfmt("    cptr_points_ringbuffer:  {} \n", cptr_points_ringbuffer);
	// printfmt("    ptrPoints:               {} \n", ptrPoints);

	cuEventRecord(ce_update_start, 0);
	// printfmt("launch update \n");
	auto res_launch = cuLaunchCooperativeKernel(cuda_program_update->kernels["kernel_construct"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);
	// printfmt("update done\n");

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_update_end, 0);

	// benchmark kernel- slows down overall loading!
	if(requestBenchmark){ 
		cuCtxSynchronize();

		float duration;
		cuEventElapsedTime(&duration, ce_update_start, ce_update_end);

		kernelUpdateDuration    += duration;
		minKernelUpdateDuration = std::min(minKernelUpdateDuration, double(duration));
		maxKernelUpdateDuration = std::max(maxKernelUpdateDuration, double(duration));
		avgKernelUpdateDuration = (cntKernelUpdateDuration * avgKernelUpdateDuration + duration) / (cntKernelUpdateDuration + 1.0);
		cntKernelUpdateDuration += 1.0;
	}

	requestStep = false;
	numBatchesProcessed++;

	// cuCtxSynchronize();
}

// post-process color-filtering.
// computes average color values for voxels.
// void doColorFiltering(shared_ptr<GLRenderer> renderer){

// 	if(!lastBatchFinishedDevice) return;

// 	Uniforms uniforms = getUniforms(renderer);

// 	int workgroupSize = 256;
// 	int numGroups = numSMs;

// 	void* args[] = {
// 		&uniforms,
// 		&cptr_buffer, 
// 		&cptr_nodes, 
// 		&cptr_stats
// 	};

// 	printfmt("launching color filter!\n");

// 	auto res_launch = cuLaunchCooperativeKernel(cuda_program_filter->kernels["kernel"],
// 		numGroups, 1, 1,
// 		workgroupSize, 1, 1,
// 		0, 0, args);

// 	if(res_launch != CUDA_SUCCESS){
// 		const char* str; 
// 		cuGetErrorString(res_launch, &str);
// 		printf("error: %s \n", str);
// 	}

// 	requestColorFiltering = false;
// }

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
	
	int maxActiveBlocksPerSM;
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, 
		cuda_program_render->kernels["kernel_render"], workgroupSize, 0);
	
	int numGroups = maxActiveBlocksPerSM * numSMs;
	
	void* args[] = {
		&cptr_renderbuffer,
		&uniforms, 
		&cptr_nodes, 
		&output_surf,
		&cptr_stats,
		&cptr_frameStart,
		& cudaprint.cptr
	};

	
	auto res_launch = cuLaunchCooperativeKernel(cuda_program_render->kernels["kernel_render"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_render_end, 0);

	// benchmark kernel- slows down overall loading!
	if(requestBenchmark){ 
		cuCtxSynchronize();

		float duration;
		cuEventElapsedTime(&duration, ce_render_start, ce_render_end);

		kernelRenderDuration    += duration;
		minKernelRenderDuration = std::min(minKernelRenderDuration, double(duration));
		maxKernelRenderDuration = std::max(maxKernelRenderDuration, double(duration));
		avgKernelRenderDuration = (cntKernelRenderDuration * avgKernelRenderDuration + duration) / (cntKernelRenderDuration + 1.0);
		cntKernelRenderDuration += 1.0;
	}

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

	// allocate most gpu buffers
	uint64_t nodesCapacity           = 200'000;
	uint64_t estimatedNodeSize       = 200;           // see struct Node in progressive_octree.cu, but some more just in case
	uint64_t cptr_buffer_bytes       = 300'000'000;
	uint64_t cptr_nodes_bytes        = nodesCapacity * estimatedNodeSize;
	uint64_t cptr_renderbuffer_bytes = 200'000'000;

	momentaryBufferCapacity = cptr_buffer_bytes;

	cuMemAlloc(&cptr_buffer                , cptr_buffer_bytes);
	cuMemAlloc(&cptr_nodes                 , cptr_nodes_bytes);
	cuMemAlloc(&cptr_renderbuffer          , cptr_renderbuffer_bytes);
	cuMemAlloc(&cptr_stats                 , sizeof(Stats));
	cuMemAlloc(&cptr_numBatchesUploaded    , 4);
	cuMemAlloc(&cptr_batchSizes            , 4 * BATCH_STREAM_SIZE);
	cuMemAlloc(&cptr_frameStart            , 8);
	cuMemAllocHost((void**)&h_stats_pinned , sizeof(Stats));

	// allocate ring buffer for point upload
	uint64_t cptr_points_bytes = MAX_BATCH_SIZE * sizeof(Point);
	CUdeviceptr devicemem = 0;

	cuMemAlloc(&devicemem, BATCH_STREAM_SIZE * cptr_points_bytes);

	for(uint64_t i = 0; i < BATCH_STREAM_SIZE; i++){
		cptr_points_ring[i] = devicemem + i * cptr_points_bytes;
	}

	// allocate persistent (over multiple frames) buffer with remaining GPU memory
	size_t availableMem = 0;
	size_t totalMem = 0;
	cuMemGetInfo(&availableMem, &totalMem);

	size_t cptr_buffer_persistent_bytes = static_cast<size_t>(static_cast<double>(availableMem) * 0.80);
	persistentBufferCapacity = cptr_buffer_persistent_bytes;
	cuMemAlloc(&cptr_buffer_persistent, cptr_buffer_persistent_bytes);

	uint64_t total = cptr_buffer_bytes 
		+ cptr_nodes_bytes
		+ cptr_renderbuffer_bytes
		+ sizeof(Stats)
		+ cptr_buffer_persistent_bytes;

	printfmt("cuMemAlloc(&cptr_buffer,            {:8L} MB);\n", cptr_buffer_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_nodes,             {:8L} MB);\n", cptr_nodes_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_renderbuffer,      {:8L} MB);\n", cptr_renderbuffer_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_stats,             {:8L}   );\n", sizeof(Stats));
	printfmt("cuMemAlloc(&cptr_buffer_persistent, {:8L} MB);\n", cptr_buffer_persistent_bytes / 1'000'000llu);
	printfmt("==============================================\n");
	printfmt("                                    {:8L} MB  \n", total / 1'000'000llu);
	printfmt("\n");

	cuda_program_update = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/progressive_octree_voxels.cu",
			  //"./modules/progressive_octree/progressive_octree_mno.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel_construct"}
	});

	cuda_program_reset = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/reset.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuda_program_render = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/render.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {"kernel_render"}
	});

	// cuda_program_filter = new CudaModularProgram({
	// 	.modules = {
	// 		"./modules/progressive_octree/colorfilter.cu",
	// 		"./modules/progressive_octree/utils.cu",
	// 	},
	// 	.kernels = {"kernel"}
	// });

	cuEventCreate(&ce_render_start, 0);
	cuEventCreate(&ce_render_end, 0);
	cuEventCreate(&ce_update_start, 0);
	cuEventCreate(&ce_update_end, 0);
	
	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

void reload(){

	printfmt("start loading \n");

	loadStart = static_cast<float>(now());
	totalUpdateDuration     = 0.0f;
	kernelUpdateDuration    = 0.0f;
	minKernelUpdateDuration = Infinity;
	maxKernelUpdateDuration = 0.0;
	avgKernelUpdateDuration = 0.0;
	cntKernelUpdateDuration = 0.0;

	totalRenderDuration     = 0.0f;
	kernelRenderDuration    = 0.0f;
	minKernelRenderDuration = Infinity;
	maxKernelRenderDuration = 0.0;
	avgKernelRenderDuration = 0.0;
	cntKernelRenderDuration = 0.0;

	lock_guard<mutex> lock_batchesToProcess(mtx_batchesToProcess);
	lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
	lock_guard<mutex> lock_batchesInPageableMemory(mtx_batchesInPageableMemory);
	lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);

	// reset workload and loaded data
	batchesToProcess.clear();
	batchesInPinnedMemory.clear();
	batchesInPageableMemory.clear();
	pinnedMemoryInUpload.clear();
	pinnedMemPool.refill();

	boxMin = float3{InfinityF, InfinityF, InfinityF};
	boxMax = float3{-InfinityF, -InfinityF, -InfinityF};
	boxSize = float3{0.0, 0.0, 0.0};

	numPointsTotal  = 0;
	numPointsLoaded = 0;
	numBatchesTotal = 0;
	numBytesTotal   = 0;
	numBytesLoaded  = 0;
	lazInBatches = false;

	// query bounding box and number of points in input files
	// vector<PointBatch> batches;
	for(string path : paths){

		if(!fs::exists(path)) continue;

		numBytesTotal += fs::file_size(path);

		if(iEndsWith(path, "laz")){
			lazInBatches = true;
		}

		if(iEndsWith(path, "las") || iEndsWith(path, "laz")){

			LasHeader header = loadHeader(path);
			numPointsTotal += header.numPoints;

			boxMin.x = std::min(boxMin.x, float(header.min[0]));
			boxMin.y = std::min(boxMin.y, float(header.min[1]));
			boxMin.z = std::min(boxMin.z, float(header.min[2]));

			boxMax.x = std::max(boxMax.x, float(header.max[0]));
			boxMax.y = std::max(boxMax.y, float(header.max[1]));
			boxMax.z = std::max(boxMax.z, float(header.max[2]));

			for (uint64_t first = 0; first < header.numPoints; first += MAX_BATCH_SIZE) {
				PointBatch batch;
				batch.file = path;
				batch.first = static_cast<int>(first);
				batch.count = static_cast<int>(std::min(header.numPoints - first, MAX_BATCH_SIZE));
				batch.lasHeader = header;

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}

		}else if(iEndsWith(path, "simlod")){

			auto buffer = readBinaryFile(path, 0, 24);
			uint64_t numPoints = (fs::file_size(path) - 24) / 16;

			boxMin.x = std::min(boxMin.x, buffer->get<float>( 0));
			boxMin.y = std::min(boxMin.y, buffer->get<float>( 4));
			boxMin.z = std::min(boxMin.z, buffer->get<float>( 8));

			boxMax.x = std::max(boxMax.x, buffer->get<float>(12));
			boxMax.y = std::max(boxMax.y, buffer->get<float>(16));
			boxMax.z = std::max(boxMax.z, buffer->get<float>(20));

			numPointsTotal += numPoints;

			for(uint64_t first = 0; first < numPoints; first += MAX_BATCH_SIZE){
				PointBatch batch;
				batch.file = path;
				batch.first = static_cast<int>(first);
				batch.count = static_cast<int>(std::min(numPoints - first, MAX_BATCH_SIZE));

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}
		}

		

		
	}

	// std::random_device rd;
	// std::mt19937 g(rd());
	// std::shuffle(loadQueue.begin(), loadQueue.end(), g);
	// std::reverse(loadQueue.begin(), loadQueue.end());

	// if(batches.size() > 0){
	// 	loadQueue.push_back(batches[batches.size() - 1]);
	// 	loadQueue.push_back(batches[batches.size() - 2]);
	// }

	boxSize.x = boxMax.x - boxMin.x;
	boxSize.y = boxMax.y - boxMin.y;
	boxSize.z = boxMax.z - boxMin.z;

	stats                   = Stats();
	numPointsUploaded       = 0;
	numBatchesProcessed     = 0;
	lastBatchFinishedDevice = false;
	batchStreamUploadIndex  = 0;
	processFrameTimes.clear();
}

void reset(shared_ptr<GLRenderer> renderer){
	// before locking, notify threads that they should not try to acquire the lock (otherwise we might wait for a long time to lock the others out)
	resetInProgress = true;

	// lock out upload threads and loader threads
	for (size_t i = 0; i < mtx_loader.size(); ++i) {
		mtx_loader[i]->lock();
	}
	lock_guard<mutex> lock_upload(mtx_uploader);

	// finish pending upload
	cuCtxSynchronize();

	// now reset octree-construction related state on device
	resetCUDA(renderer);
	cuCtxSynchronize();

	// read stats just to make sure device and host are on the same page
	cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
	memcpy(&stats, h_stats_pinned, sizeof(Stats));
	cuCtxSynchronize();

	// finally, reset host state to start reloading current data set from disk
	reload();

	requestReset = false;

	// notify other threads that they may now try to acquire the lock again
	resetInProgress = false;

	// unlock loader threads
	for (size_t i = 0; i < mtx_loader.size(); ++i) {
		mtx_loader[i]->unlock();
	}
}

void spawnLoader(size_t i) {

	auto cpu = getCpuData();

	thread t([&, i]() {
		cuCtxSetCurrent(context);

		while (true) {
			bool everythingIsDone = batchStreamUploadIndex == numBatchesTotal;
			bool processingLagsBehind = numPointsLoaded > stats.numPointsProcessed + BATCH_STREAM_SIZE * MAX_BATCH_SIZE;

			// if (processingLagsBehind) {
			// 	printfmt("processing lags behind\n");
			// }

			if (everythingIsDone || processingLagsBehind || resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// lock thread's own mutex
			lock_guard<mutex> lock_loader(*mtx_loader[i]);

			PointBatch batch;

			mtx_batchesToProcess.lock();
			// get batch to load, if load queue isn't full
			if (batchesToProcess.size() > 0) {
				batch = batchesToProcess.front();

				// if it's a laz file, limit loading to fewer thrads.
				int maxThreads = std::max(double(cpu.numProcessors) / 2.0, 1.0);
				if(batch.count > 0 && iEndsWith(batch.file, "laz"))
				if(numThreadsLoading >= maxThreads)
				{
					mtx_batchesToProcess.unlock();
					continue;
				}

				batchesToProcess.pop_front();
			}
			mtx_batchesToProcess.unlock();

			PinnedMemorySlot pinnedMemSlot = pinnedMemPool.acquire();
			Point* pinnedPoints = (Point*)pinnedMemSlot.memLocation;
			
			if (batch.count > 0) {
				// load points in batch

				int batchID = batch.first / MAX_BATCH_SIZE;
				double t_start = now();
				// printfmt("start loading batch {} at {:.3f} \n", batchID, t_start);

				numThreadsLoading++;
				if(iEndsWith(batch.file, "las")){
					void* target = (void*)pinnedPoints;

					double translation[3] = {-boxMin.x, -boxMin.y, -boxMin.z};
					loadLasNative(batch.file, batch.lasHeader, batch.first, batch.count, target, translation);
					numBytesLoaded += batch.count * batch.lasHeader.bytesPerPoint;

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);

				}else if(iEndsWith(batch.file, "laz")){
					laszip_POINTER laszip_reader = nullptr;
					laszip_header* header = nullptr;
					laszip_point* laz_point = nullptr;

					laszip_BOOL is_compressed;
					laszip_BOOL request_reader = true;

					laszip_create(&laszip_reader);
					laszip_request_compatibility_mode(laszip_reader, request_reader);
					laszip_open_reader(laszip_reader, batch.file.c_str(), &is_compressed);

					laszip_get_header_pointer(laszip_reader, &header);
					laszip_get_point_pointer(laszip_reader, &laz_point);
					laszip_seek_point(laszip_reader, batch.first);

					for (int i = 0; i < batch.count; i++) {
						double XYZ[3];
						laszip_read_point(laszip_reader);
						laszip_get_coordinates(laszip_reader, XYZ);

						Point point;
						point.x = static_cast<float>(XYZ[0] - boxMin.x);
						point.y = static_cast<float>(XYZ[1] - boxMin.y);
						point.z = static_cast<float>(XYZ[2] - boxMin.z);

						auto rgb = laz_point->rgb;
						point.rgba[0] = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
						point.rgba[1] = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
						point.rgba[2] = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];
						
						//int intensity = laz_point->intensity;
						//point.rgba[0] = intensity / 200;
						//point.rgba[1] = intensity / 200;
						//point.rgba[2] = intensity / 200;


						pinnedPoints[i] = point;
					}

					laszip_close_reader(laszip_reader);

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}else if(iEndsWith(batch.file, "simlod")){
					
					batch.pinnedMem = pinnedMemSlot;

					// At least on windows, this uses winapi to do unbuffered loading to maximize SSD perf
					uint64_t padding;
					loadFileNative(batch.file, 24llu + 16llu * uint64_t(batch.first), 16llu * batch.count, pinnedMemSlot.memLocation, &padding);
					batch.pinnedMem.memOffset = padding;

					numBytesLoaded += 16llu * batch.count;
					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}

				// double t_end = now();
				// double millies = (t_end - t_start) * 1000.0;
				// printfmt("finished loading batch {} at {:.3f}. duration: {:.3f} ms \n", batchID, t_end, millies);

				numThreadsLoading--;
			}else {
				// give back pinned memory slot if we didn't use it
				pinnedMemPool.release(pinnedMemSlot);
			}

			std::this_thread::sleep_for(1ms);
		}

		});
	t.detach();
}

// Spawns a thread that uploads data to the GPU
// - Asynchronously schedules uploads in stream_upload
// - After every async copy, it also asynchronously updates the number of uploaded points
void spawnUploader(shared_ptr<GLRenderer> renderer) {
	double timestamp = now();

	thread t([&]() {

		vector<PinnedMemorySlot> availableSlots;

		while (true) {

			double t_start = now();
			double spintime = 0.0001;

			// actually seems to wait way, way longer than 100ns.
			// std::this_thread::sleep_for(100ns);

			// go to sleep if reset is in progress
			if (resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// this lock ensures that we don't reset and upload at the same time
			lock_guard<mutex> lock_uploader(mtx_uploader);

			// spin instead of sleep, because sleep takes too long
			while (now() < t_start + spintime) {
				// do some spinning to avoid attempting locks too often
			}

			if(requestStepthrough){
				if(requestStep){
					requestStep = false;
				}else{
					continue;
				}
			}

			// reclaim all pinned memory slots that are no longer needed and give them back to the pool
			mtx_pinnedMemoryInUpload.lock();
			while (!pinnedMemoryInUpload.empty() && cuEventQuery(pinnedMemoryInUpload.front().uploadEnd) == cudaSuccess) {
				availableSlots.push_back(pinnedMemoryInUpload.front());
				pinnedMemoryInUpload.pop_front();
			}
			mtx_pinnedMemoryInUpload.unlock();

			pinnedMemPool.release(availableSlots);
			availableSlots.clear();

			bool everythingIsDone = batchStreamUploadIndex == numBatchesTotal;
			bool processingLagsBehind = numPointsUploaded > stats.numPointsProcessed + BATCH_STREAM_SIZE * MAX_BATCH_SIZE;

			if (everythingIsDone) continue;
			if (processingLagsBehind) continue;

			auto t_00 = now();

			// acquire work, or keep spinning if there is none
			PointBatch batch;
			{
				lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);

				if (batchesInPinnedMemory.size() > 0) {
					batch = batchesInPinnedMemory.front();
					batchesInPinnedMemory.pop_front();
				} else {
					continue;
				}
			}

			// UPLOAD
			uint32_t targetSlot = batchStreamUploadIndex;
			int uploadRingIndex = targetSlot % BATCH_STREAM_SIZE;

			auto source     = ((uint8_t*)batch.pinnedMem.memLocation) + batch.pinnedMem.memOffset;
			auto target     = cptr_points_ring[uploadRingIndex];
			size_t byteSize = batch.count * sizeof(Point);

			cuMemcpyHtoDAsync(target, source, byteSize, stream_upload);

			// record upload event
			cuEventRecord(batch.pinnedMem.uploadEnd, stream_upload);

			// since we process N batches per frame and batches may have varying amounts of points, 
			// we need to let the kernel know the size of each individual batch
			cuMemsetD32Async(cptr_batchSizes + 4 * uploadRingIndex, batch.count, 1, stream_upload);

			// also let the kernel know how many batches we uploaded
			cuMemsetD32Async(cptr_numBatchesUploaded, batchStreamUploadIndex + 1, 1, stream_upload);

			batchStreamUploadIndex++;
			numPointsUploaded += batch.count;

			lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);
			pinnedMemoryInUpload.push_back(batch.pinnedMem);
		}

		});
	t.detach();

	setThreadPriorityHigh(t);
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
	renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
	renderer->controls->target = {
		boxSize.x * 0.5f,
		boxSize.y * 0.5f,
		boxSize.z * 0.1f
	};

	// renderer->controls->yaw    = 0.982;
	// renderer->controls->pitch  = -0.875;
	// renderer->controls->radius = 449.807;
	// renderer->controls->target = { 1154.460, 218.177, -92.225, };

	// renderer->controls->yaw    = 7.670;
	// renderer->controls->pitch  = -0.677;
	// renderer->controls->radius = 929.239;
	// renderer->controls->target = { 606.560, 385.040, 13.848, };

	// position: 448.8209204653559, 768.7683535080489, 23.676426584479366 
	// renderer->controls->yaw    = -4.660;
	// renderer->controls->pitch  = -0.293;
	// renderer->controls->radius = 94.341;
	// renderer->controls->target = { 354.609, 764.038, 25.101, };


	initCuda();
	initCudaProgram(renderer);

	pinnedMemPool.reserveSlots(PINNED_MEM_POOL_SIZE);

	mtx_loader.reserve(numThreads);
	for(int i = 0; i < numThreads; i++){
		mtx_loader.push_back(make_unique<mutex>());
		spawnLoader(i);
	}

	cudaprint.init();
	
	spawnUploader(renderer);
	
	reload();

	renderer->onFileDrop([&](vector<string> files){
		vector<string> pointCloudFiles;

		t_drop_start = now();
		printfmt("drop at {:.3f} \n", now());

		for(auto file : files){
			printfmt("dropped: {} \n", file);

			if(iEndsWith(file, "las") || iEndsWith(file, "laz")){
				pointCloudFiles.push_back(file);
			}else if(iEndsWith(file, "simlod")){
				pointCloudFiles.push_back(file);
			}
		}

		paths = pointCloudFiles;

		reset(renderer);

		if(settings.autoFocusOnLoad){
			renderer->controls->yaw = -1.15;
			renderer->controls->pitch = -0.57;
			renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
			renderer->controls->target = {
				boxSize.x * 0.5f,
				boxSize.y * 0.5f,
				boxSize.z * 0.1f
			};
		}
	});

	auto update = [&](){
		cudaprint.update();

		renderer->camera->fovy = settings.fovy;
		renderer->camera->update();
	};

	auto render = [&](){

		timeSinceLastFrame = static_cast<float>(now()) - lastFrameTime;
		lastFrameTime = static_cast<float>(now());

		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		if(!lastBatchFinishedDevice){
			processFrameTimes.push_back(timeSinceLastFrame);
		}

		if(requestReset){
			reset(renderer);
		}

		renderCUDA(renderer);

		if(!lastBatchFinishedDevice){
			updateOctree(renderer);
		}

		// if(requestColorFiltering){
		// 	doColorFiltering(renderer);
		// }

		if(!lastBatchFinishedDevice){
			totalUpdateDuration = 1000.0f * (static_cast<float>(now()) - loadStart);
		}else if(requestBenchmark && lastBatchFinishedDevice){
			requestBenchmark = false;
			printfmt("finished loading, disabling benchmarking. \n");
		}

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

			static uint64_t previousNumPointsProcessed = 0;
			uint64_t numPointsProcessed = stats.numPointsProcessed;

			// if(numPointsProcessed != previousNumPointsProcessed){
			// 	printfmt("processed {} at {:.3f}. since drop: {:.3f} \n", numPointsProcessed, now(), now() - t_drop_start);

			// 	previousNumPointsProcessed = numPointsProcessed;
			// }
			

			bool newLastBatchFinishedDevice = stats.numPointsProcessed == uint64_t(numPointsTotal);
			if(stats.memCapacityReached){
				newLastBatchFinishedDevice = true;
			}
			if(newLastBatchFinishedDevice != lastBatchFinishedDevice){
				lastBatchFinishedDevice = newLastBatchFinishedDevice;
				printfmt("stats.numPointsProcessed = {} \n", stats.numPointsProcessed);
				printfmt("numPointsTotal = {} \n", uint64_t(numPointsTotal));
				printfmt("setting lastBatchFinishedDevice = {} \n", lastBatchFinishedDevice ? "true" : "false");
			}
		}

		if(Runtime::showGUI)
		{ // RENDER IMGUI SETTINGS WINDOW

			auto windowSize = ImVec2(490, 280);
			ImGui::SetNextWindowPos(ImVec2(10, 300));
			ImGui::SetNextWindowSize(windowSize);

			ImGui::Begin("Settings");
			// ImGui::Text("Test abc");
			ImGui::Checkbox("Show Bounding Box",        &settings.showBoundingBox);
			ImGui::Checkbox("Update Visibility",        &settings.doUpdateVisibility);
			ImGui::Checkbox("Show Points",              &settings.showPoints);
			ImGui::Checkbox("Color by Node",            &settings.colorByNode);
			ImGui::Checkbox("Color by LOD",             &settings.colorByLOD);
			// ImGui::Checkbox("Color white",              &settings.colorWhite);
			ImGui::Checkbox("enable Eye Dome Lighting", &settings.enableEDL);
			ImGui::Checkbox("High-Quality-Shading",     &settings.useHighQualityShading);
			ImGui::Checkbox("Auto-focus on load",       &settings.autoFocusOnLoad);
			ImGui::Checkbox("Benchmark Rendering",      &settings.benchmarkRendering);

			if(ImGui::Button("Reset")){
				requestReset = true;
				requestStepthrough = false;
			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Reset + Benchmark")){
				requestReset = true;
				requestBenchmark = true;
				requestStepthrough = false;
			}
			ImGui::SameLine(0.0f);
			
			if(ImGui::Button("Stepthrough")){
				if(lastBatchFinishedDevice){
					requestReset = true;
				}
				requestStepthrough = true;
				requestStep = true;
			}

			ImGui::Text("Test Data Views");

			// original size: 1920x1080
			// clip size:
			//     offset: 63, 38
			//     size: 1795, 1010
			// resized to 50%, saved as 80% jpeg quality

			if(ImGui::Button("Chiller - bird")){
				// position: 39.55564356573898, -4.472634983341328, 9.256686713258468 
				renderer->controls->yaw    = -5.237;
				renderer->controls->pitch  = -0.542;
				renderer->controls->radius = 34.626;
				renderer->controls->target = { 9.595, 10.394, 0.295, };
			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Chiller - close")){
				// position: 19.21071216298619, -0.590067491220811, 1.5756389652824982 
				renderer->controls->yaw    = -5.752;
				renderer->controls->pitch  = 0.090;
				renderer->controls->radius = 16.153;
				renderer->controls->target = { 11.035, 13.285, 2.828, };
			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Retz - bird")){
				// position: -442.751714425827, 1032.8670571391256, -310.45475033534075 
				renderer->controls->yaw    = -1.808;
				renderer->controls->pitch  = -0.997;
				renderer->controls->radius = 1166.684;
				renderer->controls->target = { 691.401, 884.472, -80.610, };

			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Retz - close")){
				// position: 627.9994594851617, 802.2611126991757, 76.41850773669957 
				renderer->controls->yaw    = 0.750;
				renderer->controls->pitch  = -0.418;
				renderer->controls->radius = 80.902;
				renderer->controls->target = { 572.854, 856.372, 52.416, };

			}

			if(ImGui::Button("Morro Bay - bird")){
				// position: 1602.1138827457712, -475.95272623084657, 2313.666990965035 
				renderer->controls->yaw    = -0.207;
				renderer->controls->pitch  = -0.797;
				renderer->controls->radius = 3866.886;
				renderer->controls->target = { 2398.747, 2167.120, -394.165, };
			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Morro Bay - close")){
				// position: 2840.684032348224, 949.9487599422316, 81.9126308772043 
				renderer->controls->yaw    = -11.270;
				renderer->controls->pitch  = -0.225;
				renderer->controls->radius = 93.982;
				renderer->controls->target = { 2750.218, 974.775, 76.230, };
			}
			

			if(ImGui::Button("Meroe - bird")){
				// position: -366.08263489517935, 261.79980089364733, 206.0866739972536 
				renderer->controls->yaw    = -7.430;
				renderer->controls->pitch  = -0.617;
				renderer->controls->radius = 929.239;
				renderer->controls->target = { 480.880, 573.485, -15.254, };
			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Meroe - close")){
				// position: 386.9127932944783, 808.8478521019321, 16.78255342532026 
				renderer->controls->yaw    = -4.527;
				renderer->controls->pitch  = -0.192;
				renderer->controls->radius = 44.011;
				renderer->controls->target = { 343.652, 800.906, 18.330, };
			}

			if(ImGui::Button("Endeavor - bird")){
				// position: 641.9867239682803, 464.3862478069415, 613.116113282369 
				renderer->controls->yaw    = -6.045;
				renderer->controls->pitch  = -0.713;
				renderer->controls->radius = 187.827;
				renderer->controls->target = { 597.671, 602.508, 493.795, };
			}
			ImGui::SameLine(0.0f);

			if(ImGui::Button("Endeavor - close")){
				// position: 600.8022710580775, 597.6937750182759, 508.70460986245035 
				renderer->controls->yaw    = -12.560;
				renderer->controls->pitch  = -0.018;
				renderer->controls->radius = 8.087;
				renderer->controls->target = { 600.751, 605.780, 508.563, };
			}
			
			ImGui::SliderFloat("minNodeSize", &settings.minNodeSize, 32.0f, 1024.0f);
			ImGui::SliderInt("Point Size", &settings.pointSize, 1, 10);
			ImGui::SliderFloat("FovY", &settings.fovy, 20.0f, 100.0f);
			ImGui::SliderFloat("EDL Strength", &settings.edlStrength, 0.0f, 3.0f);

			if(ImGui::Button("Copy Camera")){
				auto controls = renderer->controls;
				auto pos = controls->getPosition();
				auto target = controls->target;

				stringstream ss;
				ss<< std::setprecision(2) << std::fixed;
				ss << format("// position: {}, {}, {} \n", pos.x, pos.y, pos.z);
				ss << format("renderer->controls->yaw    = {:.3f};\n", controls->yaw);
				ss << format("renderer->controls->pitch  = {:.3f};\n", controls->pitch);
				ss << format("renderer->controls->radius = {:.3f};\n", controls->radius);
				ss << format("renderer->controls->target = {{ {:.3f}, {:.3f}, {:.3f}, }};\n", target.x, target.y, target.z);

				string str = ss.str();
				
#ifdef _WIN32
				toClipboard(str);
#endif
			}

			// if(ImGui::Button("Do Color Filtering!")){
			// 	requestColorFiltering = true;
			// }

			ImGui::End();
		}
		
		if(Runtime::showGUI)
		{ // RENDER IMGUI STATS WINDOW

			auto windowSize = ImVec2(490, 440);
			ImGui::SetNextWindowPos(ImVec2(10, 590));
			ImGui::SetNextWindowSize(windowSize);

			ImGui::Begin("Stats");

			{ // used/total mem progress
				size_t availableMem = 0;
				size_t totalMem = 0;
				cuMemGetInfo(&availableMem, &totalMem);
				size_t unavailableMem = totalMem - availableMem;

				string strProgress = format("{:3.1f} / {:3.1f}", 
					double(unavailableMem) / 1'000'000'000.0, 
					double(totalMem) / 1'000'000'000.0
				);
				float progress = static_cast<float>(static_cast<double>(unavailableMem) / static_cast<double>(totalMem));
				ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Used GPU Memory");
			}

			{ // loaded/total points

				string strProgress = format("{} M / {} M", 
					numPointsLoaded / 1'000'000,
					numPointsTotal / 1'000'000
				);
				float progress = static_cast<float>(static_cast<double>(numPointsLoaded) / static_cast<double>(numPointsTotal));
				ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Loaded Points");
			}

			{ // loaded/total points

				string strProgress = format("{} M / {} M", 
					stats.numPointsProcessed / 1'000'000, 
					numPointsTotal / 1'000'000
				);
				float progress = static_cast<float>(static_cast<double>(stats.numPointsProcessed) / static_cast<double>(numPointsTotal));
				ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
				ImGui::Text("Processed Points");
			}

			auto locale = getSaneLocale();
			uint32_t numEmptyLeaves = stats.numLeaves - stats.numNonemptyLeaves;

			auto toMS = [locale](double millies){
				string str = "-";

				if(millies > 0.0){
					str = format("{:.1Lf} ms", millies);
				}

				return leftPad(str, 15);
			};

			auto toM = [locale](double number){
				string str = format(locale, "{:.1Lf} M", number / 1'000'000.0);
				return leftPad(str, 14);
			};

			auto toB = [locale](double number) {
				string str = format(locale, "{:.1Lf} B", number / 1'000'000'000.0);
				return leftPad(str, 14);
			};

			auto toMB = [locale](double number){
				string str = format(locale, "{:.1Lf} MB", number / 1'000'000.0);
				return leftPad(str, 15);
			};
			auto toGB = [locale](double number){
				string str = format(locale, "{:.1Lf} GB", number / 1'000'000'000.0);
				return leftPad(str, 15);
			};

			auto toIntString = [locale](double number){
				string str = format(locale, "{:L}", number);
				return leftPad(str, 10);
			};

			double pointsSecUpdate = double(stats.numPointsProcessed) / (double(kernelUpdateDuration) / 1000.0);
			double pointsSecTotal = double(stats.numPointsProcessed) / (double(totalUpdateDuration) / 1000.0);
			double gbs_file = double(numBytesLoaded) / (totalUpdateDuration / 1000.0);
			double gbs_gpu = double(16llu * stats.numPointsProcessed) / (totalUpdateDuration / 1000.0);

			double millionPointsSecRendered = 1000.0 * float(stats.numVisiblePoints) / renderingDuration;
			double millionVoxelsSecRendered = 1000.0 * float(stats.numVisibleVoxels) / renderingDuration;
			double millionSamplesSecRendered = 1000.0 * (float(stats.numVisiblePoints + stats.numVisibleVoxels) / renderingDuration);

			if(!settings.benchmarkRendering){
				millionPointsSecRendered = 0.0;
				millionVoxelsSecRendered = 0.0;
				millionSamplesSecRendered = 0.0;
				renderingDuration = 0.0;
			}

			double M = 1'000'000.0;
			double B = 1'000'000'000.0;
			double MB = 1'000'000.0; // TIL: MB = 1'000'000 vs. MiB = 1024 * 1024
			double GB = 1'000'000'000.0;

			vector<vector<string>> table = {
				{"#update kernel duration  ", toMS(kernelUpdateDuration)                            , format("{:.1f}", kernelUpdateDuration)},
				{"    max                  ", toMS(maxKernelUpdateDuration)                         , format("{:.1f}", maxKernelUpdateDuration)},
				{"    avg                  ", toMS(avgKernelUpdateDuration)                         , format("{:.1f}", avgKernelUpdateDuration)},
				{"#update total duration   ", toMS(totalUpdateDuration)                             , format("{:.1f}", totalUpdateDuration)},
				{"points/sec update kernel ", toM(pointsSecUpdate)                                  , format("{:.1f}", totalUpdateDuration / M)},
				{"points/sec total         ", toM(pointsSecTotal)                                   , format("{:.1f}", pointsSecTotal / M)},
				{"GB/s (disk I/O)          ", toGB(gbs_file)                                        , format("{:.1f}", gbs_file / GB)},
				{"GB/s (gpu)               ", toGB(gbs_gpu)                                         , format("{:.1f}", gbs_gpu / GB)},
				{"=========================", " "                                                   , " "},
				{"#render kernel duration  ", toMS(kernelRenderDuration)                            , format("{:.1f}", kernelRenderDuration)},
				{"=========================", " "                                                   , " "},
				{"rendering duration       ", toMS(renderingDuration)                               , format("{:.1f}", renderingDuration)},
				{"    points / sec         ", toB(millionPointsSecRendered)                        , format("{:.1f}", millionPointsSecRendered / B)},
				{"    voxels / sec         ", toB(millionVoxelsSecRendered)                        , format("{:.1f}", millionVoxelsSecRendered / B)},
				{"    samples / sec        ", toB(millionSamplesSecRendered)                       , format("{:.1f}", millionSamplesSecRendered / B)},
				{"=========================", " "                                                   , " "},
				{"#points processed        ", toM(double(stats.numPointsProcessed))                 , format("{:.1f}", stats.numPointsProcessed / M)},
				{"#nodes                   ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
				{"    #inner               ", toIntString(stats.numInner)                           , format("{}", stats.numInner)},
				{"    #leaves (nonempty)   ", toIntString(stats.numNonemptyLeaves)                  , format("{}", stats.numNonemptyLeaves)},
				{"    #leaves (empty)      ", toIntString(numEmptyLeaves)                           , format("{}", numEmptyLeaves)},
				{"#chunks                  ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
				{"    #voxels              ", toIntString(stats.numChunksVoxels)                    , format("{}", stats.numChunksVoxels)},
				{"    #points              ", toIntString(stats.numChunksPoints)                    , format("{}", stats.numChunksPoints)},
				{"#samples                 ", toM(stats.numPoints + stats.numVoxels)                , format("{:.1f}", (stats.numPoints + stats.numVoxels) / M)},
				{"    #points              ", toM(stats.numPoints)                                  , format("{:.1f}", stats.numPoints / M)},
				{"    #voxels              ", toM(stats.numVoxels)                                  , format("{:.1f}", stats.numVoxels / M)},
				{"momentary buffer         ", toMB(stats.allocatedBytes_momentary)                  , format("{:.1f}", stats.allocatedBytes_momentary / MB)},
				{"persistent buffer        ", toMB(stats.allocatedBytes_persistent)                 , format("{:.1f}", stats.allocatedBytes_persistent / MB)},
				{"=========================", " "                                                   , " "},
				{"#visible nodes           ", toIntString(stats.numVisibleNodes)                    , format("{}", stats.numVisibleNodes)},
				{"    #inner               ", toIntString(stats.numVisibleInner)                    , format("{}", stats.numVisibleInner)},
				{"    #leaves              ", toIntString(stats.numVisibleLeaves)                   , format("{}", stats.numVisibleLeaves)},
				{"#visible samples         ", toM(stats.numVisiblePoints + stats.numVisibleVoxels)  , format("{:.1f}", (stats.numVisiblePoints + stats.numVisibleVoxels) / 1'000'000.0f)},
				{"    #points              ", toM(stats.numVisiblePoints)                           , format("{:.1f}", stats.numVisiblePoints / 1'000'000.0f)},
				{"    #voxels              ", toM(stats.numVisibleVoxels)                           , format("{:.1f}", stats.numVisibleVoxels / 1'000'000.0f)},
			};

			if(ImGui::Button("Copy Stats")){
				stringstream ss;
				for (int row = 0; row < table.size(); row++){
					for (int column = 0; column < 2; column++){
						ss << table[row][column];
					}
					ss << "\n";
				}


				string str = ss.str();
				toClipboard(str);
			}
			
			
			auto flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV;
			if (ImGui::BeginTable("table1", 3, flags)){
				ImGui::TableSetupColumn("AAA", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("BBB", ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("CCC", ImGuiTableColumnFlags_WidthFixed);
				for (int row = 0; row < table.size(); row++){
					ImGui::TableNextRow();
					for (int column = 0; column < 2; column++){
						ImGui::TableSetColumnIndex(column);
						
						ImGui::Text(table[row][column].c_str());
					}

					ImGui::PushID(row);

					ImGui::TableSetColumnIndex(2);
					if (ImGui::SmallButton("c")) {
						string str = table[row][2];
						toClipboard(str);
					}

					ImGui::PopID();
				}
				ImGui::EndTable();
			}
			

			ImGui::End();
		}

		if(stats.memCapacityReached || (lazInBatches && !lastBatchFinishedDevice))
		{
			ImGuiIO& io = ImGui::GetIO();
			ImGui::SetNextWindowSize(ImVec2(700, 100));
			// ImGui::SetNextWindowPos(ImVec2(300, 200));
			ImGui::SetNextWindowPos(
				ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y - 100), 
				ImGuiCond_Always, ImVec2(0.5f,0.5f)
			);

			ImGui::Begin("Message");

			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));

			if(stats.memCapacityReached){
#ifdef __cpp_lib_format
        string message = std::format("WARNING: Octree mem usage ({} MB) approaches total capacity ({} MB). Further points are ignored.",
					stats.allocatedBytes_persistent / 1'000'000llu,
					persistentBufferCapacity / 1'000'000llu
				);
#else
        string message = fmt::format("WARNING: Octree mem usage ({} MB) approaches total capacity ({} MB). Further points are ignored.",
                                     stats.allocatedBytes_persistent / 1'000'000llu,
                                     persistentBufferCapacity / 1'000'000llu
        );
#endif

				ImGui::Text(message.c_str());
			}

			if(lazInBatches){
				ImGui::Text("WARNING: Loading *.laz files. Load routines are optimized for *.las or *.simlod - laz may be slow.");
			}
			
			ImGui::Text(" ");

			ImGui::PopStyleColor();


			ImGui::End();
		}

		

		frameCounter++;
	};

	renderer->loop(update, render);

	return 0;
}
