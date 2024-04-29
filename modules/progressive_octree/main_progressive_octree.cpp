

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
//#include "src/arithmeticdecoder.hpp"
//#include "src/integercompressor.hpp"

using namespace std;

vector<string> paths = {
	"NONE",
	// "d:/dev/pointclouds/riegl/retz.las",
};

struct Chunk_HostData{
	uint32_t tileID;
	uint32_t chunkIndex; // within tile
	bool isLoading = false;
	bool isLoaded = false;
};

vector<string> tilePaths;
vector<Tile> tiles;
vector<Chunk> chunks;
vector<Chunk_HostData> chunks_hostData;

CUdevice device;
CUcontext context;
int numSMs;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_stats;
CUdeviceptr cptr_frameStart;
CUdeviceptr cptr_tiles;
CUdeviceptr cptr_chunks;
CUdeviceptr cptr_commandsQueue;
CUdeviceptr cptr_commandsQueueCounter;
CUdeviceptr cptr_chunksToLoad, cptr_numChunksToLoad;



CUgraphicsResource cugl_colorbuffer;
CUevent ce_render_start, ce_render_end;
CUevent ce_update_start, ce_update_end;
cudaStream_t stream_upload, stream_download;

CudaModularProgram* cuda_program = nullptr;

glm::mat4 transform;
glm::mat4 transform_updatebound;

Stats stats;
void* h_stats_pinned = nullptr;

void* h_commandQueue_pinned = nullptr;
void* h_commandQueueCounter_pinned = nullptr;
uint64_t commandsLoadedFromDevice = 0;

void* h_chunksToLoad_pinned = nullptr;
void* h_numChunksToLoad_pinned = nullptr;

struct PendingCommandLoad{
	uint64_t start_0;
	uint64_t end_0;
	uint64_t start_1;
	uint64_t end_1;
	CUevent ce_ranges_loaded;
};
deque<PendingCommandLoad> pendingCommandLoads;

struct Task_LoadChunk{
	int tileID;
	int chunkIndex; // within tile
	int chunkID;
};
struct Task_UploadChunk {
	int tileID;
	int chunkIndex;
	int chunkID;
	shared_ptr<Buffer> points;
	int numPoints;
};
struct Task_UnloadChunk{
	int tileID;
	int chunkIndex;
	int chunkID;
	uint64_t cptr;
};

deque<Task_LoadChunk> tasks_loadChunk;
deque<Task_UploadChunk> tasks_uploadChunk;
deque<Task_UnloadChunk> tasks_unloadChunk;
mutex mtx_loadChunk;
mutex mtx_uploadChunk;
mutex mtx_unloadChunk;



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

struct LaszipVlrItem{
	int type = 0;
	int size = 0;
	int version = 0;
};

struct LaszipVlr{
	int compressor = 0;
	int coder = 0;
	int versionMajor = 0;
	int versionMinor = 0;
	int versionRevision = 0;
	int options = 0;
	int chunkSize = 0;
	int numberOfSpecialEvlrs = 0;
	int offsetToSpecialEvlrs = 0;
	int numItems = 0;
	vector<LaszipVlrItem> items;
};

LaszipVlr parseLaszipVlr(shared_ptr<Buffer> buffer){
	LaszipVlr vlr;

	vlr.compressor = buffer->get<uint16_t>(0);
	vlr.coder = buffer->get<uint16_t>(2);
	vlr.versionMajor = buffer->get<uint8_t>(4);
	vlr.versionMinor = buffer->get<uint8_t>(5);
	vlr.versionRevision = buffer->get<uint16_t>(6);
	vlr.options = buffer->get<uint32_t>(8);
	vlr.chunkSize = buffer->get<uint32_t>(12);
	vlr.numberOfSpecialEvlrs = buffer->get<uint64_t>(16);
	vlr.offsetToSpecialEvlrs = buffer->get<uint64_t>(24);
	vlr.numItems = buffer->get<uint16_t>(32);

	for(int i = 0; i < vlr.numItems; i++){
		LaszipVlrItem item = {
			.type = buffer->get<uint16_t>(34 + i * 6 + 0),
			.size = buffer->get<uint16_t>(34 + i * 6 + 2),
			.version = buffer->get<uint16_t>(34 + i * 6 + 4),
		};

		vlr.items.push_back(item);
	}

	return vlr;
}


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
	uniforms.numTiles               = tiles.size();
	uniforms.numChunks              = chunks.size();

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
		&cptr_buffer,
		&uniforms, 
		&output_surf,
		&cptr_stats,
		&cptr_tiles, &cptr_chunks,
		&cptr_commandsQueue, &cptr_commandsQueueCounter,
		&cptr_chunksToLoad, &cptr_numChunksToLoad
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

// void reset(){

// 	int workgroupSize = 128;

// 	auto& kernel = cuda_program->kernels["kernel_init"];

// 	int maxActiveBlocksPerSM;
// 	cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, 
// 		kernel, workgroupSize, 0);
	
// 	int numGroups = maxActiveBlocksPerSM * numSMs;
	
// 	void* args[] = {
// 		&cptr_buffer,
// 		&cptr_stats,
// 		&cptr_tiles, &cptr_chunks,
// 		&cptr_commandsQueue, &cptr_commandsQueueCounter
// 	};

// 	auto res_launch = cuLaunchCooperativeKernel(kernel,
// 		numGroups, 1, 1,
// 		workgroupSize, 1, 1,
// 		0, 0, args);

// 	if(res_launch != CUDA_SUCCESS){
// 		const char* str; 
// 		cuGetErrorString(res_launch, &str);
// 		printf("error: %s \n", str);
// 	}

// }

// compile kernels and allocate buffers
void initCudaProgram(shared_ptr<GLRenderer> renderer){

	uint64_t cptr_buffer_bytes       = 1'000'000'000;

	cuMemAlloc(&cptr_buffer                , cptr_buffer_bytes);
	// cuMemAlloc(&cptr_points                , 8'000'000'000); // ~500M Points
	cuMemAlloc(&cptr_tiles                 , 100'000 * sizeof(Tile));    // ~5MB
	cuMemAlloc(&cptr_chunks                , 1'000'000 * sizeof(Chunk)); // ~50MB, should be good for 50 billion points
	cuMemAlloc(&cptr_commandsQueue         , COMMAND_QUEUE_CAPACITY * sizeof(Command)); 
	cuMemAlloc(&cptr_commandsQueueCounter  , 8);

	cuMemAlloc(&cptr_chunksToLoad  , 4 * MAX_CHUNKS_TO_LOAD);
	cuMemAlloc(&cptr_numChunksToLoad  , 8);

	cuMemAlloc(&cptr_stats                 , sizeof(Stats));
	cuMemAllocHost((void**)&h_stats_pinned , sizeof(Stats));
	cuMemAllocHost((void**)&h_commandQueue_pinned, COMMAND_QUEUE_CAPACITY * sizeof(Command));
	cuMemAllocHost((void**)&h_commandQueueCounter_pinned, 8);

	cuMemAllocHost((void**)&h_chunksToLoad_pinned, 4 * MAX_CHUNKS_TO_LOAD);
	cuMemAllocHost((void**)&h_numChunksToLoad_pinned, 8);

	uint32_t maxint = -1;
	cuMemsetD32(cptr_chunksToLoad, maxint, MAX_CHUNKS_TO_LOAD);
	for (int i = 0; i < MAX_CHUNKS_TO_LOAD; i++) {
		int32_t* chunksToLoad = (int32_t*)h_chunksToLoad_pinned;
		chunksToLoad[i] = -1;
	}


	uint64_t zero_u64 = 0;
	cuMemcpyHtoD(cptr_commandsQueueCounter, &zero_u64, 8);
	memcpy(h_commandQueueCounter_pinned, &zero_u64, 8);
	// memset(h_chunksToLoad_pinned, 0, 4 * MAX_CHUNKS_TO_LOAD);
	memcpy(h_numChunksToLoad_pinned, &zero_u64, 8);
	

	printfmt("sizeof(Tile): {} \n", sizeof(Tile));
	printfmt("sizeof(Chunk): {} \n", sizeof(Chunk));

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/progressive_octree/render.cu",
			"./modules/progressive_octree/utils.cu",
		},
		.kernels = {
			"kernel_render",
			"kernel_init",
			"kernel_chunkLoaded",
			"kernel_chunkUnloaded"
		}
	});

	cuEventCreate(&ce_render_start, 0);
	cuEventCreate(&ce_render_end, 0);
	cuEventCreate(&ce_update_start, 0);
	cuEventCreate(&ce_update_end, 0);
	
	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	{ // run init cuda program
		int workgroupSize = 128;
		auto& kernel = cuda_program->kernels["kernel_init"];
		int maxActiveBlocksPerSM;
		cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, kernel, workgroupSize, 0);
		int numGroups = maxActiveBlocksPerSM * numSMs;
		
		void* args[] = {
			&cptr_buffer, &cptr_stats,
			&cptr_tiles, &cptr_chunks,
			&cptr_commandsQueue, &cptr_commandsQueueCounter
		};

		auto res_launch = cuLaunchCooperativeKernel(kernel,
			numGroups, 1, 1,
			workgroupSize, 1, 1,
			0, 0, args);

		if(res_launch != CUDA_SUCCESS){
			const char* str; 
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
		}
	}
}

void spawnLoader(){

	thread t([&](){

		while (true) {

			mtx_loadChunk.lock();
			
			if(tasks_loadChunk.size() > 0){

				// Retrieve Load Task
				Task_LoadChunk task = tasks_loadChunk.front();
				tasks_loadChunk.pop_front();

				mtx_loadChunk.unlock();

				// Load chunk data
				Tile tile = tiles[task.tileID];
				Chunk& chunk = chunks[task.chunkID];
				Chunk_HostData& hostData = chunks_hostData[task.chunkID];
				string file = tilePaths[task.tileID];

				uint32_t firstPoint = task.chunkIndex * 50'000;
				int numPoints = min((tile.numPoints - firstPoint), 50'000u);

				LasHeader header = loadHeader(file);
				auto buffer = make_shared<Buffer>(header.bytesPerPoint * numPoints);

				double translation[3] = {-header.min[0], -header.min[1], -header.min[2]};
				loadLasNative(file, header, firstPoint, numPoints, buffer->data, translation);

				// Create Upload Task
				Task_UploadChunk task_upload = {
					.tileID = task.tileID,
					.chunkIndex = task.chunkIndex,
					.chunkID = task.chunkID,
					.points = buffer,
					.numPoints = numPoints,
				};

				mtx_uploadChunk.lock();
				mtx_loadChunk.lock();

				tasks_uploadChunk.push_back(task_upload);

				hostData.isLoading = false;
				hostData.isLoaded = true;

				mtx_loadChunk.unlock();
				mtx_uploadChunk.unlock();
				
			}else{
				mtx_loadChunk.unlock();
			}

			this_thread::sleep_for(1ms);
		}
		
	});

	t.detach();

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

	numThreads = 32;
	for(int i = 0; i < numThreads; i++){
		spawnLoader();
	}

	// reset();


	{ // test data

		// vector<string> files = listFiles("E:/resources/pointclouds/CA13_las");
		vector<string> files = listFiles("D:/resources/pointclouds/CA13_las_tmp");


		float3 acc_min = { Infinity, Infinity, Infinity };
		float3 acc_max = { -Infinity, -Infinity, -Infinity };

		// for(int tileIndex = 0; tileIndex < files.size(); tileIndex++)
		 for(int tileIndex = 0; tileIndex < min(int(files.size()), 3000); tileIndex++)
		{
			string file = files[tileIndex];
			
			if(!iEndsWith(file, "las")) continue;


			auto buffer = readBinaryFile(file, 0, 4096);

			uint32_t color = tileIndex * 12345678;

			uint32_t offsetToPointData = buffer->get<uint32_t>(96);
			uint32_t offsetToPointData_fixed = offsetToPointData; // must subtract laszip vlr
			uint32_t pointFormat = buffer->get<uint8_t>(104) % 128;
			uint32_t offset_rgb = 0;
			uint16_t headerSize = buffer->get<uint16_t>(94);
			uint32_t numVLRs = buffer->get<uint32_t>(100);
			uint32_t numPoints = buffer->get<uint32_t>(107);

			{
				//vector<VLR> vlrs;
				uint64_t byteOffset = headerSize;
				for (int vlr_index = 0; vlr_index < numVLRs; vlr_index++) {

					uint64_t offset_vlrHeaderBuffer = byteOffset;

					// let vlrHeaderBuffer = Buffer.alloc(54);
					// await handle.read(vlrHeaderBuffer, 0, vlrHeaderBuffer.byteLength, byteOffset);

					// console.log("vlr");

					// let userId = parseCString(vlrHeaderBuffer.slice(2, 18));
					int recordId = buffer->get<uint16_t>(byteOffset + 18);
					int recordLength = buffer->get<uint16_t>(byteOffset + 20);
					bool isLaszipVlr = recordId == 22204;

					LaszipVlr vlr;

					if(isLaszipVlr){
						shared_ptr<Buffer> vlrContentBuffer = make_shared<Buffer>(recordLength);
						memcpy(vlrContentBuffer->data, buffer->data_u8 + byteOffset + 54, recordLength);
						vlr = parseLaszipVlr(vlrContentBuffer);

						offsetToPointData_fixed = offsetToPointData_fixed - recordLength - 54;
					}

					byteOffset = byteOffset + 54 + recordLength;
				}

				//offsetToPointData = offsetToPointData_fixed;

				//uint64_t chunkTableStart = buffer->get<uint64_t>(byteOffset);
				//uint64_t chunkTableSize = fs::file_size(file) - chunkTableStart;
				//shared_ptr<Buffer> chunkTableBuffer = readBinaryFile(file, chunkTableStart, chunkTableSize);

				//uint32_t version = chunkTableBuffer->get<uint32_t>(0);
				//uint32_t numChunks = chunkTableBuffer->get<uint32_t>(4);

				// let dec = new ArithmeticDecoder(chunkTableBuffer, 8);
				// let ic = new IntegerCompressor(dec, 32, 2);

				// ByteStreamInArray* streamin = new ByteStreamInArray(data, size);
				// ArithmeticDecoder dc;
				// IntegerCompressor ic;

			}

			if(pointFormat == 2) offset_rgb = 20;
			if(pointFormat == 3) offset_rgb = 28;

			if(offsetToPointData < 4000){
				int R = buffer->get<uint16_t>(offsetToPointData + offset_rgb + 0);
				int G = buffer->get<uint16_t>(offsetToPointData + offset_rgb + 2);
				int B = buffer->get<uint16_t>(offsetToPointData + offset_rgb + 4);

				int r = R < 256 ? R : R / 256;
				int g = G < 256 ? G : G / 256;
				int b = B < 256 ? B : B / 256;

				color = r | (g << 8) | (b << 16);
			}

			float3 bbm = {734074.53, 3889472.73, 236.21};

			Tile tile;
			tile.min = float3{
				float(buffer->get<double>(187)) - bbm.x,
				float(buffer->get<double>(203)) - bbm.y,
				float(buffer->get<double>(219)) - bbm.z
			};
			tile.max = float3{
				float(buffer->get<double>(179)) - bbm.x,
				float(buffer->get<double>(195)) - bbm.y,
				float(buffer->get<double>(211)) - bbm.z
			};

			acc_min.x = min(acc_min.x, tile.min.x);
			acc_min.y = min(acc_min.y, tile.min.y);
			acc_min.z = min(acc_min.z, tile.min.z);
			acc_max.x = max(acc_max.x, tile.max.x);
			acc_max.y = max(acc_max.y, tile.max.y);
			acc_max.z = max(acc_max.z, tile.max.z);

			tile.color = color;
			tile.numPoints = numPoints;
			tile.numPointsLoaded = 0;
			tile.state = STATE_EMPTY;

			tiles.push_back(tile);
			tilePaths.push_back(file);

			int chunkSize = 50'000;
			int numChunks = (numPoints + chunkSize - 1) / chunkSize;

			for(int chunkIndex = 0; chunkIndex < numChunks; chunkIndex++){
				Chunk chunk;
				chunk.min = tile.min;
				chunk.max = tile.max;
				chunk.chunkIndex = chunkIndex;
				chunk.color = 0x000000ff;
				chunk.tileID = tileIndex;
				chunk.numPoints = min(numPoints - (chunkIndex * chunkSize), uint32_t(chunkSize));
				chunk.numPointsLoaded = 0;
				chunk.state = STATE_EMPTY;
				
				chunks.push_back(chunk);

				Chunk_HostData hostData = {
					.tileID       = uint32_t(tileIndex),
					.chunkIndex   = uint32_t(chunkIndex),
					.isLoading    = false,
					.isLoaded     = false,
				};
				chunks_hostData.push_back(hostData);
			}
		}

		float3 diff = {
			acc_max.x - acc_min.x,
			acc_max.y - acc_min.y,
			acc_max.z - acc_min.z
		};
		float dist = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
		// renderer->controls->yaw = -1.15;
		// renderer->controls->pitch = -0.57;
		// renderer->controls->radius = dist;
		// renderer->controls->target = { 0.0f, 0.0f, 0.0f };

		// position: -7449.509369996216, 7225.290090406565, 1250.6768786203359 
		renderer->controls->yaw    = 0.543;
		renderer->controls->pitch  = -0.647;
		renderer->controls->radius = 5061.712;
		renderer->controls->target = { -10062.763, 10682.825, -1364.143, };

		cuMemcpyHtoD(cptr_tiles, tiles.data(), tiles.size() * sizeof(Tile));
		cuMemcpyHtoD(cptr_chunks, chunks.data(), chunks.size() * sizeof(Chunk));

	}

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


		// check pending command loads
		for(int i = 0; i < pendingCommandLoads.size(); i++){

			auto& pending = pendingCommandLoads.front();

			bool eventFinished = cuEventQuery(pending.ce_ranges_loaded) == CUDA_SUCCESS;

			if(eventFinished){
				pendingCommandLoads.pop_front();
				cuEventDestroy(pending.ce_ranges_loaded);

				// now let's check out these commands
				for(int commandIndex = pending.start_0; commandIndex < pending.end_0; commandIndex++){
					Command* commands = (Command*)h_commandQueue_pinned;

					Command* command = &commands[commandIndex % COMMAND_QUEUE_CAPACITY];
					
					if(command->command == CMD_READ_CHUNK){
						// CommandReadChunkData* data = (CommandReadChunkData*)command->data;

						// Task_LoadChunk task;
						// task.tileID = data->tileID;
						// task.chunkID = data->chunkID;
						// task.chunkIndex = data->chunkIndex;

						// // lock_guard<mutex> lock(mtx_loadChunk);
						// mtx_loadChunk.lock();
						// tasks_loadChunk.push_back(task);
						// mtx_loadChunk.unlock();
					}else if(command->command == CMD_UNLOAD_CHUNK){
						CommandUnloadChunkData* data = (CommandUnloadChunkData*)command->data;

						Task_UnloadChunk task = {
							.tileID     = int(data->tileID),
							.chunkIndex = int(data->chunkIndex),
							.chunkID    = int(data->chunkID),
							.cptr       = data->cptr_pointBatch,
						};

						mtx_unloadChunk.lock();
						tasks_unloadChunk.push_back(task);
						mtx_unloadChunk.unlock();

						
					}
				}
			}else{
				// pending events should finish in sequence, so if we encounter one that has not finished,
				// we can stop checking the other ones.
				break;
			}
		}

		{ // Deallocate/Unload least important chunks
			lock_guard<mutex> lock(mtx_unloadChunk);

			auto& kernel = cuda_program->kernels["kernel_chunkUnloaded"];

			for(int i = 0; i < tasks_unloadChunk.size(); i++){
				Task_UnloadChunk task = tasks_unloadChunk[i];

				chunks_hostData[task.chunkID].isLoaded = false;
				chunks_hostData[task.chunkID].isLoading = false;

				cuMemFree((CUdeviceptr)task.cptr);
				
				{ // invoke chunkUnloaded kernel
					uint32_t chunkID = task.chunkID;

					void* args[] = {&chunkID, &cptr_chunks};
					auto res_launch = cuLaunchCooperativeKernel(kernel,
						1, 1, 1,
						1, 1, 1,
						0, 0, args);

					if(res_launch != CUDA_SUCCESS){
						const char* str; 
						cuGetErrorString(res_launch, &str);
						printf("error: %s \n", str);
					}
				}
			}

			tasks_unloadChunk.clear();
		}

		// if(frameCounter % 100 == 0)
		{
			int32_t* chunksToLoad = (int32_t*)h_chunksToLoad_pinned;
			bool encounteredEmpty = false;

			mtx_loadChunk.lock();

			for(int i = 0; i < tasks_loadChunk.size(); i++){
				auto task = tasks_loadChunk[i];

				chunks_hostData[task.chunkID].isLoading = false;
			}

			tasks_loadChunk.clear();

			if(chunksToLoad[0] != -1)
			for(int i = 0; i < MAX_CHUNKS_TO_LOAD; i++){

				int value = chunksToLoad[i];

				// if(i < 10){
				// 	cout << value << ", ";
				// }else if(i == 10){
				// 	cout << " ... ";
				// }

				if(!encounteredEmpty){
					encounteredEmpty = value == -1;
				}

				if (encounteredEmpty) {
					// cout << " [" << i << "]" << endl;
					break;
				}

				Chunk& chunk = chunks[value];
				Chunk_HostData& hostData = chunks_hostData[value];

				if (hostData.isLoaded) continue;
				if (hostData.isLoading) continue;

				Task_LoadChunk task;
				task.tileID = chunk.tileID;
				task.chunkID = value;
				task.chunkIndex = chunk.chunkIndex;

				tasks_loadChunk.push_back(task);

				hostData.isLoading = true;
			}

			mtx_loadChunk.unlock();
		}

		for(int i = 0; i < 100; i++)
		{// upload chunks that finished loading from file

			mtx_uploadChunk.lock();
			
			if(tasks_uploadChunk.size() > 0){
				Task_UploadChunk task = tasks_uploadChunk.front();
				tasks_uploadChunk.pop_front();

				mtx_uploadChunk.unlock();

				Tile tile = tiles[task.tileID];
				Chunk chunk = chunks[task.chunkID];
				string file = tilePaths[task.tileID];

				auto buffer = task.points;

				CUdeviceptr cptr_batch;
				cuMemAlloc(&cptr_batch, 50'000 * sizeof(Point));
				cuMemcpyHtoD(cptr_batch, buffer->data, task.numPoints * sizeof(Point));

				{ // invoke chunkLoaded kernel 
					int workgroupSize = 128;
					auto& kernel = cuda_program->kernels["kernel_chunkLoaded"];
					int maxActiveBlocksPerSM;
					cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, kernel, workgroupSize, 0);
					int numGroups = maxActiveBlocksPerSM * numSMs;
					
					uint32_t chunkIndex = chunk.chunkIndex;
					uint32_t chunkID = task.chunkID;
					uint64_t ptr_points = (uint64_t)cptr_batch;

					Point point;
					point.x = buffer->get<float>(0);
					point.y = buffer->get<float>(4);
					point.z = buffer->get<float>(8);

					void* args[] = {
						&cptr_buffer,
						&chunkIndex, &chunkID, &ptr_points, &cptr_tiles, &cptr_chunks
					};

					auto res_launch = cuLaunchCooperativeKernel(kernel,
						numGroups, 1, 1,
						workgroupSize, 1, 1,
						0, 0, args);

					if(res_launch != CUDA_SUCCESS){
						const char* str; 
						cuGetErrorString(res_launch, &str);
						printf("error: %s \n", str);
					}
				
				}
			}else{
				mtx_uploadChunk.unlock();
			}
		}
		

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

			// async copy chunks that we should load
			cuMemcpyDtoHAsync(h_chunksToLoad_pinned, cptr_chunksToLoad, 4 * MAX_CHUNKS_TO_LOAD, ((CUstream)CU_STREAM_DEFAULT));
			// memcpy(&stats, h_stats_pinned, sizeof(Stats));

			// async copy commands
			cuMemcpyDtoHAsync(h_commandQueueCounter_pinned, cptr_commandsQueueCounter, 8, ((CUstream)CU_STREAM_DEFAULT));

			uint64_t commandQueueCounter = *((uint64_t*)h_commandQueueCounter_pinned);
			uint64_t commandsToLoadFromDevice = commandQueueCounter - commandsLoadedFromDevice;

			if(commandsToLoadFromDevice > 0){

				CUevent ce_ranges_loaded;
				cuEventCreate(&ce_ranges_loaded, 0);
				

				// command queue is a ring buffer, so we may have to load two individual ranges
				uint64_t start_0 = commandsLoadedFromDevice % COMMAND_QUEUE_CAPACITY;
				uint64_t end_0 = commandsLoadedFromDevice + commandsToLoadFromDevice;
				uint64_t start_1 = 0;
				uint64_t end_1 = 0;

				if(end_0 > COMMAND_QUEUE_CAPACITY){
					start_1 = 0;
					end_1 = end_0 % COMMAND_QUEUE_CAPACITY;
					end_0 = COMMAND_QUEUE_CAPACITY;
				}

				cuMemcpyDtoHAsync(
					((uint8_t*)h_commandQueue_pinned) + start_0 * sizeof(Command), 
					cptr_commandsQueue + start_0 * sizeof(Command), 
					(end_0 - start_0) * sizeof(Command), 
					((CUstream)CU_STREAM_DEFAULT));

				if(start_1 != end_1){
					cuMemcpyDtoHAsync(
						((uint8_t*)h_commandQueue_pinned) + start_1 * sizeof(Command), 
						cptr_commandsQueue + start_1 * sizeof(Command), 
						(end_1 - start_1) * sizeof(Command), 
						((CUstream)CU_STREAM_DEFAULT));
				}

				cuEventRecord(ce_ranges_loaded, 0);

				PendingCommandLoad pending;
				pending.start_0 = start_0;
				pending.end_0 = end_0;
				pending.start_1 = start_1;
				pending.end_1 = end_1;
				pending.ce_ranges_loaded = ce_ranges_loaded;

				pendingCommandLoads.push_back(pending);


				 printfmt("load {} to {} \n", start_0, end_0);
				 if(start_1 != end_1){
				 	printfmt("also load {} to {} \n", start_1, end_1);
				 }

				commandsLoadedFromDevice = commandQueueCounter;
			}

			



			statsAge = static_cast<int>(renderer->frameCount) - stats.frameID;
		}

		{ // DRAW GUI

			if(Runtime::showGUI)
			{ // RENDER IMGUI SETTINGS WINDOW

				auto windowSize = ImVec2(490, 280);
				ImGui::SetNextWindowPos(ImVec2(10, 300));
				ImGui::SetNextWindowSize(windowSize);

				ImGui::Begin("Settings");
				
				// ImGui::Checkbox("Show Bounding Box",     &settings.showBoundingBox);

				ImGui::Checkbox("Update Visibility",     &settings.doUpdateVisibility);

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

				auto locale = getSaneLocale();

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

				double M = 1'000'000.0;
				double B = 1'000'000'000.0;
				double MB = 1'000'000.0; // TIL: MB = 1'000'000 vs. MiB = 1024 * 1024
				double GB = 1'000'000'000.0;

				uint64_t commandQueueCounter = *((uint64_t*)h_commandQueueCounter_pinned);

				vector<vector<string>> table = {
					{"test  ", toMS(123.0f)   , format("{:.1f}", 123.0f)},
					{"commandQueueCounter  ", toIntString(commandQueueCounter)   , format("{:.1f}", 123.0f)}
				};

				if (ImGui::Button("Copy Stats")) {
					stringstream ss;
					for (int row = 0; row < table.size(); row++) {
						for (int column = 0; column < 2; column++) {
							ss << table[row][column];
						}
						ss << "\n";
					}

					string str = ss.str();
					toClipboard(str);
				}

				auto flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV;
				if (ImGui::BeginTable("table1", 3, flags)) {
					ImGui::TableSetupColumn("AAA", ImGuiTableColumnFlags_WidthStretch);
					ImGui::TableSetupColumn("BBB", ImGuiTableColumnFlags_WidthStretch);
					ImGui::TableSetupColumn("CCC", ImGuiTableColumnFlags_WidthFixed);
					for (int row = 0; row < table.size(); row++) {
						ImGui::TableNextRow();
						for (int column = 0; column < 2; column++) {
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

		
		}

		frameCounter++;
	};

	renderer->loop(update, render);

	return 0;
}