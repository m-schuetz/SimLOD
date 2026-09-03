

#include <iostream>
#include <filesystem>
#include<locale.h>
#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <format>

#include "CudaModularProgram.h"

#include "unsuck.hpp"

using namespace std;

CUdeviceptr cptr_buffer, cptr_input;
CudaModularProgram* cuda_program = nullptr;

// not actually used by the kernel, 
vector<int> input = {81, 23, 25, 21, 73, 7, 15, 17, 29, 11, 6, 73, 84, 21, 59, 61, 60, 90, 20, 74, 12, 43, 19, 55, 25, 36, 98, 16, 31, 60, 48, 49, 55, 34, 63, 24, 18, 94, 39, 78, 91, 16, 57, 27, 86, 82, 16, 66, 6, 87, 79, 46, 83, 85, 64, 87, 8, 78, 95, 2, 42, 42, 95, 31, 14, 23, 53, 79, 94, 24, 13, 81, 95, 96, 7, 29, 82, 91, 3, 68, 74, 63, 61, 2, 8, 83, 52, 17, 12, 4, 35, 54, 85, 40, 43, 75, 99, 27, 46, 4, 97, 82, 17, 28, 26, 61, 37, 29, 66, 98, 55, 53, 39, 60, 50, 38, 3, 44, 49, 10, 57, 89, 97, 3, 38, 85, 86, 76, 75, 65, 95, 24, 26, 97, 91, 60, 83, 47, 19, 74, 55, 55, 49, 97, 83, 41, 7, 57, 91, 52, 35, 18, 37, 54, 59, 71, 75, 3, 44, 0, 66, 77, 16, 99, 31, 36, 67, 96, 55, 69, 39, 1, 18, 90, 95, 59, 5, 53, 8, 13, 90, 8, 34, 65, 93, 92, 92, 62, 99, 32, 22, 48, 72, 75, 91, 20, 83, 71, 43, 91, 13, 78, 29, 63, 65, 82, 57, 73, 22, 74, 57, 18, 58, 51, 69, 40, 54, 16, 69, 22, 78, 69, 95, 28, 57, 61, 4, 60, 66, 55, 10, 94, 33, 73, 29, 29, 56, 8, 3, 24, 49, 20, 76, 68, 52, 18, 2, 87, 28, 79, 37, 95, 82, 4, 37, 15, 15, 92, 46, 3, 43, 74, 29, 9, 74, 65, 97, 22, 7, 99, 73, 45, 78, 92, 75, 53, 59, 77, 6, 98, 24, 82, 67, 47, 32, 43, 3, 62, 99, 74, 95, 33, 0, 81, 7, 55, 34, 23, 70, 97, 66, 20, 33, 57, 52, 18, 75, 91, 12, 93, 66, 89, 84, 81, 86, 54, 44, 12, 35, 67, 3, 7, 91, 83, 66, 36, 49, 4, 52, 9, 37, 9, 34, 30, 8, 79, 34, 47, 82, 23, 76, 14, 14, 67, 74, 92, 10, 34, 68, 76, 57, 13, 65, 54, 23, 60, 76, 85, 97, 29, 0, 73, 0, 13, 92, 94, 66, 21, 58, 88, 41, 69, 16, 53, 66, 16, 16, 57, 68, 83, 37, 63, 92, 6, 58, 58, 50, 97, 76, 88, 30, 10, 48, 54, 92, 77, 65, 68, 86, 89, 38, 28, 79, 94, 32, 22, 50, 77, 56, 73, 44, 3, 79, 56, 60, 60, 5, 62, 94, 31, 46, 54, 20, 29, 32, 56, 68, 84, 75, 61, 58, 87, 53, 41, 82, 95, 39, 26, 15, 23, 68, 46, 55, 69, 46, 80, 27, 16, 6, 19, 98, 60, 58, 77, 55, 8, 52, 35, 92, 95, 82, 89, 43, 37, 94, 49, 14, 82, 6, 79, 71, 17, 57, 46, 38, 24, 48, 48, 99, 54, 30, 77, 23, 71, 54, 4, 22, 66, 14, 33, 4, 12, 37, 24, 89, 6, 88, 7, 72, 22, 55, 61, 61, 78, 64, 64, 47, 5, 0, 28, 46, 5, 67, 48, 88, 12, 73, 20, 93, 73, 41, 49, 78, 25, 51, 76, 19, 61, 71, 9, 43, 52, 80, 31, 11, 73, 64, 43, 32, 69, 86, 10, 72, 35, 0, 43, 24, 27, 29, 95, 90, 23, 92, 1, 55, 54, 73, 82, 75, 70, 59, 82, 85, 55, 32, 17, 39, 79, 68, 96, 34, 72, 1, 20, 73, 98, 62, 37, 28, 79, 85, 44, 43, 67, 37, 90, 48, 31, 76, 32, 46, 68, 25, 5, 79, 91, 39, 91, 36, 61, 2, 39, 56, 43, 13, 32, 1, 7, 75, 79, 41, 56, 78, 72, 59, 49, 6, 93, 93, 39, 3, 47, 38, 46, 76, 59, 45, 92, 19, 58, 6, 31, 5, 96, 26, 88, 63, 82, 78, 60, 68, 60, 89, 18, 68, 98, 33, 63, 69, 8, 40, 24, 63, 14, 40, 36, 3, 46, 5, 72, 82, 61, 40, 20, 78, 95, 60, 60, 65, 97, 57, 49, 99, 91, 48, 53, 90, 60, 43, 5, 87, 57, 25, 13, 62, 62, 6, 86, 21, 35, 5, 68, 3, 53, 24, 85, 44, 88, 8, 34, 69, 1, 95, 16, 14, 22, 91, 43, 68, 29, 74, 91, 38, 26, 95, 6, 35, 59, 47, 8, 11, 44, 74, 32, 42, 27, 13, 60, 65, 92, 62, 44, 2, 69, 4, 34, 71, 12, 72, 26, 73, 23, 82, 27, 43, 74, 23, 88, 63, 94, 6, 50, 25, 64, 73, 63, 33, 87, 85, 23, 29, 72, 1, 19, 16, 21, 45, 87, 66, 43, 72, 7, 72, 65, 80, 5, 88, 14, 37, 61, 32, 11, 9, 24, 4, 40, 79, 91, 31, 30, 24, 40, 47, 34, 5, 15, 7, 31, 16, 48, 90, 29, 49, 55, 52, 70, 78, 85, 19, 86, 14, 16, 55, 30, 68, 5, 95, 68, 27, 61, 24, 76, 15, 9, 55, 69, 74, 19, 22, 25, 34, 29, 74, 33, 42, 65, 51, 79, 44, 76, 66, 50, 10, 83, 39, 29, 14, 75, 0, 31, 18, 20, 56, 57, 18, 4, 97, 52, 54, 22, 88, 90, 24, 18, 60, 20, 22, 58, 40, 97, 60, 2, 65, 34, 4, 93, 3, 39, 37, 19, 20, 95, 40, 34, 92, 95, 10, 81, 82, 73, 24, 98, 59, 80, 70, 23, 79, 51, 23, 82, 37, 67, 82, 9, 89, 47, 3, 39, 46, 26, 80, 65, 40, 84, 83, 76, 21, 5, 12, 47, 90, 2, 23, 38, 30, 47, 6, 12, 11, 95, 46, 59, 2, 0, 69, 65, 63, 41, 83, 9, 16, 79, 66, 13, 10, 91, 42, 63, 43, 77, 60, 87, 28, 52, 23, 38, 26, 94, 78, 10, 24, 12, 27, 85, 97, 1, 63, 27, 91, 79, 29, 43, 84, 66, 25, 61, 70, 6, 95, 71, 16, 96, 96, 80, 28, 76, 16, 57, 53, 48, 11, 83, 53, 28, 75, 28, 13, 17, 92, 27};

void initCuda(){
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);
}

void runCudaProgram(){

	cout << "================================================================================" << endl;
	cout << "=== RUNNING" << endl;
	cout << "================================================================================" << endl;

	CUresult resultcode = CUDA_SUCCESS;
	CUevent cevent_start, cevent_end;
	cuEventCreate(&cevent_start, 0);
	cuEventCreate(&cevent_end, 0);

	CUdevice device;
	int numSMs;
	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

	int workgroupSize = 256;

	int numGroups;
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&numGroups, cuda_program->kernels["kernel"], workgroupSize, 0);
	numGroups *= numSMs;
	
	// make sure at least 10 workgroups are spawned)
	numGroups = std::clamp(numGroups, 10, numGroups);

	cuEventRecord(cevent_start, 0);

	void* args[] = {&cptr_buffer, &cptr_input};

	auto res_launch = cuLaunchCooperativeKernel(cuda_program->kernels["kernel"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str; 
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(cevent_end, 0);
	cuEventSynchronize(cevent_end);

	{
		float total_ms;
		cuEventElapsedTime(&total_ms, cevent_start, cevent_end);

		cout << "CUDA durations: " << endl;
		cout << std::format("total:     {:6.1f} ms", total_ms) << endl;
	}

	cuCtxSynchronize();

	Buffer buffer(100);
	cuMemcpyDtoH(buffer.data, cptr_buffer, buffer.size);

	cout << "print generated random values from host: ";
	for(int i = 0; i < 10; i++){
		cout << buffer.get<uint32_t>(4 * i) << ", ";
	}
	cout << " ..." << endl;

}

void initCudaProgram(){

	cuMemAlloc(&cptr_buffer, 100'000'000);
	cuMemAlloc(&cptr_input, 100'000'000);

	cuMemcpyHtoD(cptr_input, input.data(), input.size() * sizeof(int));

	cuda_program = new CudaModularProgram({
		.modules = {
			"./modules/randomNumbers/randomNumbers.cu",
			"./modules/randomNumbers/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuda_program->onCompile([&](){
		runCudaProgram();
	});
}



int main(){

	cout << std::setprecision(2) << std::fixed;
	setlocale( LC_ALL, "en_AT.UTF-8" );

	initCuda();
	initCudaProgram();
	runCudaProgram();

	while(true){
		EventQueue::instance->process();

		std::this_thread::sleep_for(1ms);
	}

	return 0;
}
