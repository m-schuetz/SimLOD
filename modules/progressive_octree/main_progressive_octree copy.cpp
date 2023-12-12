#define NOMINMAX

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

#include "windows.h"

#include "unsuck.hpp"

using namespace std;
namespace fs = std::filesystem;

static thread_local bool initialized = false;
static thread_local void* thread_buffer = nullptr;
static thread_local uint64_t thread_buffer_size = 0;
static thread_local bool isLoadPending = false;
static thread_local bool shouldRetry = false;
static thread_local bool isLoadDone = false;

VOID CALLBACK FileIOCompletionRoutine(
	__in DWORD errorCode,
	__in DWORD numBytesTransfered,
	__in LPOVERLAPPED lpOverlapped)
{

	if (errorCode != 0) {
		printfmt("read error: {} \n", errorCode);
	}

	if (numBytesTransfered == 0) {
		printfmt("0 bytes read. offset: {:14L}, numBytes: {:9} \n", lpOverlapped->Offset, numBytesTransfered);

		isLoadPending = false;
		shouldRetry = true;
		isLoadDone = false;
	}
	else {
		// string filename = fs::path(dbg_file).filename().string();

		isLoadPending = false;
		shouldRetry = false;
		isLoadDone = true;

		// printfmt("loaded\n");
	}
}

void loadFile(string file, uint64_t offset, uint64_t size){

	uint64_t sectorSize = getPhysicalSectorSize(file);
	uint64_t requiredThreadBufferSize = 10 * sectorSize + size;

	if(!initialized){
		thread_buffer = malloc(requiredThreadBufferSize);
		thread_buffer_size = requiredThreadBufferSize;
		initialized = true;
	}else if(thread_buffer_size < requiredThreadBufferSize){
		free(thread_buffer);
		thread_buffer = malloc(requiredThreadBufferSize);
		thread_buffer_size = requiredThreadBufferSize;
	}

	isLoadPending = false;
	shouldRetry = false;
	isLoadDone = false;

	uint64_t firstByte = offset;
	uint64_t numBytes = size;
	uint64_t endByte = firstByte + numBytes;
	uint64_t firstByte_aligned = firstByte - firstByte % sectorSize;
	uint64_t endByte_aligned = endByte;
	if((endByte % sectorSize) != 0){
		endByte_aligned = endByte - (endByte % sectorSize) + sectorSize;
	}
	uint64_t numBytes_aligned = endByte_aligned - firstByte_aligned;
	
	LPTSTR winPath = new TCHAR[1000];
	std::copy(file.begin(), file.end(), winPath);
	winPath[file.length()] = 0;

	HANDLE hFile = CreateFile(winPath,
		GENERIC_READ,
		FILE_SHARE_READ,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
		NULL);

	do{

		// vital because ReadFileEx calls the current thread when it's sleeping
		SleepEx(1, TRUE);

		if(isLoadPending) continue;
		if(isLoadDone) break;

		OVERLAPPED ol = {0};
		ol.Offset     = (firstByte_aligned >>  0llu) & 0xffffffffllu;
		ol.OffsetHigh = (firstByte_aligned >> 32llu) & 0xffffffffllu;

		isLoadPending = true;

		auto returnValue = ReadFileEx(
			hFile, 
			(char*)thread_buffer, 
			numBytes_aligned, 
			&ol, 
			FileIOCompletionRoutine
		);

		if (returnValue == 0) {
			printfmt("ERROR: ReadFileEx failed \n");

			isLoadPending = false;
			isLoadDone = false;
		}
		
		SleepEx(1, TRUE);

	}while(shouldRetry || isLoadPending);

	CloseHandle(hFile);

	isLoadPending = false;
	shouldRetry = false;
	isLoadDone = false;

	uint64_t padding = firstByte - firstByte_aligned;

	uint8_t* source = ((uint8_t*)thread_buffer) + padding;
	uint32_t numPoints;
	memcpy(&numPoints, source + 107, 4);

	 //printfmt("{} points in {}\n", numPoints, file);
}

int main(){

	string dir = "E:/resources/pointclouds/CA13";

	mutex mtx_files;
	vector<string> files;

	bool requestShutdown = false;

	int numThreads = 512;
	atomic_uint32_t numActiveThreads = numThreads;

	for(int i = 0; i < numThreads; i++){
		thread t([&, i]() {

			while (true) {

				mtx_files.lock();

				string file;


				if(files.size() > 0){
					// there is a file to load

					file = files.back();
					files.pop_back();

					mtx_files.unlock();

					// loadFile(file, 0, 375);
					loadFile(file, 0, 4096);

					// for(int j = 0; j < 100; j++){
					// 	int64_t offset = 1 * j * 4096;
					// 	int64_t size = 4096;
					// 	loadFile(file, offset, size);
					// }

				} else if (files.size() == 0 && requestShutdown) {

					// no more files to load and no more files coming
					mtx_files.unlock();

					numActiveThreads--;

					return;
				} else if (files.size() == 0) {
					// no file to load, but wait for more

					mtx_files.unlock();
					std::this_thread::sleep_for(1ns);

				} 

				//mtx_files.unlock();
				
				//std::this_thread::sleep_for(1ms);

			}

		});

		t.detach();
	}

	double t_start = now();

	// now add files to the todo-list
	for (const auto& entry : fs::directory_iterator(dir)) {
		string file = entry.path().string();
		
		mtx_files.lock();
		files.push_back(file);
		mtx_files.unlock();
	}

	requestShutdown = true;


	// loop main thread until done
	while (true) {

		{
			std::lock_guard<mutex> lock(mtx_files);
			if (files.size() == 0 && numActiveThreads == 0) break;
		}


		std::this_thread::sleep_for(1ns);

	}
	
	
	// // vector<string> todo_files;

	// double t_start = now();

	// for(string file : files){
	// 	loadFile(file, 0, 375);
	// }

	double duration = now() - t_start;

	printfmt("done in {:.1f} ms \n", duration * 1000.0);



	return 0;
}
