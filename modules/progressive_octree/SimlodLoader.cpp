#define NOMINMAX

#include "SimlodLoader.h"

#include <filesystem>
#include <mutex>
#include <deque>
#include <atomic>
#include <string>
#include <algorithm>
#include <locale>
#include <codecvt>

#include "unsuck.hpp"

// - Standard file IO routines like fread or ifstream are fairly slow when it comes to PCIe 5 SSDs on windows.
// - Windows caches files when they are read, making the first read slower and surprisingly CPU-intensive, 
//   but subsequent reads much faster because windows then serves the data from RAM.
// - However, we are interested in optimizing the first access to a point cloud file.
// - To do this, we use ReadFileEx with the unbuffered flag, which does not cache files.
// - Unbuffered reads add some additional complexity because they must be aligned to the disk's physical sector size.
#ifdef _WIN32

static thread_local bool isLoadPending = false;
static thread_local bool shouldRetry = false;
static thread_local bool isLoadDone = false;
static thread_local string dbg_file;

#include "windows.h"

// - Callback that is invoked when ReadFileEx finished
// - The callback is invoked within the calling thread's context!
// - A thread needs to SleepEx so that the callback can be invoked.
VOID CALLBACK FileIOCompletionRoutine_simlod(
	__in DWORD errorCode,
	__in DWORD numBytesTransfered,
	__in LPOVERLAPPED lpOverlapped)
{

	if(errorCode != 0){
		printfmt("read error: {} \n", errorCode);
	}

	if(numBytesTransfered == 0){
		printfmt("0 bytes read. offset: {:14L}, numBytes: {:9} \n", lpOverlapped->Offset, numBytesTransfered);

		isLoadPending = false;
		shouldRetry = true;
		isLoadDone = false;
	} else {
		string filename = fs::path(dbg_file).filename().string();

		isLoadPending = false;
		shouldRetry = false;
		isLoadDone = true;
	}
}

void loadFileNative(
	string file, 
	uint64_t firstByte, 
	uint64_t numBytes, 
	void* target,
	uint64_t* out_padding
){

	isLoadPending = false;
	shouldRetry = false;
	isLoadDone = false;

	string filename = fs::path(file).filename().string();

	// unbuffered reads require alignment to disk's sector sizes
	uint64_t sectorSize = getPhysicalSectorSize(file);
	uint64_t endByte = firstByte + numBytes;
	uint64_t firstByte_aligned = firstByte - firstByte % sectorSize;
	uint64_t endByte_aligned = endByte;
	if((endByte % sectorSize) != 0){
		endByte_aligned = endByte - (endByte % sectorSize) + sectorSize;
	}
	uint64_t numBytes_aligned = endByte_aligned - firstByte_aligned;

	dbg_file = file;

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

	double maxtime = 0.5;
	double tStart = now();
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
			(char*)target, 
			numBytes_aligned, 
			&ol, 
			FileIOCompletionRoutine_simlod
		);

		if (returnValue == 0) {
			printfmt("ERROR: ReadFileEx failed \n");

			isLoadPending = false;
			isLoadDone = false;
		}
		
		SleepEx(1, TRUE);

	}while(shouldRetry || isLoadPending);
	
	CloseHandle(hFile);

	uint64_t padding = firstByte - firstByte_aligned;

	*out_padding = padding;

	isLoadPending = false;
	shouldRetry = false;
	isLoadDone = false;
}


#elif defined(__linux__)


void loadFileNative(
	string file, 
	uint64_t firstByte, 
	uint64_t numBytes, 
	void* target,
	uint64_t* out_padding
){
	readBinaryFile(file, firstByte, numBytes, target);

	*out_padding = 0;
}

#endif