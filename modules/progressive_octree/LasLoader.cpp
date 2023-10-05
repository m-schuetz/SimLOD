#define NOMINMAX

#include "LasLoader.h"

#include <filesystem>
#include <mutex>
#include <deque>
#include <atomic>
#include <string>
#include <algorithm>
#include <locale>
#include <codecvt>


using std::string;

static thread_local bool initialized = false;
static thread_local void* thread_buffer = nullptr;
static thread_local uint64_t thread_buffer_size = 0;


#ifdef _WIN32

#include "windows.h"

static thread_local bool isLoadPending = false;
static thread_local bool shouldRetry = false;
static thread_local bool isLoadDone = false;
static thread_local string dbg_file;

VOID CALLBACK FileIOCompletionRoutine_las(
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

uint64_t loadLasNative(string file, LasHeader header, uint64_t firstPoint, uint64_t numPoints){
	uint64_t sectorSize = getPhysicalSectorSize(file);
	uint64_t requiredThreadBufferSize = 10 * sectorSize + std::max(header.bytesPerPoint, 36llu) * numPoints;

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

	string filename = fs::path(file).filename().string();

	uint64_t firstByte = header.offsetToPointData + header.bytesPerPoint * firstPoint;
	uint64_t numBytes = header.bytesPerPoint * numPoints;
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
			FileIOCompletionRoutine_las
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

	return padding;
}

#elif defined(__linux__)

uint64_t loadLasNative(string file, LasHeader header, uint64_t firstPoint, uint64_t numPoints){
  uint64_t requiredThreadBufferSize = header.bytesPerPoint * numPoints;
  if(!initialized){
    thread_buffer = malloc(requiredThreadBufferSize);
    thread_buffer_size = requiredThreadBufferSize;
    initialized = true;
  }else if(thread_buffer_size < requiredThreadBufferSize){
    free(thread_buffer);
    thread_buffer = malloc(requiredThreadBufferSize);
    thread_buffer_size = requiredThreadBufferSize;
  }

	readBinaryFile(
		file, 
		header.offsetToPointData + header.bytesPerPoint * firstPoint,
		header.bytesPerPoint * numPoints,
		thread_buffer);

	return 0;
}

#endif


void loadLasNative(string file, LasHeader header, uint64_t firstPoint, uint64_t numPoints, void* target, double translation[3]){

	// - figure out what buffer range to load based on firstPoint and numPoints
	// - load that buffer with winapi
	// - parse points and store them in <target>

	uint64_t padding = loadLasNative(file, header, firstPoint, numPoints);

	// now parse loaded points and write results to <target>
	uint64_t offset_rgb = 0;
	if(header.format == 2){
		offset_rgb = 20;
	}else if(header.format == 3){
		offset_rgb = 28;
	}if(header.format == 5){
		offset_rgb = 28;
	}if(header.format == 7){
		offset_rgb = 30;
	}

	struct Point{
		float x, y, z;
		uint8_t r, g, b, a;
	};

	double tStart = now();

	uint8_t* source = ((uint8_t*)thread_buffer) + padding;
	Point* pTarget = (Point*)target;
	Point point;
	int32_t XYZ[3];
	double scale_x = header.scale[0];
	double scale_y = header.scale[1];
	double scale_z = header.scale[2];
	double offset_x = header.offset[0] + translation[0];
	double offset_y = header.offset[1] + translation[1];
	double offset_z = header.offset[2] + translation[2];


	for(uint64_t i = 0; i < numPoints; i++){
		// int32_t* XYZ = (int32_t*)(source + header.bytesPerPoint * i);

		memcpy(XYZ, source + header.bytesPerPoint * i, 12);

		point.x = double(XYZ[0]) * scale_x + offset_x;
		point.y = double(XYZ[1]) * scale_y + offset_y;
		point.z = double(XYZ[2]) * scale_z + offset_z;

		if(offset_rgb > 0){
			uint16_t* rgb = (uint16_t*)(source + header.bytesPerPoint * i + offset_rgb);
			point.r = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
			point.g = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
			point.b = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];
		}
		
		pTarget[i] = point;
	}

}