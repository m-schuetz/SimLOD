
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FALSE 0
#define TRUE 1

typedef unsigned int uint32_t;
typedef int int32_t;
// typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

// #define Infinity 0x7f800000u
#define Infinity 1.0f / 0.0f

constexpr uint32_t MAX_STRING_LENGTH = 1'000;

inline uint32_t strlen(const char* str){

	uint32_t length = 0;

	for(int i = 0; i < MAX_STRING_LENGTH; i++){
		if(str[i] != 0){
			length++;
		}else{
			break;
		}
	}


	return length;
}

// calls function <f> <size> times
// calls are distributed over all available threads
template<typename Function>
void processRange(int first, int size, Function&& f){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = first + block_offset + thread_offset + i;

		if(index >= first + size){
			break;
		}

		f(index);
	}
}

// calls function <f> <size> times
// calls are distributed over all available threads
template<typename Function>
void processRange(int size, Function&& f){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = block_offset + thread_offset + i;

		if(index >= size){
			break;
		}

		f(index);
	}
}

void printNumber(int64_t number, int leftPad = 0);

constexpr bool BUFFER_CHECK_BOUNDS = true;
struct Buffer{

	uint64_t size = 0;
	uint8_t* data = nullptr;

	template<typename T>
	T get(uint64_t offset){
		T value;

		if(BUFFER_CHECK_BOUNDS){
			if(offset + sizeof(T) > size){
				printf("get out of bounds. size: %llu, offset: %llu, valuesize: %llu \n", 
					size, offset, sizeof(T));

				return;
			}
		}

		memcpy(&value, data + offset, sizeof(T));

		return value;
	}

	template<typename T>
	void set(uint64_t offset, const T& value){

		if(BUFFER_CHECK_BOUNDS){
			if(offset + sizeof(T) > size){
				printf("set out of bounds. size: %llu, offset: %llu, valuesize: %llu \n", 
					size, offset, sizeof(T));

				return;
			}
		}

		memcpy(data + offset, &value, sizeof(T));
	}

};

constexpr bool ARRAY_CHECK_BOUNDS = true;

template<typename T>
struct Array{
	uint64_t capacity = 0;
	uint64_t size = 0;
	T* data = nullptr;

	T get(uint64_t index){

		if(ARRAY_CHECK_BOUNDS){
			if(index >= size || index >= capacity){
				printf("get out of bounds. capacity: %llu. size: %llu, tried to access %llu \n", 
					capacity, size, index);

				return;
			}
		}

		return data[index];
	}

	void set(uint64_t index, T value){

		if(ARRAY_CHECK_BOUNDS){
			if(index >= size || index >= capacity){
				printf("set out of bounds. capacity: %llu, size: %llu, tried to access %llu \n", 
					capacity, size, index);

				return;
			}
		}

		data[index] = value;
	}

	void push(T value){
		uint64_t index = atomicAdd(&size, 1);

		if(ARRAY_CHECK_BOUNDS){
			if(index >= capacity){
				printf("push out of bounds. capacity: %llu, size: %llu, tried to access %llu \n", 
					capacity, size, index);

				return;
			}
		}

		data[index] = value;
	}
};

struct AllocatorGlobal{

	uint8_t* buffer = nullptr;
	uint64_t offset = 0;

	uint8_t* alloc(uint64_t size){

		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		// round up to nearest 16
		uint64_t size_16 = 16ll * ((size + 16ll) / 16ll);

		uint64_t oldOffset = atomicAdd(&offset, size_16);

		uint8_t* ptr = buffer + oldOffset;

		return ptr;
	}

	Buffer allocBuffer(uint64_t size){

		uint64_t size_16 = 16ll * ((size + 16ll) / 16ll);

		uint64_t oldOffset = atomicAdd(&offset, size_16);

		uint8_t* ptr = buffer + oldOffset;

		Buffer buffer;
		buffer.size = size;
		buffer.data = ptr;

		return buffer;
	}

	template<typename T>
	Array<T> allocArray(uint64_t capacity){

		uint64_t byteSize = capacity * sizeof(T);

		Array<T> array;
		array.capacity = capacity;
		array.size = 0;
		array.data = (T*)alloc(byteSize);

		return array;
	}

};

struct Allocator{

	uint8_t* buffer = nullptr;
	int64_t offset = 0;

	template<class T>
	Allocator(T buffer){
		this->buffer = reinterpret_cast<uint8_t*>(buffer);
		this->offset = 0;
	}

	Allocator(unsigned int* buffer, int64_t offset){
		this->buffer = reinterpret_cast<uint8_t*>(buffer);
		this->offset = offset;
	}

	template<class T>
	T alloc(int64_t size){

		auto ptr = reinterpret_cast<T>(buffer + offset);

		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		
		this->offset = newOffset;

		return ptr;
	}

	template<class T>
	T* alloc2(int numElements){

		auto ptr = reinterpret_cast<T*>(buffer + offset);

		int64_t size = numElements * sizeof(T);
		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		
		this->offset = newOffset;

		return ptr;
	}

	template<class T>
	T alloc(int64_t size, const char* label){

		// if(isFirstThread()){
		// 	printf("offset: ");
		// 	printNumber(offset, 13);
		// 	printf(", allocating: ");
		// 	printNumber(size, 13);
		// 	printf(", label: %s \n", label);
		// }

		auto ptr = reinterpret_cast<T>(buffer + offset);

		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		
		this->offset = newOffset;

		return ptr;
	}

	// template<typename T>
	// Array<T>* allocArray(uint64_t capacity){

	// 	Array<Point>* array = alloc<Array<Point>*>(sizeof(Array<Point>));

	// 	uint64_t byteSize = capacity * sizeof(T);

	// 	array->capacity = capacity;
	// 	array->size = 0;
	// 	array->data = (T*)alloc(byteSize);

	// 	return array;
	// }

};

inline uint64_t nanotime(){

	uint64_t nanotime;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));

	return nanotime;
}

inline float millitime(){

	uint64_t nanotime;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));

	float millies = float(nanotime / 1000llu) / 1000.0f;

	return millies;
}

inline bool strequal(uint8_t* a, const char* _b){

	uint8_t* b = (uint8_t*)_b;

	for(int i = 0; i < 100; i++){
		if(a[i] != b[i]) return false;

		if(a[i] == 0 || b[i] == 0) break;
	}

	return true;
}

// inline bool operator==(uint8_t* a, const char* _b) { 
// 	return strequal(a, _b);
// }


#define PRINT(...) {                     \
	if(cg::this_grid().thread_rank() == 0){ \
		printf(__VA_ARGS__);                \
	}                                       \
}