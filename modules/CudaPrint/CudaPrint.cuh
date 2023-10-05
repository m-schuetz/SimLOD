
#pragma once

constexpr uint32_t TYPE_UINT32_T = 0;
constexpr uint32_t TYPE_UINT64_T = 1;
constexpr uint32_t TYPE_INT32_T  = 2;
constexpr uint32_t TYPE_INT64_T  = 3;
constexpr uint32_t TYPE_FLOAT    = 4;
constexpr uint32_t TYPE_DOUBLE   = 5;
constexpr uint32_t TYPE_CSTRING  = 20;

struct CudaPrintEntry{
	uint32_t keylen;

	uint8_t numArgs;
	uint8_t method;
	uint8_t padding_0;
	uint8_t padding_1;

	uint8_t data[1024 - 16];
};

template<class T>
uint32_t sizeOf(T value){
	return sizeof(value);
}

template<>
uint32_t sizeOf<const char*>(const char* value){
	return strlen(value);
}

struct CudaPrint{
	uint64_t entryCounter = 0;
	uint64_t padding;
	CudaPrintEntry entries[1000];

	static uint32_t typeof(uint32_t value){ return TYPE_UINT32_T; }
	static uint32_t typeof(uint64_t value){ return TYPE_UINT64_T; }
	static uint32_t typeof(int32_t  value){ return TYPE_INT32_T; }
	static uint32_t typeof(int64_t  value){ return TYPE_INT64_T; }
	static uint32_t typeof(float    value){ return TYPE_FLOAT; }
	static uint32_t typeof(double   value){ return TYPE_DOUBLE; }
	static uint32_t typeof(const char* value){ return TYPE_CSTRING; }

	

	template <typename... Args>
	inline void print(const char* key, const Args&... args) {

		return;

		uint32_t entryIndex = atomicAdd(&entryCounter, 1) % 1000;
		CudaPrintEntry *entry = &entries[entryIndex];

		constexpr uint32_t numargs{ sizeof...(Args) };

		int argsSize = 0;

		for(const auto p : {args...}) {
			// testType(p);
			argsSize += sizeof(p);
			entry->numArgs++;
		}

		entry->keylen = strlen(key);
		entry->method = 0;

		memcpy(entry->data, key, entry->keylen);

		uint32_t offset = entry->keylen;
		for(const auto p : {args...}) {
			// testType(p);
			// uint32_t argSize = sizeof(p);
			uint32_t argtype = CudaPrint::typeof(p);

			// printf("arg: %s \n", p);

			// if(argtype == TYPE_CSTRING){
				uint32_t argSize = sizeOf(p);
				uint32_t typeAndSize = argtype | (argSize << 8);
				memcpy(entry->data + offset, &typeAndSize, 4);
				offset += 4;
				memcpy(entry->data + offset, &p, argSize);
				offset += argSize;
			// }else{
			// 	uint32_t argSize = CudaPrint::sizeOf(p);
			// 	memcpy(entry->data + offset, &argtype, 4);
			// 	offset += 4;
			// 	memcpy(entry->data + offset, &p, sizeof(p));
			// 	offset += argSize;
			// }
			
			

		}
	}

	inline void set(const char* key, const char* value) {

		// uint32_t entryIndex = atomicAdd(&entryCounter, 1) % 1000;
		// CudaPrintEntry *entry = &entries[entryIndex];

		// entry->keylen   = strlen(key);
		// entry->numArgs  = 1;
		// entry->method   = 1;

		// uint32_t argtype = CudaPrint::typeof(value);

		// memcpy(entry->data, key, entry->keylen);
		// memcpy(entry->data + entry->keylen, &argtype, 4);
		// memcpy(entry->data + entry->keylen + 4, &value, sizeof(value));
	}

	template <typename T>
	inline void set(const char* key, const T value) {

		uint32_t entryIndex = atomicAdd(&entryCounter, 1) % 1000;
		CudaPrintEntry *entry = &entries[entryIndex];

		entry->keylen   = strlen(key);
		entry->numArgs  = 1;
		entry->method   = 1;

		uint32_t argtype = CudaPrint::typeof(value);

		memcpy(entry->data, key, entry->keylen);
		memcpy(entry->data + entry->keylen, &argtype, 4);
		memcpy(entry->data + entry->keylen + 4, &value, sizeof(value));
	}
};