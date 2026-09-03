#pragma once

constexpr float PI = 3.1415;

constexpr bool RIGHTSIDE_BOXES = false;
constexpr bool RIGHTSIDE_NODECOLORS = false;

constexpr bool ENABLE_TRACE = false;

// 核心常量：节点超过 MAX_POINTS_PER_NODE 点即分裂；内部节点在 GRID_SIZE^3
// 网格上采样体素作为 LOD 代理；MAX_DEPTH_GRIDSIZE 为根节点的世界尺寸上限
constexpr int MAX_POINTS_PER_NODE    = 50'000;
constexpr uint32_t POINTS_PER_CHUNK  = 1000;
constexpr uint32_t GRID_SIZE         = 128;
constexpr uint32_t GRID_NUM_CELLS    = GRID_SIZE * GRID_SIZE * GRID_SIZE;
constexpr int MAX_DEPTH              = 20;
constexpr float MAX_DEPTH_GRIDSIZE   = 268'435'456.0f;

constexpr uint64_t BATCH_STREAM_SIZE = 50;

struct Point{
	float x;
	float y;
	float z;
	uint32_t color;
};

struct Voxel{
	uint8_t X;
	uint8_t Y;
	uint8_t Z;
	uint8_t filler;
	uint32_t color;
};

struct Lines{
	unsigned int count = 0;
	unsigned int padding0;
	unsigned int padding1;
	unsigned int padding2;
	Point* vertices;
};

// ============================================================================
// GPU 布局契约锁
//
// 本文件由 NVRTC 在运行时编译，宿主 exe 由 MSVC 编译，两侧仅靠字段布局达成
// 默契：CPU 按此布局写 pinned memory → cuMemcpyHtoD 原样上传 → kernel 按
// 相同布局解释。布局一旦漂移（加字段、改顺序、padding 变化），不会有任何
// 编译错误，只会产生数据错位/非法访问类疑难杂症。
// 这里的 static_assert 在 NVRTC 编译每个 kernel 时生效；HostDeviceInterface.h
// 中的对应断言由 MSVC 与 NVRTC 双侧检查。有意修改布局时必须同步更新断言。
// 注意：NVRTC 不接受字符串字面量中的非 ASCII 字符，断言消息只能用英文。
// ============================================================================
static_assert(sizeof(Point) == 16,                    "Point: float xyz + packed RGBA; host loader and upload ring buffer rely on 16 bytes");
static_assert(sizeof(Voxel) == 8,                     "Voxel: uint8 xyz + filler + color");
static_assert(POINTS_PER_CHUNK == 1000,               "upload ring buffer sizing and kernel batching rely on this");
static_assert(GRID_SIZE == 128,                       "voxel sampling grid resolution, core paper constant");
static_assert(GRID_NUM_CELLS == 128 * 128 * 128,      "GRID_SIZE squared; do not change independently");

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

struct Chunk{
	Point points[POINTS_PER_CHUNK];
	int size;
	int padding_0;
	Chunk* next;
};

struct OccupancyGrid{
	// gridsize^3 occupancy grid; 1 bit per voxel
	uint32_t values[GRID_NUM_CELLS / 32u];
};

struct Node{
	Node* children[8];
	uint32_t counter = 0;
	// uint32_t counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};

	uint32_t numPoints = 0;
	uint32_t level = 0;
	uint32_t X = 0;
	uint32_t Y = 0;
	uint32_t Z = 0;
	uint32_t countIteration = 0;
	uint32_t countFlag = 0;
	uint8_t name[20] = {'r', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	bool visible = false;
	bool isFiltered = false;
	bool isLeaf = true;
	bool isLarge = false;

	OccupancyGrid* grid = nullptr;
	
	Chunk* points = nullptr;
	Chunk* voxelChunks = nullptr;

	uint32_t numVoxels = 0;
	uint32_t numVoxelsStored = 0;

	// bool spilled(){
	// 	return counter > MAX_POINTS_PER_NODE;
	// }

	bool isLeafFn(){

		if(children[0] != nullptr) return false;
		if(children[1] != nullptr) return false;
		if(children[2] != nullptr) return false;
		if(children[3] != nullptr) return false;
		if(children[4] != nullptr) return false;
		if(children[5] != nullptr) return false;
		if(children[6] != nullptr) return false;
		if(children[7] != nullptr) return false;

		return true;
	}

	uint64_t getID(){
		uint64_t id = 0;

		id = id | ((name[ 0] == 'r' ? 1 : 0));
		id = id | ((name[ 1] - '0') <<  3);
		id = id | ((name[ 2] - '0') <<  6);
		id = id | ((name[ 3] - '0') <<  9);
		id = id | ((name[ 4] - '0') << 12);
		id = id | ((name[ 5] - '0') << 15);
		id = id | ((name[ 6] - '0') << 18);
		id = id | ((name[ 7] - '0') << 21);
		id = id | ((name[ 8] - '0') << 24);
		id = id | ((name[ 9] - '0') << 27);
		id = id | (uint64_t((name[10] - '0')) << 30);
		id = id | (uint64_t((name[11] - '0')) << 33);
		id = id | (uint64_t((name[12] - '0')) << 36);
		id = id | (uint64_t((name[13] - '0')) << 39);
		id = id | (uint64_t((name[14] - '0')) << 42);
		id = id | (uint64_t((name[15] - '0')) << 45);
		id = id | (uint64_t((name[16] - '0')) << 48);
		id = id | (uint64_t((name[17] - '0')) << 51);
		id = id | (uint64_t((name[18] - '0')) << 53);

		return id;
	}

};

// Chunk / OccupancyGrid 的断言必须位于其定义之后
static_assert(sizeof(Chunk) == POINTS_PER_CHUNK * sizeof(Point) + 16, "Chunk: points array + size + padding_0 + next pointer");
static_assert(sizeof(OccupancyGrid) == GRID_NUM_CELLS / 8, "OccupancyGrid: 1 bit per voxel");