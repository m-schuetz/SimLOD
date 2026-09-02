#pragma once

#include <string>
#include <unordered_map>
#include "unsuck.hpp"
#include "nvrtc.h"
#include <nvJitLink.h>
#include <cmath>
#include "cuda.h"

using std::string;

using namespace std;

#define NVJITLINK_SAFE_CALL(h,x)                                  \
do {                                                              \
   nvJitLinkResult result = x;                                    \
   if (result != NVJITLINK_SUCCESS) {                             \
      std::cerr << "\nerror: " #x " failed with error "           \
                << result << '\n';                                \
      size_t lsize;                                               \
      result = nvJitLinkGetErrorLogSize(h, &lsize);               \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {             \
         char *log = (char*)malloc(lsize);                        \
         result = nvJitLinkGetErrorLog(h, log);                   \
         if (result == NVJITLINK_SUCCESS) {                       \
            std::cerr << "error: " << log << '\n';                \
            free(log);                                            \
         }                                                        \
      }                                                           \
      exit(1);                                                    \
   }                                                              \
} while(0)

struct CudaModule{

	void cu_checked(CUresult result){
		if(result != CUDA_SUCCESS){
			cout << "cuda error code: " << result << endl;
		}
	};

	string path = "";
	string name = "";
	bool compiled = false;
	bool success = false;
	
	size_t ptxSize = 0;
	char* ptx = nullptr;

	size_t ltoirSize = 0;
	char* ltoir = nullptr;

	//size_t nvvmSize;
	//char *nvvm = nullptr;

	CudaModule(string path, string name){
		this->path = path;
		this->name = name;
	}

	void compile(){
		auto tStart = now();

		cout << "================================================================================" << endl;
		cout << "=== COMPILING: " << fs::path(path).filename().string() << endl;
		cout << "================================================================================" << endl;

		success = false;

		string dir = fs::path(path).parent_path().string();
		// string optInclude = "-I " + dir;

		string cuda_path = std::getenv("CUDA_PATH");
		// string cuda_include = "-I " + cuda_path + "/include";

		string optInclude = std::format("-I {}", dir).c_str();
		string cuda_include = std::format("-I {}/include", cuda_path);
		string cudastd_include = std::format("-I {}/include/cuda/std", cuda_path);

		// LTO IR 的虚拟架构不得高于链接目标 sm 架构，硬编码 compute_89 在 sm_86 等
		// 更老架构上会报 "LTOIR arch newer than target arch"，按 0 号卡实际计算能力生成
		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);
		int ccMajor = 0, ccMinor = 0;
		cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
		cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
		string optArch = std::format("--gpu-architecture=compute_{}{}", ccMajor, ccMinor);

		nvrtcProgram prog;
		string source = readFile(path);
		nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, NULL, NULL);
		std::vector<const char*> opts = {
			optArch.c_str(),
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			cuda_include.c_str(),
			optInclude.c_str(),
			"-I ./",
			"--relocatable-device-code=true",
			"-default-device",
			"-dlto",
			// 本项目不使用半精度；CUDA 12.5 的 NVRTC prelude 会把 __half/bf16 转换构造
			// 函数编成强符号，多模块 LTO 链接时报 "symbol multiply defined"
			"-D__CUDA_NO_HALF_OPERATORS__",
			"-D__CUDA_NO_HALF_CONVERSIONS__",
			"-D__CUDA_NO_HALF2_OPERATORS__",
			"-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
			"-D__CUDA_NO_BFLOAT16_OPERATORS__",
			"--std=c++20",
			"--disable-warnings",
		};


		for(auto opt : opts){
			cout << opt << endl;
		}
		cout << "====" << endl;

		nvrtcResult res = nvrtcCompileProgram(prog, opts.size(), opts.data());
		
		if (res != NVRTC_SUCCESS)
		{
			size_t logSize;
			nvrtcGetProgramLogSize(prog, &logSize);
			char* log = new char[logSize];
			nvrtcGetProgramLog(prog, log);
			std::cerr << log << std::endl;
			delete[] log;

			if(res != NVRTC_SUCCESS){
				return;
			}
		}

		nvrtcGetLTOIRSize(prog, &ltoirSize);
		ltoir = new char[ltoirSize];
		nvrtcGetLTOIR(prog, ltoir);

		printfmt("compiled ltoir. size: {} byte \n", ltoirSize);

		nvrtcDestroyProgram(&prog);

		compiled = true;
		success = true;

		printElapsedTime("compile " + name, tStart);

	}

};


struct CudaModularProgram{

	struct CudaModularProgramArgs{
		vector<string> modules;
		vector<string> kernels;
	};

	void cu_checked(CUresult result){
		if(result != CUDA_SUCCESS){
			cout << "cuda error code: " << result << endl;
		}
	};

	vector<CudaModule*> modules;

	CUmodule mod;
	// CUfunction kernel = nullptr;
	void* cubin;
	size_t cubinSize;

	vector<std::function<void(void)>> compileCallbacks;

	vector<string> kernelNames;
	unordered_map<string, CUfunction> kernels;


	CudaModularProgram(CudaModularProgramArgs args){
	// CudaModularProgram(vector<string> modulePaths, vector<string> kernelNames = {}){

		vector<string> modulePaths = args.modules;
		vector<string> kernelNames = args.kernels;

		this->kernelNames = kernelNames;

		for(auto modulePath : modulePaths){

			string moduleName = fs::path(modulePath).filename().string();
			auto module = new CudaModule(modulePath, moduleName);

			module->compile();

			monitorFile(modulePath, [&, module]() {
				module->compile();
				link();
			});

			modules.push_back(module);
		}

		link();
	}

	void link(){

		cout << "================================================================================" << endl;
		cout << "=== LINKING" << endl;
		cout << "================================================================================" << endl;
		
		auto tStart = now();

		for(auto module : modules){
			if(!module->success){
				return;
			}
		}

		float walltime;
		constexpr uint32_t v_optimization_level = 1;
		constexpr uint32_t logSize = 8192;
		char info_log[logSize];
		char error_log[logSize];
		
		CUlinkState linkState;

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		int major = 0;
		int minor = 0;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

		int arch = major * 10 + minor;
		string strArch = std::format("-arch=sm_{}", arch);

		const char *lopts[] = {"-dlto", strArch.c_str()};

		nvJitLinkHandle handle;
		nvJitLinkCreate(&handle, 2, lopts);

		for(auto module : modules){
			NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void *)module->ltoir, module->ltoirSize, "module label"));
		}

		NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
		NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));

		cubin = malloc(cubinSize);
		NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin));
		NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

		static int cubinID = 0;
		// writeBinaryFile(format("./program_{}.cubin", cubinID), (uint8_t*)cubin, cubinSize);
		cubinID++;

		cu_checked(cuModuleLoadData(&mod, cubin));

		for(string kernelName : kernelNames){
			CUfunction kernel;
			cu_checked(cuModuleGetFunction(&kernel, mod, kernelName.c_str()));

			kernels[kernelName] = kernel;
		}

		for(auto& callback : compileCallbacks){
			callback();
		}

		// printElapsedTime("cuda link duration: ", tStart);

	}

	void onCompile(std::function<void(void)> callback){
		compileCallbacks.push_back(callback);
	}

};