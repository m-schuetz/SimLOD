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
		string cudastd_detail_include = std::format("-I {}/include/cuda/std/detail/libcxx/include", cuda_path);
		string wtf = "-I C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt";
		string wtf2 = "-I C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.36.32532/include";
		
		//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\cuda\std\detail\libcxx\include\iterator"

		

		string i_cub            = format("-I {}", "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/cub");
		string i_libcuda        = format("-I {}", "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/libcudacxx/include");
		string i_cudastd_detail = format("-I {}", "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/libcudacxx/include/cuda/std/detail/libcxx/include/");
		string i_libcudastd     = format("-I {}", "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/libcudacxx/include/cuda/std");
		string i_thrust         = format("-I {}", "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/thrust");

		nvrtcProgram prog;
		string source = readFile(path);
		nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, NULL, NULL);
		std::vector<const char*> opts = { 
			"--gpu-architecture=compute_89",
			// "--gpu-architecture=compute_86",
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			// cudastd_detail_include.c_str(),
			// cudastd_include.c_str(),
			i_cub.c_str(),
			i_libcuda.c_str(),
			// i_cudastd_detail.c_str(),
			i_libcudastd.c_str(),
			// i_thrust.c_str(),
			cuda_include.c_str(),
			optInclude.c_str(),
			"-I ./",
			// wtf.c_str(),
			// wtf2.c_str(),
			"--relocatable-device-code=true",
			"-default-device",
			"-dlto", 
			// "--dopt=on",
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

		//if(nvvmSize > 0){
		//	delete[] nvvm;
		//	nvvmSize = 0;
		//}

		nvrtcGetLTOIRSize(prog, &ltoirSize);
		ltoir = new char[ltoirSize];
		nvrtcGetLTOIR(prog, ltoir);

		cout << format("compiled ltoir. size: {} byte \n", ltoirSize);

		//nvrtcGetNVVMSize(prog, &nvvmSize);
		//nvvm = new char[nvvmSize];
		//nvrtcGetNVVM(prog, nvvm);
		//// Destroy the program.
		nvrtcDestroyProgram(&prog);

		compiled = true;
		success = true;

		printElapsedTime("compile " + name, tStart);

	}

};

struct OptionalLaunchSettings{
	uint32_t gridsize[3] = {1, 1, 1};
	uint32_t blocksize[3] = {1, 1, 1};
	vector<void*> args;
	bool measureDuration = false;
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
	unordered_map<string, CUevent> events_launch_start;
	unordered_map<string, CUevent> events_launch_end;
	unordered_map<string, float> last_launch_duration;


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

		// vector<CUjit_option> options = {
		// 	CU_JIT_LTO,
		// 	CU_JIT_WALL_TIME,
		// 	CU_JIT_OPTIMIZATION_LEVEL,
		// 	CU_JIT_INFO_LOG_BUFFER,
		// 	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		// 	CU_JIT_ERROR_LOG_BUFFER,
		// 	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
		// 	CU_JIT_LOG_VERBOSE,
		// 	// CU_JIT_FAST_COMPILE // CUDA internal only (?)
		// };

		// vector<void*> optionVals = {
		// 	(void*) 1,
		// 	(void*) &walltime,
		// 	(void*) 4,
		// 	(void*) info_log,
		// 	(void*) logSize,
		// 	(void*) error_log,
		// 	(void*) logSize,
		// 	(void*) 1,
		// 	// (void*) 1
		// };
		
		CUlinkState linkState;

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		int major = 0;
		int minor = 0;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

		int arch = major * 10 + minor;
		// char smbuf[16];
		// memset(smbuf, 0, 16);
		// sprintf(smbuf, "-arch=sm_%d\n", arch);

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



		// int numOptions = options.size();
		// cu_checked(cuLinkCreate(numOptions, options.data(), optionVals.data(), &linkState));

		// for(auto module : modules){

		// 	CU_JIT_INPUT_
		// 	cu_checked(cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
		// 		module->ltoir, module->ltoirSize, module->name.c_str(),
		// 		0, 0, 0));

		// 	//cu_checked(cuLinkAddData(linkState, CU_JIT_INPUT_NVVM,
		// 	//	module->nvvm, module->nvvmSize, module->name.c_str(),
		// 	//	0, 0, 0));
		// }

		// size_t cubinSize;
		// void *cubin;

		// cu_checked(cuLinkComplete(linkState, &cubin, &cubinSize));

		static int cubinID = 0;
		// writeBinaryFile(format("./program_{}.cubin", cubinID), (uint8_t*)cubin, cubinSize);
		cubinID++;

		// {
		// 	printf("link duration: %f ms \n", walltime);
			// if (strlen(error_log) <= 0) 
			// 	printf("link SUCCESS (i.e., no link error messages)\n");
			// else 
			// 	printf("link error message: %s \n", error_log);

			// if (strlen(info_log) <= 0)
			// 	printf("NO link info messages\n");
			// else
			// 	printf("link info message: %s \n", info_log);
		// }

		cu_checked(cuModuleLoadData(&mod, cubin));
		//cu_checked(cuModuleGetFunction(&kernel, mod, "kernel"));

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

	void launch(string kernelName, OptionalLaunchSettings launchArgs){


		void** args = &launchArgs.args[0];

		auto res_launch = cuLaunchKernel(kernels[kernelName],
			launchArgs.gridsize[0], launchArgs.gridsize[1], launchArgs.gridsize[2],
			launchArgs.blocksize[0], launchArgs.blocksize[1], launchArgs.blocksize[2],
			0, 0, args, nullptr);

		if (res_launch != CUDA_SUCCESS) {
			const char* str;
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
			cout << __FILE__ << " - " << __LINE__ << endl;
		}

	}

	void launchCooperative(string kernelName, void* args[], OptionalLaunchSettings launchArgs = {}){

		CUevent event_start = events_launch_start[kernelName];
		CUevent event_end   = events_launch_end[kernelName];

		cuEventRecord(event_start, 0);

		CUdevice device;
		int numSMs;
		cuCtxGetDevice(&device);
		cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

		int blockSize = 128;
		int numBlocks;
		CUresult resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernels[kernelName], blockSize, 0);
		numBlocks *= numSMs;
		
		//numGroups = 100;
		// make sure at least 10 workgroups are spawned)
		numBlocks = std::clamp(numBlocks, 10, 100'000);

		auto kernel = this->kernels[kernelName];
		auto res_launch = cuLaunchCooperativeKernel(kernel,
			numBlocks, 1, 1,
			blockSize, 1, 1,
			0, 0, args);

		if(res_launch != CUDA_SUCCESS){
			const char* str; 
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
			cout << __FILE__ << " - " << __LINE__ << endl;
		}

		cuEventRecord(event_end, 0);

		if(launchArgs.measureDuration){
			cuCtxSynchronize();

			float duration;
			cuEventElapsedTime(&duration, event_start, event_end);

			last_launch_duration[kernelName] = duration;
		}
	}

};