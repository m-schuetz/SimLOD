#pragma once

#include <string>
#include <unordered_map>

#include "unsuck.hpp"
#include "nvrtc.h"
#include <cmath>
#include "cuda.h"

using std::string;

using namespace std;

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

	size_t nvvmSize;
	char *nvvm = nullptr;

	CudaModule(string path, string name){
		this->path = path;
		this->name = name;
	}

	void compile(){
		auto tStart = now();

		// cout << "================================================================================" << endl;
		// cout << "=== COMPILING: " << fs::path(path).filename().string() << endl;
		// cout << "================================================================================" << endl;
		printfmt("compiling {} ", fs::path(path).filename().string());

		success = false;

		string dir = fs::path(path).parent_path().string();
		string optInclude = "-I " + dir;

		string cuda_path = std::getenv("CUDA_PATH");
		string cuda_include = "-I " + cuda_path + "/include";

		nvrtcProgram prog;
		string source = readFile(path);
		nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, NULL, NULL);
		std::vector<const char*> opts = { 
			// "--gpu-architecture=compute_75",
			"--gpu-architecture=compute_86",
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			optInclude.c_str(),
			cuda_include.c_str(),
			"--relocatable-device-code=true",
			"-default-device",
			"-dlto", 
			// "--dopt=on",
			"--std=c++17"
		};

		nvrtcResult res = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
		
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

		if(nvvmSize > 0){
			delete[] nvvm;
			nvvmSize = 0;
		}

		nvrtcGetNVVMSize(prog, &nvvmSize);
		nvvm = new char[nvvmSize];
		nvrtcGetNVVM(prog, nvvm);
		// Destroy the program.
		nvrtcDestroyProgram(&prog);

		compiled = true;
		success = true;

		// printElapsedTime("compiled " + name, tStart);
		auto duration = now() - tStart;
		printfmt("- compiled in {:.3}s \n", duration);
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

	vector<std::function<void(void)>> compileCallbacks;

	vector<string> kernelNames;
	unordered_map<string, CUfunction> kernels;


	CudaModularProgram(CudaModularProgramArgs args){
	// CudaModularProgram(vector<string> modulePaths, vector<string> kernelNames = {}){

		vector<string> modulePaths = args.modules;
		vector<string> kernelNames = args.kernels;

		this->kernelNames = kernelNames;

		printfmt("================================================================================\n");
		printfmt("building CUDA program \n");

		for(auto modulePath : modulePaths){

			string moduleName = fs::path(modulePath).filename().string();
			auto module = new CudaModule(modulePath, moduleName);

			module->compile();

			monitorFile(modulePath, [&, module]() {
				printfmt("================================================================================\n");
				printfmt("building CUDA program \n");

				module->compile();
				link();
			});

			modules.push_back(module);
		}

		link();
	}

	void link(){

		// cout << "================================================================================" << endl;
		// cout << "=== LINKING" << endl;
		// cout << "================================================================================" << endl;
		
		auto tStart = now();

		for(auto module : modules){
			if(!module->success){
				return;
			}
		}

		float walltime;
		constexpr uint32_t v_optimization_level = 1;
		constexpr size_t logSize = 8192;
		char info_log[logSize];
		char error_log[logSize];

		vector<CUjit_option> options = {
			CU_JIT_LTO,
			CU_JIT_WALL_TIME,
			CU_JIT_OPTIMIZATION_LEVEL,
			CU_JIT_INFO_LOG_BUFFER,
			CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_ERROR_LOG_BUFFER,
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_LOG_VERBOSE,
			// CU_JIT_FAST_COMPILE // CUDA internal only (?)
		};

		vector<void*> optionVals = {
			(void*) 1,
			(void*) &walltime,
			(void*) 4,
			(void*) info_log,
			(void*) logSize,
			(void*) error_log,
			(void*) logSize,
			(void*) 1,
			// (void*) 1
		};
		
		CUlinkState linkState;

		int numOptions = static_cast<int>(options.size());
		cu_checked(cuLinkCreate(numOptions, options.data(), optionVals.data(), &linkState));

		for(auto module : modules){
			cu_checked(cuLinkAddData(linkState, CU_JIT_INPUT_NVVM,
				module->nvvm, module->nvvmSize, module->name.c_str(),
				0, 0, 0));
		}

		size_t cubinSize;
		void *cubin;

		cu_checked(cuLinkComplete(linkState, &cubin, &cubinSize));

		// {
		// 	printf("link duration: %f ms \n", walltime);
			printf("link error message: %s \n", error_log);
			printf("link info message: %s \n", info_log);
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

		// printElapsedTime("link duration: ", tStart);
		auto duration = now() - tStart;
		printfmt("link duration: {:.3} \n", duration);
		printfmt("================================================================================\n");
	}

	void onCompile(std::function<void(void)> callback){
		compileCallbacks.push_back(callback);
	}

};