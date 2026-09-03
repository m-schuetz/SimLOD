cmake_minimum_required(VERSION 3.22)
project(SimLOD)

set(CMAKE_CXX_STANDARD 20)

# 中文 Windows (CP936) 下部分依赖头文件中的 UTF-8 字符会导致 C4819/C2001
if(MSVC)
	add_compile_options(/utf-8)
endif()

option(SimLOD_CreateCUDASymlinks "Create symbolic links instead of copying CUDA files." OFF)

# Target SimLOD
# 依赖全部来自全局 vcpkg（classic 模式），构建时需传入 vcpkg 工具链：
#   cmake --preset vcpkg-debug   （见 CMakePresets.json）
#   或 cmake -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake

add_executable(${PROJECT_NAME}
	src/GLRenderer.cpp
    include/unsuck_platform_specific.cpp
    modules/progressive_octree/LasLoader.cpp
    modules/progressive_octree/SimlodLoader.cpp
	modules/progressive_octree/main_progressive_octree.cpp)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                                               VS_DEBUGGER_COMMAND           "$<TARGET_FILE:${PROJECT_NAME}>"
                                               VS_DEBUGGER_ENVIRONMENT       "PATH=%PATH%;${CMAKE_PREFIX_PATH}")
# Build options based on CMAKE_BUILD_TYPE
function(configure_build_type)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		target_compile_options(${PROJECT_NAME} PRIVATE -O0 -g)
	elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
		target_compile_options(${PROJECT_NAME} PRIVATE -O3)
	else()
		message(WARNING "No CMAKE_BUILD_TYPE specified, defaulting to Release settings.")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type: Release" FORCE)
		target_compile_options(${PROJECT_NAME} PRIVATE -O3)
	endif ()
endfunction()
configure_build_type()

target_include_directories(${PROJECT_NAME} PRIVATE
    include
	modules/CudaPrint
	modules/progressive_octree)

# Dependencies


find_package(CUDAToolkit 12.5 REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE
	${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
	CUDA::cuda_driver
	CUDA::nvrtc)

if(TARGET CUDA::nvJitLink)
	target_link_libraries(${PROJECT_NAME} CUDA::nvJitLink)
endif()

## OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY})

# glm 1.0.x: gtx/transform.hpp 需要显式启用实验扩展（0.9.9 无此要求）
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)

## vcpkg 依赖（classic 模式全局安装）
find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_path(LASZIP_INCLUDE_DIR laszip_api.h PATH_SUFFIXES laszip)
find_library(LASZIP_LIBRARY NAMES laszip3 laszip)

target_link_libraries(${PROJECT_NAME}
	fmt::fmt
	glfw
	GLEW::GLEW
	glm::glm
	imgui::imgui
	implot::implot
	${LASZIP_LIBRARY})
target_include_directories(${PROJECT_NAME} PRIVATE ${LASZIP_INCLUDE_DIR})

## TODO: disable copying of cuda source files
## - developer builds should use the project root dir as the working directory and directly load from ./modules
## - public release builds should be shipped with the precompiled cuda programs (cached in temp/cuda_programs)
# Post-Build
if (SimLOD_CreateCUDASymlinks)
	execute_process(
		COMMAND ${CMAKE_COMMAND} -E create_symlink
			"${PROJECT_SOURCE_DIR}/CMakelists.txt"
			"${CMAKE_BINARY_DIR}/.symlinktest.txt"
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		RESULT_VARIABLE symlinksSupported)
	if (symlinksSupported)
		message(STATUS "Cannot create symbolic links - missing user privileges. Falling back to copying.")
	else ()
		execute_process(
				COMMAND ${CMAKE_COMMAND} -E remove
				"${CMAKE_BINARY_DIR}/.symlinktest.txt"
				WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	endif (symlinksSupported)
endif (SimLOD_CreateCUDASymlinks)

if (symlinksSupported OR NOT SimLOD_CreateCUDASymlinks)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_directory
			"${PROJECT_SOURCE_DIR}/modules"
			"$<TARGET_FILE_DIR:${PROJECT_NAME}>/modules"
		COMMENT "POST BUILD: copying modules folder")
else ()
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E create_symlink
			"${PROJECT_SOURCE_DIR}/modules"
			"$<TARGET_FILE_DIR:${PROJECT_NAME}>/modules"
		COMMENT "POST BUILD: creating symlink for modules folder")
endif (symlinksSupported OR NOT SimLOD_CreateCUDASymlinks)
