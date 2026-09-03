cmake_minimum_required(VERSION 3.22)
project(SimLOD)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

# 涓枃 Windows (CP936) 涓?vendored 搴撲腑鐨?UTF-8 瀛楃浼氬鑷?C4819/C2001
if(MSVC)
	add_compile_options(/utf-8)
endif()

option(SimLOD_CreateCUDASymlinks "Create symbolic links instead of copying CUDA files." OFF)

# Target SimLOD

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

## CUDA toolkit
find_package(CUDAToolkit 11.8 REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE
	${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
	CUDA::cuda_driver
	CUDA::nvrtc)

## CudaModularProgram 鐢?nvJitLink 鍋?LTO 閾炬帴锛圕UDA 11.8 绮剧畝瀹夎缂鸿缁勪欢锛?2.x 鑷甫锛?
if(TARGET CUDA::nvJitLink)
	target_link_libraries(${PROJECT_NAME} CUDA::nvJitLink)
endif()

## OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY})

## fmt 
add_subdirectory(thirdparty/fmt)
target_link_libraries(${PROJECT_NAME} fmt::fmt)

## glew
target_include_directories(${PROJECT_NAME} PRIVATE
	thirdparty/glew/include)
target_sources(${PROJECT_NAME} PRIVATE
	thirdparty/glew/glew.c)
# 鑷甫婧愮爜闈欐€佺紪璇?glew锛岄伩鍏嶅ご鏂囦欢 dllimport 澹版槑涓庣郴缁?glew32.lib 鍐茬獊 (LNK2005)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLEW_STATIC)

## GLFW
include(cmake/glfw.cmake)
target_include_directories(${PROJECT_NAME} PRIVATE
	${glfw_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} glfw)

## glm
target_include_directories(${PROJECT_NAME} PRIVATE
	thirdparty/glm)

## imgui
target_include_directories(${PROJECT_NAME} PRIVATE
	thirdparty/imgui
	thirdparty/imgui/backends)
target_sources(${PROJECT_NAME} PRIVATE
	thirdparty/imgui/imgui.cpp
	thirdparty/imgui/imgui_demo.cpp
	thirdparty/imgui/imgui_draw.cpp
	thirdparty/imgui/imgui_tables.cpp
	thirdparty/imgui/imgui_widgets.cpp
	thirdparty/imgui/backends/imgui_impl_glfw.cpp
	thirdparty/imgui/backends/imgui_impl_opengl3.cpp)

## implot
target_include_directories(${PROJECT_NAME} PRIVATE
	thirdparty/implot)
target_sources(${PROJECT_NAME} PRIVATE
	thirdparty/implot/implot_items.cpp
	thirdparty/implot/implot.cpp)

## laszip
add_subdirectory(thirdparty/laszip)
target_link_libraries(${PROJECT_NAME} laszip)
target_include_directories(${PROJECT_NAME} PRIVATE
	thirdparty/laszip)

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

