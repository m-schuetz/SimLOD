cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)


project(CreateTraining)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/include/unsuck_platform_specific.cpp
    ${PROJECT_SOURCE_DIR}/src/main_create_training.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE include)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} laszip)
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/laszip)


project(SimLOD)

include(FetchContent)

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
	CUDAToolkit_INCLUDE_DIRS)

target_link_libraries(${PROJECT_NAME}
	CUDA::cuda_driver
	CUDA::nvrtc)

## OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY})

## fmt 
add_subdirectory(libs/fmt)
target_link_libraries(${PROJECT_NAME} fmt::fmt)

## glew
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/glew/include)
target_sources(${PROJECT_NAME} PRIVATE
	libs/glew/glew.c)

## GLFW
include(cmake/glfw.cmake)
target_include_directories(${PROJECT_NAME} PRIVATE
	${glfw_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} glfw)

## glm
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/glm)

## imgui
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/imgui
	libs/imgui/backends)
target_sources(${PROJECT_NAME} PRIVATE
	libs/imgui/imgui.cpp
	libs/imgui/imgui_demo.cpp
	libs/imgui/imgui_draw.cpp
	libs/imgui/imgui_tables.cpp
	libs/imgui/imgui_widgets.cpp
	libs/imgui/backends/imgui_impl_glfw.cpp
	libs/imgui/backends/imgui_impl_opengl3.cpp)

## implot
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/implot)
target_sources(${PROJECT_NAME} PRIVATE
	libs/implot/implot_items.cpp
	libs/implot/implot.cpp)

## laszip
add_subdirectory(libs/laszip)
target_link_libraries(${PROJECT_NAME} laszip)
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/laszip)

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

# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)../..")


