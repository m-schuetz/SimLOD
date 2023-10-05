cmake_minimum_required(VERSION 3.14)

include(FetchContent)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

FetchContent_Declare(
		glfw
		GIT_REPOSITORY      https://github.com/glfw/glfw.git
		GIT_TAG             3.3.2
)

FetchContent_MakeAvailable(glfw)
