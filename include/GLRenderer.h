
#pragma once

#include <functional>
#include <vector>
#include <string>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "implot.h"
#include "implot_internal.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"
#include "OrbitControls.h"

using namespace std;
using glm::dvec3;
using glm::dvec4;
using glm::dmat4;

struct GLRenderer;

// ScrollingBuffer from ImPlot implot_demo.cpp.
// MIT License
// url: https://github.com/epezent/implot
struct ScrollingBuffer {
	int MaxSize;
	int Offset;
	ImVector<ImVec2> Data;
	ScrollingBuffer() {
		MaxSize = 2000;
		Offset = 0;
		Data.reserve(MaxSize);
	}
	void AddPoint(float x, float y) {
		if (Data.size() < MaxSize)
			Data.push_back(ImVec2(x, y));
		else {
			Data[Offset] = ImVec2(x, y);
			Offset = (Offset + 1) % MaxSize;
		}
	}
	void Erase() {
		if (Data.size() > 0) {
			Data.shrink(0);
			Offset = 0;
		}
	}
};

struct GLBuffer{

	GLuint handle = -1;
	int64_t size = 0;

};

struct Texture {

	GLRenderer* renderer = nullptr;
	GLuint handle = -1;
	GLuint colorType = -1;
	int width = 0;
	int height = 0;

	static shared_ptr<Texture> create(int width, int height, GLuint colorType, GLRenderer* renderer);

	void setSize(int width, int height);

};

struct Framebuffer {

	vector<shared_ptr<Texture>> colorAttachments;
	shared_ptr<Texture> depth;
	GLuint handle = -1;
	GLRenderer* renderer = nullptr;

	int width = 0;
	int height = 0;

	Framebuffer() {
		
	}

	static shared_ptr<Framebuffer> create(GLRenderer* renderer);

	void setSize(int width, int height) {


		bool needsResize = this->width != width || this->height != height;

		if (needsResize) {

			// COLOR
			for (int i = 0; i < this->colorAttachments.size(); i++) {
				auto& attachment = this->colorAttachments[i];
				attachment->setSize(width, height);
				glNamedFramebufferTexture(this->handle, GL_COLOR_ATTACHMENT0 + i, attachment->handle, 0);
			}

			{ // DEPTH
				this->depth->setSize(width, height);
				glNamedFramebufferTexture(this->handle, GL_DEPTH_ATTACHMENT, this->depth->handle, 0);
			}
			
			this->width = width;
			this->height = height;
		}
		

	}

};

struct View{
	dmat4 view;
	dmat4 proj;
	shared_ptr<Framebuffer> framebuffer = nullptr;
};

struct Camera{

	glm::dvec3 position;
	glm::dmat4 rotation;

	glm::dmat4 world;
	glm::dmat4 view;
	glm::dmat4 proj;

	double aspect = 1.0;
	double fovy = 60.0;
	double near = 0.1;
	double far = 2'000'000.0;
	int width = 128;
	int height = 128;

	Camera(){

	}

	void setSize(int width, int height){
		this->width = width;
		this->height = height;
		this->aspect = double(width) / double(height);
	}

	void update(){
		view =  glm::inverse(world);

		double pi = glm::pi<double>();
		proj = glm::perspective(pi * fovy / 180.0, aspect, near, far);
	}


};



struct GLRenderer{

	GLFWwindow* window = nullptr;
	double fps = 0.0;
	int64_t frameCount = 0;
	
	shared_ptr<Camera> camera = nullptr;
	shared_ptr<OrbitControls> controls = nullptr;

	bool vrEnabled = false;
	
	View view;

	vector<function<void(vector<string>)>> fileDropListeners;

	int width = 0;
	int height = 0;
	string selectedMethod = "";

	GLRenderer();

	void init();

	shared_ptr<Texture> createTexture(int width, int height, GLuint colorType);

	shared_ptr<Framebuffer> createFramebuffer(int width, int height);

	inline GLBuffer createBuffer(int64_t size){
		GLuint handle;
		glCreateBuffers(1, &handle);
		glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);

		GLBuffer buffer;
		buffer.handle = handle;
		buffer.size = size;

		return buffer;
	}

	inline GLBuffer createSparseBuffer(int64_t size){
		GLuint handle;
		glCreateBuffers(1, &handle);
		glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT | GL_SPARSE_STORAGE_BIT_ARB );

		GLBuffer buffer;
		buffer.handle = handle;
		buffer.size = size;

		return buffer;
	}

	inline GLBuffer createUniformBuffer(int64_t size){
		GLuint handle;
		glCreateBuffers(1, &handle);
		glNamedBufferStorage(handle, size, 0, GL_DYNAMIC_STORAGE_BIT );

		GLBuffer buffer;
		buffer.handle = handle;
		buffer.size = size;

		return buffer;
	}

	inline shared_ptr<Buffer> readBuffer(GLBuffer glBuffer, uint32_t offset, uint32_t size){

		auto target = make_shared<Buffer>(size);

		glGetNamedBufferSubData(glBuffer.handle, offset, size, target->data);

		return target;
	}

	void loop(function<void(void)> update, function<void(void)> render);

	void onFileDrop(function<void(vector<string>)> callback){
		fileDropListeners.push_back(callback);
	}

};