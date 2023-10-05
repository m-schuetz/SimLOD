
#include <filesystem>

#include "GLRenderer.h"
#include "Runtime.h"

namespace fs = std::filesystem;

auto _controls = make_shared<OrbitControls>();


static void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {

	if (
		severity == GL_DEBUG_SEVERITY_NOTIFICATION 
		|| severity == GL_DEBUG_SEVERITY_LOW 
		|| severity == GL_DEBUG_SEVERITY_MEDIUM
		) {
		return;
	}

	cout << "OPENGL DEBUG CALLBACK: " << message << endl;
}

void error_callback(int error, const char* description){
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){

	cout << "key: " << key << ", scancode: " << scancode << ", action: " << action << ", mods: " << mods << endl;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	Runtime::keyStates[key] = action;

	cout << key << endl;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}
	
	Runtime::mousePosition = {xpos, ypos};

	_controls->onMouseMove(xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}

	_controls->onMouseScroll(xoffset, yoffset);
}


void drop_callback(GLFWwindow* window, int count, const char **paths){
	for(int i = 0; i < count; i++){
		cout << paths[i] << endl;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){

	// cout << "start button: " << button << ", action: " << action << ", mods: " << mods << endl;

	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse){
		return;
	}

	// cout << "end button: " << button << ", action: " << action << ", mods: " << mods << endl;


	if(action == 1){
		Runtime::mouseButtons = Runtime::mouseButtons | (1 << button);
	}else if(action == 0){
		uint32_t mask = ~(1 << button);
		Runtime::mouseButtons = Runtime::mouseButtons & mask;
	}

	_controls->onMouseButton(button, action, mods);
}

GLRenderer::GLRenderer(){
	this->controls = _controls;
	camera = make_shared<Camera>();

	init();

	view.framebuffer = this->createFramebuffer(128, 128);
}

void GLRenderer::init(){
	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		// Initialization failed
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DECORATED, true);
	glfwWindowHint(GLFW_RESIZABLE, true);
	glfwWindowHint(GLFW_VISIBLE, true);

	int numMonitors;
	GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

	cout << "<create windows>" << endl;
	if (numMonitors > 1) {
		const GLFWvidmode * modeLeft = glfwGetVideoMode(monitors[0]);
		const GLFWvidmode * modeRight = glfwGetVideoMode(monitors[1]);

		window = glfwCreateWindow(1920, 1080, "stuff", nullptr, nullptr);
		// window = glfwCreateWindow(modeRight->width, modeRight->height - 300, "Simple example", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		// SECOND MONITOR
		//int xpos;
		//int ypos;
		// glfwGetMonitorPos(monitors[1], &xpos, &ypos);
		// glfwSetWindowPos(window, xpos, ypos);

		// FIRST
		// glfwSetWindowPos(window, 0, 0);
	} else {
		const GLFWvidmode * mode = glfwGetVideoMode(monitors[0]);

		window = glfwCreateWindow(mode->width - 100, mode->height - 100, "Simple example", nullptr, nullptr);

		if (!window) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetWindowPos(window, 50, 50);
	}

	cout << "<set input callbacks>" << endl;
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	// glfwSetDropCallback(window, drop_callback);

	static GLRenderer* ref = this;
	glfwSetDropCallback(window, [](GLFWwindow*, int count, const char **paths){

		vector<string> files;
		for(int i = 0; i < count; i++){
			string file = paths[i];
			files.push_back(file);
		}

		for(auto &listener : ref->fileDropListeners){
			listener(files);
		}
	});

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "glew error: %s\n", glewGetErrorString(err));
	}

	cout << "<glewInit done> " << "(" << now() << ")" << endl;

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	glDebugMessageCallback(debugCallback, NULL);

	{ // SETUP IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImPlot::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 450");
		ImGui::StyleColorsDark();
	}
}

shared_ptr<Texture> GLRenderer::createTexture(int width, int height, GLuint colorType) {

	auto texture = Texture::create(width, height, colorType, this);

	return texture;
}

shared_ptr<Framebuffer> GLRenderer::createFramebuffer(int width, int height) {

	auto framebuffer = Framebuffer::create(this);

	GLenum status = glCheckNamedFramebufferStatus(framebuffer->handle, GL_FRAMEBUFFER);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		cout << "framebuffer incomplete" << endl;
	}

	framebuffer->setSize(width, height);

	return framebuffer;
}

void GLRenderer::loop(function<void(void)> update, function<void(void)> render){

	int fpsCounter = 0;
	double start = now();
	double tPrevious = start;
	double tPreviousFPSMeasure = start;

	vector<float> frameTimes(1000, 0);

	while (!glfwWindowShouldClose(window)){

		// TIMING
		double timeSinceLastFrame;
		{
			double tCurrent = now();
			timeSinceLastFrame = tCurrent - tPrevious;
			tPrevious = tCurrent;

			double timeSinceLastFPSMeasure = tCurrent - tPreviousFPSMeasure;

			if(timeSinceLastFPSMeasure >= 1.0){
				this->fps = double(fpsCounter) / timeSinceLastFPSMeasure;

				tPreviousFPSMeasure = tCurrent;
				fpsCounter = 0;
			}
			frameTimes[frameCount % frameTimes.size()] = static_cast<float>(timeSinceLastFrame);
		}
		

		// WINDOW
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		camera->setSize(width, height);
		this->width = width;
		this->height = height;

		EventQueue::instance->process();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, this->width, this->height);

		glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
		glViewport(0, 0, this->width, this->height);


		{ 
			controls->update();

			camera->world = controls->world;
			camera->position = camera->world * dvec4(0.0, 0.0, 0.0, 1.0);
		}

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		{ // UPDATE & RENDER
			camera->update();
			update();
			camera->update();

			render();
		}

		// IMGUI
		{
			ImGui::SetNextWindowPos(ImVec2(10, 0));
			ImGui::SetNextWindowSize(ImVec2(490, 15));
			ImGui::Begin("Test", nullptr,
				ImGuiWindowFlags_NoTitleBar 
				| ImGuiWindowFlags_NoScrollbar
				| ImGuiWindowFlags_NoMove
				| ImGuiWindowFlags_NoResize
				| ImGuiWindowFlags_NoBackground
			);

			bool toggleGUI = ImGui::Button("Toggle GUI");
			if(toggleGUI){
				Runtime::showGUI = !Runtime::showGUI;
			}
			ImGui::End();
		}

		auto windowSize_perf = ImVec2(490, 260);

		if(Runtime::showGUI && Runtime::showPerfGraph)
		// if(false)
		{ // RENDER IMGUI PERFORMANCE WINDOW

			stringstream ssFPS; 
			ssFPS << this->fps;
			string strFps = ssFPS.str();

			ImGui::SetNextWindowPos(ImVec2(10, 30));
			ImGui::SetNextWindowSize(windowSize_perf);

			ImGui::Begin("Performance");
			string fpsStr = rightPad("FPS:", 30) + strFps;
			ImGui::Text(fpsStr.c_str());

			static float history = 2.0f;
			static ScrollingBuffer sFrames;
			static ScrollingBuffer s60fps;
			static ScrollingBuffer s120fps;
			float t = static_cast<float>(now());

			sFrames.AddPoint(t, 1000.0f * static_cast<float>(timeSinceLastFrame));

			// sFrames.AddPoint(t, 1000.0f * timeSinceLastFrame);
			s60fps.AddPoint(t, 1000.0f / 60.0f);
			s120fps.AddPoint(t, 1000.0f / 120.0f);
			static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
			ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
			ImPlot::SetNextPlotLimitsY(0, 30, ImGuiCond_Always);

			if (ImPlot::BeginPlot("Timings", nullptr, nullptr, ImVec2(-1, 200))){

				auto x = &sFrames.Data[0].x;
				auto y = &sFrames.Data[0].y;
				ImPlot::PlotShaded("frame time(ms)", x, y, sFrames.Data.size(), -Infinity, sFrames.Offset, 2 * sizeof(float));

				ImPlot::PlotLine("16.6ms (60 FPS)", &s60fps.Data[0].x, &s60fps.Data[0].y, s60fps.Data.size(), s60fps.Offset, 2 * sizeof(float));
				ImPlot::PlotLine(" 8.3ms (120 FPS)", &s120fps.Data[0].x, &s120fps.Data[0].y, s120fps.Data.size(), s120fps.Offset, 2 * sizeof(float));

				ImPlot::EndPlot();
			}

			ImGui::End();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		auto source = view.framebuffer;
		glBlitNamedFramebuffer(
			source->handle, 0,
			0, 0, source->width, source->height,
			0, 0, 0 + source->width, 0 + source->height,
			GL_COLOR_BUFFER_BIT, GL_LINEAR);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, this->width, this->height);


		glfwSwapBuffers(window);
		glfwPollEvents();

		fpsCounter++;
		frameCount++;
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}

shared_ptr<Framebuffer> Framebuffer::create(GLRenderer* renderer) {

	auto fbo = make_shared<Framebuffer>();
	fbo->renderer = renderer;

	glCreateFramebuffers(1, &fbo->handle);

	{ // COLOR ATTACHMENT 0

		auto texture = renderer->createTexture(fbo->width, fbo->height, GL_RGBA8);
		fbo->colorAttachments.push_back(texture);

		glNamedFramebufferTexture(fbo->handle, GL_COLOR_ATTACHMENT0, texture->handle, 0);
	}

	{ // DEPTH ATTACHMENT

		auto texture = renderer->createTexture(fbo->width, fbo->height, GL_DEPTH_COMPONENT32F);
		fbo->depth = texture;

		glNamedFramebufferTexture(fbo->handle, GL_DEPTH_ATTACHMENT, texture->handle, 0);
	}

	fbo->setSize(128, 128);

	return fbo;
}

shared_ptr<Texture> Texture::create(int width, int height, GLuint colorType, GLRenderer* renderer){

	GLuint handle;
	glCreateTextures(GL_TEXTURE_2D, 1, &handle);

	auto texture = make_shared<Texture>();
	texture->renderer = renderer;
	texture->handle = handle;
	texture->colorType = colorType;

	texture->setSize(width, height);

	return texture;
}

void Texture::setSize(int width, int height) {

	bool needsResize = this->width != width || this->height != height;

	if (needsResize) {

		glDeleteTextures(1, &this->handle);
		glCreateTextures(GL_TEXTURE_2D, 1, &this->handle);

		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTextureStorage2D(this->handle, 1, this->colorType, width, height);

		this->width = width;
		this->height = height;
	}

}