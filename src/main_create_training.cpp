
#include <string>
#include <filesystem>
#include <execution>
#include <atomic>

#include "laszip_api.h"

#include "unsuck.hpp"

using namespace std;

namespace fs = filesystem;


string pointcloud_dir = "E:/resources/pointclouds/CA13_las";

struct Point{
	double x, y, z;
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
};

struct vec3 {
	double x, y, z;
};

struct LasFile{
	string file;
	vec3 min;
	vec3 max;
};

int main() {

	auto files = listFiles(pointcloud_dir);
	vector<string> lasLazFiles;

	// filter las and laz files
	for(string file : files){
		if(iEndsWith(file, "las")) lasLazFiles.push_back(file);
		if(iEndsWith(file, "laz")) lasLazFiles.push_back(file);
	}

	// gather some data about all point clouds
	mutex mtx_results;

	vec3 min = { Infinity, Infinity, Infinity };
	vec3 max = { -Infinity, -Infinity, -Infinity };
	uint64_t numPointsTotal = 0;
	uint64_t numFilesProcessed = 0;
	vector<LasFile> lasfiles;

	for_each(execution::par, lasLazFiles.begin(), lasLazFiles.end(),
		[&](string file){
			laszip_POINTER laszip_reader = nullptr;
			laszip_header* header = nullptr;
			laszip_point* laz_point = nullptr;

			laszip_BOOL is_compressed;
			laszip_BOOL request_reader = true;

			laszip_create(&laszip_reader);
			laszip_request_compatibility_mode(laszip_reader, request_reader);
			laszip_open_reader(laszip_reader, file.c_str(), &is_compressed);

			laszip_get_header_pointer(laszip_reader, &header);

			uint64_t numPoints = std::max(uint64_t(header->number_of_point_records), header->extended_number_of_point_records);

			laszip_close_reader(laszip_reader);

			mtx_results.lock();

			min.x = std::min(min.x, header->min_x);
			min.y = std::min(min.y, header->min_y);
			min.z = std::min(min.z, header->min_z);
			max.x = std::max(max.x, header->max_x);
			max.y = std::max(max.y, header->max_y);
			max.z = std::max(max.z, header->max_z);

			numPointsTotal += numPoints;

			numFilesProcessed++;

			LasFile lasfile;
			lasfile.min = {header->min_x, header->min_y, header->min_z};
			lasfile.max = {header->max_x, header->max_y, header->max_z};
			lasfile.file = file;

			lasfiles.push_back(lasfile);

			mtx_results.unlock();
		} 
	);

	vec3 size = {
		max.x - min.x, 
		max.y - min.y, 
		max.z - min.z
	};

	printfmt("min:  {:.2f}, {:.2f}, {:.2f} \n", min.x, min.y, min.z);
	printfmt("max:  {:.2f}, {:.2f}, {:.2f} \n", max.x, max.y, max.z);
	printfmt("size: {:.2f}, {:.2f}, {:.2f} \n", size.x, size.y, size.z);
	printfmt("#points: {:L} \n", numPointsTotal);

	double pixelSize     = 10.0;
	int heightmap_width  = ceil(size.x / pixelSize);
	int heightmap_height = ceil(size.y / pixelSize);
	int numPixels = heightmap_width * heightmap_height;

	printfmt("heightmap size: {:L} x {:L} \n", heightmap_width, heightmap_height);

	if(numPixels > 15'000 * 15'000){
		printfmt("pretty large amounts of heightmap pixels: {:L}", numPixels);
		printfmt("Aborting. Make sure this is correct, and adapt {}:{}", __FILE__, __LINE__);

		return 123;
	}

	printfmt("sort tiles along x axis \n");
	std::sort(lasfiles.begin(), lasfiles.end(), [](LasFile& a, LasFile& b) {
		return a.min.x < b.min.x;
	});


	






	return 0;
	
	// printfmt("gathering data \n");
	// for (string file : lasLazFiles) {
	// 	laszip_POINTER laszip_reader = nullptr;
	// 	laszip_header* header = nullptr;
	// 	laszip_point* laz_point = nullptr;

	// 	laszip_BOOL is_compressed;
	// 	laszip_BOOL request_reader = true;

	// 	laszip_create(&laszip_reader);
	// 	laszip_request_compatibility_mode(laszip_reader, request_reader);
	// 	laszip_open_reader(laszip_reader, file.c_str(), &is_compressed);

	// 	laszip_get_header_pointer(laszip_reader, &header);

	// 	uint64_t numPoints = std::max(uint64_t(header->number_of_point_records), header->extended_number_of_point_records);

	// 	min.x = std::min(min.x, header->min_x);
	// 	min.y = std::min(min.y, header->min_y);
	// 	min.z = std::min(min.z, header->min_z);
	// 	max.x = std::max(max.x, header->max_x);
	// 	max.y = std::max(max.y, header->max_y);
	// 	max.z = std::max(max.z, header->max_z);

	// 	numPointsTotal += numPoints;
	// 	if(numFilesProcessed % 100 == 0){
	// 		printfmt("file {}\n", numFilesProcessed);
	// 	}

	// 	laszip_close_reader(laszip_reader);

	// 	numFilesProcessed++;
	// }

	// printfmt("min: {:.2f}, {:.2f}, {:.2f} \n", min.x, min.y, min.z);
	// printfmt("max: {:.2f}, {:.2f}, {:.2f} \n", max.x, max.y, max.z);
	// printfmt("#points: {:L} \n", numPointsTotal);

	// return 0;

	// for(string file : lasLazFiles){
	// 	printfmt("{} \n", file);

	// 	{
	// 		laszip_POINTER laszip_reader = nullptr;
	// 		laszip_header* header = nullptr;
	// 		laszip_point* laz_point = nullptr;

	// 		laszip_BOOL is_compressed;
	// 		laszip_BOOL request_reader = true;

	// 		laszip_create(&laszip_reader);
	// 		laszip_request_compatibility_mode(laszip_reader, request_reader);
	// 		laszip_open_reader(laszip_reader, file.c_str(), &is_compressed);

	// 		laszip_get_header_pointer(laszip_reader, &header);
	// 		laszip_get_point_pointer(laszip_reader, &laz_point);

	// 		uint64_t numPoints = std::max(uint64_t(header->number_of_point_records), header->extended_number_of_point_records);
			
	// 		vec3 boxMin = {header->min_x, header->min_y, header->min_z};

	// 		for (int i = 0; i < numPoints; i++) {
	// 			double XYZ[3];
	// 			laszip_read_point(laszip_reader);
	// 			laszip_get_coordinates(laszip_reader, XYZ);

	// 			Point point;
	// 			point.x = XYZ[0];
	// 			point.y = XYZ[1];
	// 			point.z = XYZ[2];

	// 			auto rgb = laz_point->rgb;
	// 			point.rgba[0] = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
	// 			point.rgba[1] = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
	// 			point.rgba[2] = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];

	// 			printfmt("{:.2f}, {:.2f}, {:.2f} \n", point.x, point.y, point.z);

	// 			break;
	// 		}

	// 		laszip_close_reader(laszip_reader);
	// 	}

	// 	break;
	// }

	


	return 0;
}