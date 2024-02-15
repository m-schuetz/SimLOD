
#include <string>
#include <filesystem>
#include <execution>
#include <atomic>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <format>
#include <sstream>

#include "laszip_api.h"

#include "unsuck.hpp"

using namespace std;

namespace fs = filesystem;

//constexpr int numQueryPoints = 10; // 10'000;
constexpr int numQueryPoints = 10'000;
constexpr double pixelSize     = 10.0;
constexpr int heightmapSize    = 64;


mutex mtx;

//string pointcloud_dir = "E:\\datasets\\point clouds\\CA13_SAN_SIM_small";
string pointcloud_dir = "E:\\datasets\\point clouds\\CA13_SAN_SIM";

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
	int64_t numPoints;
};

struct Heightmap{
	Point queryPoint;
	vec3 world_min;
	vec3 world_max;
	double sum[heightmapSize * heightmapSize];
	int count[heightmapSize * heightmapSize];
	float values[heightmapSize * heightmapSize];
};

vector<Point> loadPoints(LasFile lasfile){
	laszip_POINTER laszip_reader = nullptr;
	laszip_header* header = nullptr;
	laszip_point* laz_point = nullptr;
	laszip_BOOL is_compressed;
	laszip_BOOL request_reader = true;

	laszip_create(&laszip_reader);
	laszip_request_compatibility_mode(laszip_reader, request_reader);
	laszip_open_reader(laszip_reader, lasfile.file.c_str(), &is_compressed);
	laszip_get_header_pointer(laszip_reader, &header);

	laszip_get_point_pointer(laszip_reader, &laz_point);

	vector<Point> points;
	points.reserve(lasfile.numPoints);

	double XYZ[3];
	for(int pointIndex = 0; pointIndex < lasfile.numPoints; pointIndex++){
		laszip_read_point(laszip_reader);
		laszip_get_coordinates(laszip_reader, XYZ);

		Point point;
		point.x = XYZ[0];
		point.y = XYZ[1];
		point.z = XYZ[2];

		points.push_back(point);
	}

	laszip_close_reader(laszip_reader);

	return points;
}

void saveHeightmaps(vector<Heightmap>& heightmaps, string outPath){

	// heightmap file format:
	// - all heightmaps, one after the other [heightmap_0][heightmap_1]...
	// - Each heightmap starts with the query point as x, y, z in double precision coordinates
	// - followed by heightmapSize*heightmapSize floats, each representing the height of a pixel
	// - stored row-wise
	//     - pixelIndex = x + heightmapSize * y
	int heightmapByteSize = 3 * sizeof(double) + heightmapSize * heightmapSize * sizeof(float);
	int buffersize = heightmaps.size() * heightmapByteSize;
		
	Buffer buffer(buffersize);

	for(int heightmapIndex = 0; heightmapIndex < heightmaps.size(); heightmapIndex++)
	{
		Heightmap& heightmap = heightmaps[heightmapIndex];
		int heightmapByteOffset = heightmapIndex * heightmapByteSize;

		// Write query point
		buffer.set<double>(heightmap.queryPoint.x, heightmapByteOffset +  0);
		buffer.set<double>(heightmap.queryPoint.y, heightmapByteOffset +  8);
		buffer.set<double>(heightmap.queryPoint.z, heightmapByteOffset + 16);

		for(int px = 0; px < heightmapSize; px++)
		for(int py = 0; py < heightmapSize; py++)
		{
			int pixelID = px + heightmapSize * py;

			// float height = heightmap.values[pixelID];
			float height = heightmap.sum[pixelID] / double(heightmap.count[pixelID]);
			if (heightmap.count[pixelID] == 0)
				height = std::numeric_limits<float>::quiet_NaN();

			int R = 0;
			int G = 255;
			int B = 0;

			if(isnan(height)){
				R = 255;
				G = 0;
				B = 0;
			}

			double heightmap_meters = double(heightmapSize) * pixelSize;
			double x = double(px) * pixelSize + heightmap.world_min.x;
			double y = double(py) * pixelSize + heightmap.world_min.y;

			buffer.set<float>(height, heightmapByteOffset + 24 + pixelID * 4);
		}
	}

	printfmt("writing file {} \n", outPath);
	writeBinaryFile(outPath, buffer);
}

void dbg_dumpHeightmap(Heightmap& heightmap, string filename){

	stringstream ss;

	// dump query point
	ss << format("{:.2f}, {:.2f}, {:.2f}, {}, {}, {} \n", 
		heightmap.queryPoint.x, heightmap.queryPoint.y, heightmap.queryPoint.z, 
		255, 255, 0);

	// dump height values as colored points
	for(int px = 0; px < heightmapSize; px++)
	for(int py = 0; py < heightmapSize; py++)
	{
		int pixelID = px + heightmapSize * py;

		//float height = heightmap.values[pixelID];
		float height = heightmap.sum[pixelID] / double(heightmap.count[pixelID]);
		if (heightmap.count[pixelID] == 0)
			height = std::numeric_limits<float>::quiet_NaN();

		int R = 0;
		int G = 255;
		int B = 0;

		if(isnan(height)){
			R = 255;
			G = 0;
			B = 0;
		}

		double heightmap_meters = double(heightmapSize) * pixelSize;
		double x = double(px) * pixelSize + heightmap.world_min.x;
		double y = double(py) * pixelSize + heightmap.world_min.y;
		double z = isnan(height) ? 0.0 : height;

		ss << format("{:.2f}, {:.2f}, {:.2f}, {}, {}, {} \n", x, y, z, R, G, B);
	}

	string str = ss.str();

	// printfmt("writing file {} \n", filename);
	writeFile(filename, str);
}

vector<Point> loadQueryPoints(vector<LasFile>& lasfiles, int64_t numPointsTotal) {

	// Generate list of random numbers to use as query points
	unordered_set<int64_t> queryPointIndicesSet;
	while(queryPointIndicesSet.size() < numQueryPoints){

		// not sure if random is inclusive/exclusive last one, 
		// but it really doesn't matter for a random selection of billions of points,
		// so just exclude it explicitly.
		int64_t index = random(0.0, double(numPointsTotal) - 1.0);

		queryPointIndicesSet.insert(index);
	}

	vector<int64_t> queryPointIndices;
	queryPointIndices.insert(queryPointIndices.end(), queryPointIndicesSet.begin(), queryPointIndicesSet.end());

	// load the query points
	vector<Point> queryPoints;
	for_each(execution::par, queryPointIndices.begin(), queryPointIndices.end(),
		[&](int64_t globalPointIndex) {

			// find file to which point belongs
			int64_t globalPointCounter = 0;
			int64_t pointIndex_withinFile = -1;
			int64_t lasfileIndex = 0;
			for (auto& lasfile : lasfiles) {
				if (globalPointCounter <= globalPointIndex && globalPointIndex < globalPointCounter + lasfile.numPoints) {
					// point is in this file
					pointIndex_withinFile = globalPointIndex - globalPointCounter;
					break;
				}
				else {
					globalPointCounter += lasfile.numPoints;
				}

				lasfileIndex++;
			}

			if (pointIndex_withinFile == -1) {
				printfmt("error at {}:{} \n", __FILE__, __LINE__);
				exit(123);
			}

			LasFile& lasfile = lasfiles[lasfileIndex];

			laszip_POINTER laszip_reader = nullptr;
			laszip_header* header = nullptr;
			laszip_point* laz_point = nullptr;
			laszip_BOOL is_compressed;
			laszip_BOOL request_reader = true;

			laszip_create(&laszip_reader);
			laszip_request_compatibility_mode(laszip_reader, request_reader);
			laszip_open_reader(laszip_reader, lasfile.file.c_str(), &is_compressed);
			laszip_get_header_pointer(laszip_reader, &header);

			laszip_get_point_pointer(laszip_reader, &laz_point);
			laszip_seek_point(laszip_reader, pointIndex_withinFile);

			double XYZ[3];
			laszip_read_point(laszip_reader);
			laszip_get_coordinates(laszip_reader, XYZ);

			Point point;
			point.x = XYZ[0];
			point.y = XYZ[1];
			point.z = XYZ[2];

			auto rgb = laz_point->rgb;
			point.rgba[0] = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
			point.rgba[1] = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
			point.rgba[2] = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];

			laszip_close_reader(laszip_reader);

			mtx.lock();
			queryPoints.push_back(point);
			mtx.unlock();
		}
	);

	printfmt("query points: \n");
	for (int i = 0; i < 10; i++) {
		Point point = queryPoints[i];
		printfmt("{:.2f}, {:.2f}, {:.2f} \n", point.x, point.y, point.z);
	}
	printfmt("... [{}] \n", numQueryPoints);

	return queryPoints;
}

struct LasFilesData {
	vec3 min = { Infinity, Infinity, Infinity };
	vec3 max = { -Infinity, -Infinity, -Infinity };
	vec3 size = {0.0, 0.0, 0.0};
	uint64_t numPointsTotal = 0;
	uint64_t numFilesProcessed = 0;
	vector<LasFile> lasfiles;
};

LasFilesData gatherLasfileData(vector<string> lasLazFiles) {

	LasFilesData data;

	for_each(execution::par, lasLazFiles.begin(), lasLazFiles.end(),
		[&](string file) {
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

			mtx.lock();

			data.min.x = std::min(data.min.x, header->min_x);
			data.min.y = std::min(data.min.y, header->min_y);
			data.min.z = std::min(data.min.z, header->min_z);
			data.max.x = std::max(data.max.x, header->max_x);
			data.max.y = std::max(data.max.y, header->max_y);
			data.max.z = std::max(data.max.z, header->max_z);

			data.numPointsTotal += numPoints;

			data.numFilesProcessed++;

			LasFile lasfile;
			lasfile.min = { header->min_x, header->min_y, header->min_z };
			lasfile.max = { header->max_x, header->max_y, header->max_z };
			lasfile.file = file;
			lasfile.numPoints = numPoints;

			data.lasfiles.push_back(lasfile);

			mtx.unlock();
		}
	);

	data.size = {
		data.max.x - data.min.x, 
		data.max.y - data.min.y, 
		data.max.z - data.min.z
	};

	return data;
}

int main() {

	auto files = listFiles(pointcloud_dir);
	vector<string> lasLazFiles;

	// filter las and laz files
	for(string file : files){
		if(iEndsWith(file, "las")) lasLazFiles.push_back(file);
		if(iEndsWith(file, "laz")) lasLazFiles.push_back(file);
	}

	// gather some data about all point clouds
	auto [
		min, max, size,
		numPointsTotal, numFilesProcessed, 
		lasfiles
	] = gatherLasfileData(lasLazFiles);

	printfmt("min:  {:.2f}, {:.2f}, {:.2f} \n", min.x, min.y, min.z);
	printfmt("max:  {:.2f}, {:.2f}, {:.2f} \n", max.x, max.y, max.z);
	printfmt("size: {:.2f}, {:.2f}, {:.2f} \n", size.x, size.y, size.z);
	printfmt("#points: {:L} \n", numPointsTotal);

	// double pixelSize     = 10.0;
	// int heightmap_width  = ceil(size.x / pixelSize);
	// int heightmap_height = ceil(size.y / pixelSize);
	// int numPixels = heightmap_width * heightmap_height;

	// printfmt("heightmap size: {:L} x {:L} \n", heightmap_width, heightmap_height);

	// if(numPixels > 15'000 * 15'000){
	// 	printfmt("pretty large amounts of heightmap pixels: {:L}", numPixels);
	// 	printfmt("Aborting. Make sure this is correct, and adapt {}:{}", __FILE__, __LINE__);

	// 	return 123;
	// }

	vector<Point> queryPoints = loadQueryPoints(lasfiles, numPointsTotal);
	
	// Create heightmaps.
	// Also, for each tile, create a list of query points it affects
	printfmt("allocate heightmaps and find out which tiles affect them \n");
	vector<Heightmap> heightmaps;
	vector<vector<int>> affectedQueryPointsPerTile(lasfiles.size());
	for(int queryPointIndex = 0; queryPointIndex < queryPoints.size(); queryPointIndex++){

		Point point = queryPoints[queryPointIndex];

		double heightmap_meters = double(heightmapSize) * pixelSize;

		// all tiles within that bounding box contribute to this point's height map
		vec3 heightmap_worldspace_min = {
			point.x - heightmap_meters / 2.0,
			point.y - heightmap_meters / 2.0,
			0.0
		};
		vec3 heightmap_worldspace_max = {
			point.x + heightmap_meters / 2.0,
			point.y + heightmap_meters / 2.0,
			0.0
		};

		Heightmap heightmap;
		heightmap.queryPoint = point;
		heightmap.world_min = heightmap_worldspace_min;
		heightmap.world_max = heightmap_worldspace_max;
		for(int pixelID = 0; pixelID < heightmapSize * heightmapSize; pixelID++){
			heightmap.values[pixelID] = std::numeric_limits<float>::quiet_NaN();
			heightmap.count[pixelID] = 0;
			heightmap.sum[pixelID] = 0.0;
		}
		heightmaps.push_back(heightmap);

		for(int tileIndex = 0; tileIndex < lasfiles.size(); tileIndex++){
			LasFile& tile = lasfiles[tileIndex];

			bool intersects_x = heightmap_worldspace_max.x > tile.min.x && heightmap_worldspace_min.x < tile.max.x;
			bool intersects_y = heightmap_worldspace_max.y > tile.min.y && heightmap_worldspace_min.y < tile.max.y;
			bool intersects = intersects_x && intersects_y;

			if(intersects){
				affectedQueryPointsPerTile[tileIndex].push_back(queryPointIndex);
			}
		}
	}

	// iterate through all tiles and let them update all heightmaps they affect
	printfmt("Iterating through tiles and updating affected heightmaps \n");
	std::vector<int> range(affectedQueryPointsPerTile.size());
	std::iota(range.begin(), range.end(), 0);
	int64_t numProcessedPoints = 0;
	int64_t numProcessedSinceLastMsg = 0;
	// for(int tileIndex = 0; tileIndex < affectedQueryPointsPerTile.size(); tileIndex++)
	for_each(execution::par, range.begin(), range.end(),
	[&](int tileIndex){
		LasFile& tile = lasfiles[tileIndex];
		vector<int> affectedQueryPoints = affectedQueryPointsPerTile[tileIndex];

		if (affectedQueryPoints.size() > 0)
		{
			vector<Point> points = loadPoints(tile);

			for (int queryPointIndex : affectedQueryPoints) {
				Heightmap& heightmap = heightmaps[queryPointIndex];
				double heightmap_world_size = double(heightmapSize) * pixelSize;

				for (Point point : points) {
					double u = (point.x - heightmap.world_min.x) / heightmap_world_size;
					double v = (point.y - heightmap.world_min.y) / heightmap_world_size;

					int px = double(heightmapSize) * u;
					int py = double(heightmapSize) * v;

					if (px < 0 || px > heightmapSize - 1) continue;
					if (py < 0 || py > heightmapSize - 1) continue;

					int pixelID = px + heightmapSize * py;

					heightmap.count[pixelID]++;
					heightmap.sum[pixelID] += point.z;
					if (isnan(heightmap.values[pixelID])) {
						heightmap.values[pixelID] = point.z;
					}
					else {
						heightmap.values[pixelID] = std::max(heightmap.values[pixelID], float(point.z));
					}
				}
			}
		}

		mtx.lock();

		numProcessedPoints += tile.numPoints;
		numProcessedSinceLastMsg += tile.numPoints;
		if(numProcessedSinceLastMsg > 100'000'000){
			printfmt("#processed points: {:L} \n", numProcessedPoints);
			numProcessedSinceLastMsg = 0;
		}

		mtx.unlock();

	});

	{// Now write the resulting heightmaps to disk
		printfmt("Storing results on disk \n");
		string outDir = "./results";
		fs::create_directories(outDir);
		string canonicalDir = fs::canonical(fs::path(outDir)).string();
		printfmt("writing results to {} \n", canonicalDir);

		for(int i = 0; i < heightmaps.size(); i++){
			Heightmap& heightmap = heightmaps[i];

			
			string dbgPath = format("{}/heightmap_{}.csv", outDir, i);

			dbg_dumpHeightmap(heightmap, dbgPath);
		}

		string heightmapsPath = format("{}/heightmaps.bin", outDir);
		saveHeightmaps(heightmaps, heightmapsPath);
	}

	printfmt("done! \n");
	
	return 0;
}