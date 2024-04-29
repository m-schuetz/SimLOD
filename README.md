## About

Load large, tiled data sets by quickly loading bounding boxes, then only loading points in those bounding boxes that appear large on screen. As you move, points that are not needed anymore are unloaded and new ones are loaded.

Goal: Also quickly load sparse subsample (every 50'000th point aka "chunk point") so that we can replace the bounding box with a higher-resolution subsample (~500 points per tile).

<img src="./docs/direct_vis_2.gif" />

Every 50'000th point corresponds to the compressed LAZ format, which compresses point clouds in chunks of 50k points. The first point of each chunk is uncompressed, and can therefore be easily loaded with random access. 

## Build

Dependencies: 
* Cuda 12.4
* Cmake 3.22
* Visual Studio 2022


Build commands:

```
mkdir build
cd build
cmake ../
```

Then change the path to the point cloud in main_progressive_octree.cpp.
```vector<string> files = listFiles("D:/resources/pointclouds/CA13_las_tmp");```