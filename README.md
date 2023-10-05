# SimLOD: Simultaneous LOD Generation and Rendering

This project loads point clouds, creates an LOD structure on-the-fly, and immediatelly displays the results in real time. This repository contains the source code for the paper "SimLOD: Simultaneous LOD Generation and Rendering".

On an RTX 4090 and PCIe 5.0 SSD (12GB/s), we are able to load and real-time render point clouds at rates of about 200-300 million points per second (MP/s) from the LAS format, 30 MP/s from compressed LAZ files, and up to 580 MP/s from an optimal XYZRGBA(16bytes/point) format.

Right now we only display data sets that fit in GPU memory, but support for arbitarily large point clouds is in development.

<table>
<tr style="border: none">
	<td><img src="./docs/loading.gif"/></td>
	<td><img src="./docs/generated_highres.gif"/></td>
</tr>
<tr style="border: none">
	<td colspan="2" style="border: none">
	Figure: Drag&Dropping 680 million points (11GB). Loaded in 1.7 seconds (400 MP/s; 6.4GB/s). 
	</td>
</tr>
</table>

The generated LOD structure is an octree that stores voxels in inner nodes and original point cloud data in leaf nodes. Voxels are created by sampling on a 128³ grid. Due to the sparsity in surfacic data sets, the voxels themselves are stored in lists instead of grids. In order to support growing amounts of voxels and points as new data is inserted into the octree, we use linked lists of chunks of points/voxels. 

<table>
<tr style="border: none">
	<td><img src="./docs/inner_node.jpg"/></td>
	<td><img src="./docs/leaf_node.jpg"/></td>
	<td><img src="./docs/lod_frustum.jpg"/></td>
	<td><img src="./docs/nodes_n_chunks.png"/></td>
</tr>
<tr style="border: none">
	<td>(a) Inner node with voxels.</td>
	<td>(b) Leaf node with points.</td>
	<td>(c) LOD </td>
	<td>(d) Point/Voxel storage via linked-lists</td>
</tr>
<tr style="border: none">
	<td colspan="4" style="border: none">
	Figure: Inner nodes comprise representative voxels with a resolution of 128³ and leaf nodes store the full-precision point cloud data. We render the LOD at a resolution that results in pixel-sized voxels to give the impression of looking at the full-res data. 
	</td>
</tr>
</table>
<!-- 
<table>
<tr style="border: none">
	<td><img src="./docs/lod_frustum.jpg"/></td>
	<td><img src="./docs/nodes_n_chunks.png"/></td>
</tr>
<tr style="border: none">
	<td colspan="2" style="border: none">
	Figure: Left: Selection of octree nodes for the current viewpoint. Higher-level nodes close to the camera, lower-level nodes in the distance. Right: Points/Voxels are stored in linked lists of chunks.
	</td>
</tr>
</table> -->

During rendering, a CUDA kernel first computes a list of visible octree nodes. Each node is then rendered by one workgroup that iterates through the node's chunks and the stored points and voxels, and draws them with atomic operations. 



## Getting Started

### Install Dependencies

* CUDA Toolkit 11.8

### Build & Run

#### Prebuilt windows binaries

* TODO: Add link to binary release.

#### Windows with Visual Studio 2022

* Create Visual Studio 2022 project files:

```
mkdir build
cd build
cmake ../ -G "Visual Studio 17 2022" -A x64
```

* Open build/SimLOD.sln
* Switch to Release mode
* Compile and Run (Ctrl + F5)
* Drag&Drop point clouds in *.las, *.laz or *.simlod format into the application.

#### Windows or Linux with CMake

```
mkdir out && cd out         # Create a build directory
cmake .. && make            # Configure CMake and build the SimLOD target
./SimLOD                    # Run SimLOD
```


### Notes

* The environment variable ```CUDA_PATH``` needs to point to the install location of CUDA Toolkit 11.8 (e.g., ```/usr/local/cuda-11.8```)
* If you want to modify and hot reload CUDA code at runtime, make sure to set the work directory to the top-most folder of this repository instead of the binary target path. The project loads cuda files located relative to "./modules".

## Software Architecture


<table>
	<tr>
		<th>File</th>
		<th>Description</th>
	</td>
	<tr>
		<td>main_progressive_octree.cpp</td>
		<td>C++ entry point.</td>
	</tr>
	<tr>
		<td>render.cu</td>
		<td>CUDA kernel that renders the scene. </td>
	</tr>
	<tr>
		<td>progressive_octree_voxels.cu</td>
		<td>CUDA kernel that incrementally updates the octree.</td>
	</tr>
</table>


## Bibtex

<pre>
@article{SimLOD,
    title =      "SimLOD: Simultaneous LOD Generation and Rendering",
    author =     "Markus Schütz and Lukas Herzberger and Michael Wimmer",
    year =       "2023",
    month =      oct,
    journal =    "Arxiv",
    keywords =   "point-based rendering",
}
</pre>

## References

Most relevant related work

* 