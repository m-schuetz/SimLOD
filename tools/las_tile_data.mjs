// converts LAS point cloud files to XYZRGBA binary files.
// - Smallest las format with RGB data requires 26 bytes per point
// - The bin format stores points as 3xfloat and 4xuint8 => 16 bytes per point
// - To improve float precision, the point cloud is moved towards the origin. Coordinates start at 0.0
// - A small header stores the bounding box as min_x, min_y, min_z, max_x, max_y, max_z, for a total of 6 * 4 = 24 bytes

// <header, 24 bytes>                        <points, 16 byte each>
// [min_x, min_y, min_z, max_x, max_y, max_z][XYZRGBA, XYZRGBA, XYZRGBA, ...]

import {promises as fsp} from "fs";
import * as path from "path";
import * as fs from "fs";

class LasHeader{

	constructor(){
		this.versionMajor      = 0;
		this.versionMinor      = 0;
		this.headerSize        = 0;
		this.offsetToPointData = 0;
		this.format            = 0;
		this.bytesPerPoint     = 0;
		this.numPoints         = 0;
		this.scale             = {x: 0, y: 0, z: 0};
		this.offset            = {x: 0, y: 0, z: 0};
		this.min               = {x: 0, y: 0, z: 0};
		this.max               = {x: 0, y: 0, z: 0};
	}

};

async function readHeader(file){

	let handle = await fsp.open(file);
	let buffer = Buffer.alloc(375);
	await handle.read(buffer, 0, 375);

	let header = new LasHeader();
	header.versionMajor      = buffer.readUint8(24);
	header.versionMinor      = buffer.readUint8(25);
	header.headerSize        = buffer.readUint16LE(94);
	header.offsetToPointData = buffer.readUint32LE(96);
	header.format            = buffer.readUint8(104);
	header.bytesPerPoint     = buffer.readUint16LE(105);
	if(header.versionMajor === 1 && header.versionMinor <= 2){
		header.numPoints     = buffer.readUint32LE(107); 
	}else{
		header.numPoints     = buffer.readUint64LE(247); 
	}
	header.scale             = {
		x: buffer.readDoubleLE(131),
		y: buffer.readDoubleLE(139),
		z: buffer.readDoubleLE(147),
	};
	header.offset            = {
		x: buffer.readDoubleLE(155),
		y: buffer.readDoubleLE(163),
		z: buffer.readDoubleLE(171),
	};
	header.min               = {
		x: buffer.readDoubleLE(187),
		y: buffer.readDoubleLE(203),
		z: buffer.readDoubleLE(219),
	};
	header.max               = {
		x: buffer.readDoubleLE(179),
		y: buffer.readDoubleLE(195),
		z: buffer.readDoubleLE(211),
	};

	handle.close();

	return header;
};


async function processLas(file, outPath, outCSV){
	let header = await readHeader(file);

	// console.log(header);

	let numPointsProcessed = 0;
	let handle = await fsp.open(file);
	let buffer = Buffer.alloc(header.bytesPerPoint);
	let rgbOffset = 0;
	if(header.format === 2) rgbOffset = 20;
	if(header.format === 3) rgbOffset = 28;
	if(header.format === 5) rgbOffset = 28;
	if(header.format === 7) rgbOffset = 30;

	let filename = path.basename(file);
	
	let strBoxMin = `${header.min.x.toFixed(2)}, ${header.min.y.toFixed(2)}, ${header.min.z.toFixed(2)}, `;
	let strBoxMax = `${header.max.x.toFixed(2)}, ${header.max.y.toFixed(2)}, ${header.max.z.toFixed(2)}, `;

	await fsp.appendFile(outPath, `${filename}, `);
	await fsp.appendFile(outPath, strBoxMin);
	await fsp.appendFile(outPath, strBoxMax);
	

	while(numPointsProcessed < header.numPoints){

		let byteOffset = header.offsetToPointData + numPointsProcessed * header.bytesPerPoint;

		await handle.read(buffer, 0, header.bytesPerPoint, byteOffset);

		let X = buffer.readInt32LE(0);
		let Y = buffer.readInt32LE(4);
		let Z = buffer.readInt32LE(8);

		let x = X * header.scale.x + header.offset.x;
		let y = Y * header.scale.y + header.offset.y;
		let z = Z * header.scale.z + header.offset.z;

		let R = buffer.readUint16LE(rgbOffset + 0);
		let G = buffer.readUint16LE(rgbOffset + 2);
		let B = buffer.readUint16LE(rgbOffset + 4);
		let r = Math.floor(R > 255 ? R / 256 : R);
		let g = Math.floor(G > 255 ? G / 256 : G);
		let b = Math.floor(B > 255 ? B / 256 : B);

		let line = `${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}, `;

		await fsp.appendFile(outPath, line);
		await fsp.appendFile(outCSV, `${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}, ${r}, ${g}, ${b} \n`);

		numPointsProcessed += 50_000;
	}

	await fsp.appendFile(outPath, "\n");

	// fsp.close(outPath);
	handle.close();


	// let csv = lines.join("\n");

	// fsp.writeFile("E:/resources/pointclouds/simlod/test.csv", csv);
	// fsp.writeFile(outPath, outBuffer);
	// await fsp.appendFile(outPath, outBuffer);

}

// let file = "E:/resources/pointclouds/simlod/morro_bay.las";
// let outPath = "E:/resources/pointclouds/simlod/morro_bay.simlod";

// let file = "E:/resources/pointclouds/simlod/meroe.las";
// let outPath = "E:/resources/pointclouds/simlod/meroe.simlod";

// let file = "./test.las";
// let file = "E:/resources/pointclouds/CA13_las/ot_35120A4201B_1_1.las";
let outPath = "./report.txt";
let outCSV = "./chunkPoints_2.csv";

// let files = [
// 	"E:/resources/pointclouds/CA13_las/ot_35120A4201B_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120A4202A_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120A4202B_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120A4202C_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120A4202D_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120A4203A_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120B4116A_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120B4116B_1_1.las",
// 	"E:/resources/pointclouds/CA13_las/ot_35120B4116C_1_1.las",
// ];

let files = fs.readdirSync("E:/resources/pointclouds/CA13_las");

// console.log(files);

try{
	await fsp.unlink(outPath);
	await fsp.unlink(outCSV);
}catch(e){}

await fsp.appendFile(outPath, "filename, min_x, min_y, min_z, max_x, max_y, max_z, x, y, z, x, y, z, ... \n");
for(let filename of files){

	let filepath = `E:/resources/pointclouds/CA13_las/${filename}`;
	await processLas(filepath, outPath, outCSV);
}
