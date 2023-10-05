// converts LAS point cloud files to XYZRGBA binary files.
// - Smallest las format with RGB data requires 26 bytes per point
// - The bin format stores points as 3xfloat and 4xuint8 => 16 bytes per point
// - To improve float precision, the point cloud is moved towards the origin. Coordinates start at 0.0
// - A small header stores the bounding box as min_x, min_y, min_z, max_x, max_y, max_z, for a total of 6 * 4 = 24 bytes

// <header, 24 bytes>                        <points, 16 byte each>
// [min_x, min_y, min_z, max_x, max_y, max_z][XYZRGBA, XYZRGBA, XYZRGBA, ...]

import {promises as fsp} from "fs";

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


async function main(file, outPath){
	const MAX_BATCH_SIZE = 1_000_000;

	let header = await readHeader(file);

	console.log(header);

	let lines = [];
	let outBufferHeader = Buffer.alloc(24);
	let outBufferBatch = Buffer.alloc(16 * MAX_BATCH_SIZE);

	let numPointsProcessed = 0;
	let handle = await fsp.open(file);
	let buffer = Buffer.alloc(MAX_BATCH_SIZE * header.bytesPerPoint);
	let rgbOffset = 0;
	if(header.format === 2) rgbOffset = 20;
	if(header.format === 3) rgbOffset = 28;
	if(header.format === 5) rgbOffset = 28;
	if(header.format === 7) rgbOffset = 30;

	// header storing min and max bounding box in 6 * 4 = 24 bytes
	outBufferHeader.writeFloatLE(0.0,  0);
	outBufferHeader.writeFloatLE(0.0,  4);
	outBufferHeader.writeFloatLE(0.0,  8);
	outBufferHeader.writeFloatLE(header.max.x - header.min.x, 12);
	outBufferHeader.writeFloatLE(header.max.y - header.min.y, 16);
	outBufferHeader.writeFloatLE(header.max.z - header.min.z, 20);

	try{
		await fsp.unlink(outPath);
	}catch(e){}

	await fsp.appendFile(outPath, outBufferHeader);

	while(numPointsProcessed < header.numPoints){

		let pointsLeft = header.numPoints - numPointsProcessed;
		let batchSize = Math.min(pointsLeft, MAX_BATCH_SIZE);

		let byteOffset = header.offsetToPointData + numPointsProcessed * header.bytesPerPoint;
		let byteSize = batchSize * header.bytesPerPoint;
		let outByteSize = batchSize * 16

		if(outBufferBatch.byteLength !== outByteSize){
			outBufferBatch = Buffer.alloc(outByteSize);
		}

		await handle.read(buffer, 0, byteSize, byteOffset);

		for(let i = 0; i < batchSize; i++){

			let X = buffer.readInt32LE(i * header.bytesPerPoint + 0);
			let Y = buffer.readInt32LE(i * header.bytesPerPoint + 4);
			let Z = buffer.readInt32LE(i * header.bytesPerPoint + 8);

			let x = X * header.scale.x + header.offset.x - header.min.x;
			let y = Y * header.scale.y + header.offset.y - header.min.y;
			let z = Z * header.scale.z + header.offset.z - header.min.z;

			let R = buffer.readUint16LE(i * header.bytesPerPoint + rgbOffset + 0);
			let G = buffer.readUint16LE(i * header.bytesPerPoint + rgbOffset + 2);
			let B = buffer.readUint16LE(i * header.bytesPerPoint + rgbOffset + 4);
			let r = Math.floor(R > 255 ? R / 256 : R);
			let g = Math.floor(G > 255 ? G / 256 : G);
			let b = Math.floor(B > 255 ? B / 256 : B);

			outBufferBatch.writeFloatLE(x, 16 * i + 0);
			outBufferBatch.writeFloatLE(y, 16 * i + 4);
			outBufferBatch.writeFloatLE(z, 16 * i + 8);
			outBufferBatch.writeUint8(r, 16 * i + 12);
			outBufferBatch.writeUint8(g, 16 * i + 13);
			outBufferBatch.writeUint8(b, 16 * i + 14);
			outBufferBatch.writeUint8(255, 16 * i + 15);

			numPointsProcessed++;
		}

		await fsp.appendFile(outPath, outBufferBatch);

		console.log(`points processed: ${numPointsProcessed.toLocaleString()}`);
	}


	// let csv = lines.join("\n");

	// fsp.writeFile("E:/resources/pointclouds/simlod/test.csv", csv);
	// fsp.writeFile(outPath, outBuffer);
	// await fsp.appendFile(outPath, outBuffer);

}

// let file = "E:/resources/pointclouds/simlod/morro_bay.las";
// let outPath = "E:/resources/pointclouds/simlod/morro_bay.simlod";

// let file = "E:/resources/pointclouds/simlod/meroe.las";
// let outPath = "E:/resources/pointclouds/simlod/meroe.simlod";

let file = "E:/resources/pointclouds/simlod/endeavor_p2.las";
let outPath = "E:/resources/pointclouds/simlod/endeavor_p2.simlod";

main(file, outPath);