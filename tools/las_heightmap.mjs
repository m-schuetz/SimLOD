import {promises as fsp} from "fs";
import * as path from "path";
import * as fs from "fs";

function clamp(value, min, max){
	if(value < min) return min;
	if(value > max) return max;

	return value;
}

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


async function processLas(file, outdir){
	let header = await readHeader(file);

	console.log(`start processing ${file}`);

	let numPointsProcessed = 0;
	let handle = await fsp.open(file);
	
	let rgbOffset = 0;
	if(header.format === 2) rgbOffset = 20;
	if(header.format === 3) rgbOffset = 28;
	if(header.format === 5) rgbOffset = 28;
	if(header.format === 7) rgbOffset = 30;

	let boxSize = {
		x: header.max.x - header.min.x,
		y: header.max.y - header.min.y,
		z: header.max.z - header.min.z,
	};

	let pixelSize = 10; // meters
	let heightmapSize = {
		x: Math.ceil(boxSize.x / pixelSize),
		y: Math.ceil(boxSize.y / pixelSize),
	};

	let maxColorValue = 65535;
	let ppmHeader = `P6\n`;
	ppmHeader += `${heightmapSize.x} ${heightmapSize.y}\n`;
	ppmHeader += `${maxColorValue}\n`;

	let headerByteSize = ppmHeader.length;
	let contentByteSize = 6 * heightmapSize.x * heightmapSize.y;

	let ppmbuffer = Buffer.alloc(headerByteSize + contentByteSize);
	let floatBuffer = Buffer.alloc(4 * heightmapSize.x * heightmapSize.y);

	for(let i = 0; i < headerByteSize; i++){
		ppmbuffer.writeUint8(ppmHeader.charCodeAt(i), i);
	}

	// clear buffer (probably cleared, though?)
	for(let y = 0; y < heightmapSize.y; y++)
	for(let x = 0; x < heightmapSize.x; x++)
	{
		let pixelIndex = x + heightmapSize.x * y;

		ppmbuffer.writeUint16LE(  0, headerByteSize + 6 * pixelIndex + 0);
		ppmbuffer.writeUint16LE(  0, headerByteSize + 6 * pixelIndex + 2);
		ppmbuffer.writeUint16LE(  0, headerByteSize + 6 * pixelIndex + 4);
		floatBuffer.writeFloatLE(-Infinity, 4 * pixelIndex);
	};

	let MAX_BATCH_SIZE = 100_000;
	let buffer = Buffer.alloc(header.bytesPerPoint * MAX_BATCH_SIZE);
	
	// iterate through batches of MAX_BATCH_SIZE points
	while(numPointsProcessed < header.numPoints){

		let byteOffset = header.offsetToPointData + numPointsProcessed * header.bytesPerPoint;
		let pointsLeft = header.numPoints - numPointsProcessed;
		let batchSize = Math.min(pointsLeft, MAX_BATCH_SIZE);

		await handle.read(buffer, 0, batchSize * header.bytesPerPoint, byteOffset);

		// iterate through points in a batch
		for(let i = 0; i < batchSize; i++){
			let X = buffer.readInt32LE(i * header.bytesPerPoint + 0);
			let Y = buffer.readInt32LE(i * header.bytesPerPoint + 4);
			let Z = buffer.readInt32LE(i * header.bytesPerPoint + 8);

			let x = X * header.scale.x + header.offset.x;
			let y = Y * header.scale.y + header.offset.y;
			let z = Z * header.scale.z + header.offset.z;

			let R = buffer.readUint16LE(i * header.bytesPerPoint + rgbOffset + 0);
			let G = buffer.readUint16LE(i * header.bytesPerPoint + rgbOffset + 2);
			let B = buffer.readUint16LE(i * header.bytesPerPoint + rgbOffset + 4);
			// let r = Math.floor(R > 255 ? R / 256 : R);
			// let g = Math.floor(G > 255 ? G / 256 : G);
			// let b = Math.floor(B > 255 ? B / 256 : B);

			// transform from global cooridinates to heightmap pixel coordinates
			let ix = clamp(Math.floor((x - header.min.x) / pixelSize), 0, heightmapSize.x - 1);
			let iy = clamp(Math.floor((y - header.min.y) / pixelSize), 0, heightmapSize.y - 1);

			let pixelIndex = ix + heightmapSize.x * iy;

			if(pixelIndex >= heightmapSize.x * heightmapSize.y){
				debugger;
			}

			// compute max height per pixel
			let oldHeight = ppmbuffer.readFloatLE(headerByteSize + 6 * pixelIndex + 0);
			let newHeight = Math.max(oldHeight, z);

			// store height as float32 value
			ppmbuffer.writeFloatLE(newHeight, headerByteSize + 6 * pixelIndex + 0);

			// store height as 16bit integer
			let intHeight = Math.round(newHeight)
			intHeight = Math.max(intHeight, 0);    // clamp to (0, Inf)
			ppmbuffer.writeUint16LE(intHeight, headerByteSize + 6 * pixelIndex + 4);

			// store height as float32 in dedicated float buffer
			floatBuffer.writeFloatLE(newHeight, 4 * pixelIndex);
			
		}

		numPointsProcessed += MAX_BATCH_SIZE;
	}

	let filename = path.basename(file);
	fs.writeFileSync(`${outdir}./${filename}.ppm`, ppmbuffer);
	fs.writeFileSync(`${outdir}./${filename}.heightmap`, floatBuffer);

	handle.close();

	console.log(`finished ${filename}`);

}


// PROCESS SPECIFIC FILES
// let files = [
// 	"E:/resources/pointclouds/CA13_las/ot_35120E8124C_1_1.las",
// 	// "E:/resources/pointclouds/CA13_las/ot_35120A4202A_1_1.las",
// 	// "E:/resources/pointclouds/CA13_las/ot_35120A4202B_1_1.las",
// 	// "E:/resources/pointclouds/CA13_las/ot_35120A4202C_1_1.las",
// ];

// OR LOAD FROM DIRECTORY
let sourceDir = "E:/resources/pointclouds/CA13_las";
let files = fs.readdirSync(sourceDir);
files = files.map(filename => `${sourceDir}/${filename}`);
// files.slice(1000, 2000);

let outdir = "./heightmaps/";
if(!fs.existsSync(outdir)){
	fs.mkdirSync(outdir);
}

for(let filename of files){
	let filepath = `${filename}`;
	await processLas(filepath, outdir);
}
