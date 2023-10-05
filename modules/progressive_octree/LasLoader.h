#pragma once

#include <cstdint>

#include "unsuck.hpp"

struct LasHeader{
	int versionMajor             = 0;
	int versionMinor             = 0;
	uint64_t headerSize          = 0;
	uint64_t offsetToPointData   = 0;
	uint64_t format              = 0;
	uint64_t bytesPerPoint       = 0;
	uint64_t numPoints           = 0;
	double scale[3]              = {0.0, 0.0, 0.0};
	double offset[3]             = {0.0, 0.0, 0.0};
	double min[3]                = {0.0, 0.0, 0.0};
	double max[3]                = {0.0, 0.0, 0.0};
};

inline LasHeader loadHeader(string file){
	auto headerBuffer = readBinaryFile(file, 0, 375);

	LasHeader header;
	header.versionMajor        = headerBuffer->get<uint8_t>(24);
	header.versionMinor        = headerBuffer->get<uint8_t>(25);
	header.headerSize          = headerBuffer->get<uint16_t>(94);
	header.offsetToPointData   = headerBuffer->get<uint32_t>(96);
	header.format              = headerBuffer->get<uint8_t>(104);
	header.bytesPerPoint       = headerBuffer->get<uint16_t>(105);
	if(header.versionMajor == 1 && header.versionMinor <= 3){
		header.numPoints       = headerBuffer->get<uint32_t>(107);
	}else{
		header.numPoints       = headerBuffer->get<uint64_t>(247);
	}

	
	header.scale[0] = headerBuffer->get<double>(131);
	header.scale[1] = headerBuffer->get<double>(139);
	header.scale[2] = headerBuffer->get<double>(147);

	header.offset[0] = headerBuffer->get<double>(155);
	header.offset[1] = headerBuffer->get<double>(163);
	header.offset[2] = headerBuffer->get<double>(171);

	header.min[0] = headerBuffer->get<double>(187);
	header.min[1] = headerBuffer->get<double>(203);
	header.min[2] = headerBuffer->get<double>(219);

	header.max[0] = headerBuffer->get<double>(179);
	header.max[1] = headerBuffer->get<double>(195);
	header.max[2] = headerBuffer->get<double>(211);

	return header;
}

void loadLasNative(string file, LasHeader header, uint64_t firstPoint, uint64_t numPoints, void* target, double translation[3]);