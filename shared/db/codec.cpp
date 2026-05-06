#include "codec.h"
#include <ostream>
#include <istream>
#include <cstring>

namespace db {

static void putLE(uint8_t* p, uint64_t v, int n) {
    for (int i = 0; i < n; i++) p[i] = (uint8_t)((v >> (8 * i)) & 0xFF);
}
static uint64_t getLE(const uint8_t* p, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; i++) v |= ((uint64_t)p[i]) << (8 * i);
    return v;
}

// ---- Writer ----
void BinaryWriter::writeU32(uint32_t v) {
    uint8_t b[4]; putLE(b, v, 4);
    out_.write((const char*)b, 4);
}
void BinaryWriter::writeI32(int32_t v) { writeU32((uint32_t)v); }
void BinaryWriter::writeI64(int64_t v) {
    uint8_t b[8]; putLE(b, (uint64_t)v, 8);
    out_.write((const char*)b, 8);
}
void BinaryWriter::writeF64(double v) {
    uint64_t u; std::memcpy(&u, &v, 8);
    uint8_t b[8]; putLE(b, u, 8);
    out_.write((const char*)b, 8);
}
void BinaryWriter::writeFixedStr(const std::string& s, size_t maxBytes) {
    size_t n = s.size() < maxBytes ? s.size() : maxBytes;
    if (n) out_.write(s.data(), (std::streamsize)n);
    if (maxBytes > n) {
        static const char zeros[64] = {0};
        size_t pad = maxBytes - n;
        while (pad >= sizeof(zeros)) { out_.write(zeros, sizeof(zeros)); pad -= sizeof(zeros); }
        if (pad) out_.write(zeros, (std::streamsize)pad);
    }
}
void BinaryWriter::writeBytes(const void* data, size_t n) {
    out_.write((const char*)data, (std::streamsize)n);
}
bool BinaryWriter::ok() const { return (bool)out_; }

// ---- Reader ----
uint32_t BinaryReader::readU32() {
    uint8_t b[4]; in_.read((char*)b, 4);
    return (uint32_t)getLE(b, 4);
}
int32_t BinaryReader::readI32() { return (int32_t)readU32(); }
int64_t BinaryReader::readI64() {
    uint8_t b[8]; in_.read((char*)b, 8);
    return (int64_t)getLE(b, 8);
}
double BinaryReader::readF64() {
    uint8_t b[8]; in_.read((char*)b, 8);
    uint64_t u = getLE(b, 8);
    double v; std::memcpy(&v, &u, 8);
    return v;
}
std::string BinaryReader::readFixedStr(size_t maxBytes) {
    std::string buf(maxBytes, '\0');
    if (maxBytes) in_.read(&buf[0], (std::streamsize)maxBytes);
    size_t real = 0;
    while (real < maxBytes && buf[real] != '\0') real++;
    buf.resize(real);
    return buf;
}
void BinaryReader::readBytes(void* out, size_t n) {
    in_.read((char*)out, (std::streamsize)n);
}
bool BinaryReader::ok() const { return (bool)in_; }

} // namespace db
