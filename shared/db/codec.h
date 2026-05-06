#ifndef DB_CODEC_H
#define DB_CODEC_H

#include <cstdint>
#include <iosfwd>
#include <string>

namespace db {

// Binary writer/reader little-endian, fixed-width.
class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& out) : out_(out) {}
    void writeU32(uint32_t v);
    void writeI32(int32_t v);
    void writeI64(int64_t v);
    void writeF64(double v);
    // Pad/truncate to exactly maxBytes; bytes after string null-padded.
    void writeFixedStr(const std::string& s, size_t maxBytes);
    void writeBytes(const void* data, size_t n);
    bool ok() const;
private:
    std::ostream& out_;
};

class BinaryReader {
public:
    explicit BinaryReader(std::istream& in) : in_(in) {}
    uint32_t readU32();
    int32_t  readI32();
    int64_t  readI64();
    double   readF64();
    // Reads exactly maxBytes; returns string trimmed at first NUL.
    std::string readFixedStr(size_t maxBytes);
    void     readBytes(void* out, size_t n);
    bool     ok() const;
private:
    std::istream& in_;
};

} // namespace db

#endif
