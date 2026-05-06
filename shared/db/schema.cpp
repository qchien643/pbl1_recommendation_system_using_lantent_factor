#include "schema.h"
#include <stdexcept>

namespace db {

Schema& Schema::add(const std::string& name, ColType type, uint16_t size, bool nullable) {
    if (nameToIdx_.count(name))
        throw std::runtime_error("Schema::add — duplicate column: " + name);
    Column c(name, type, size, nullable);
    nameToIdx_[name] = (int)columns_.size();
    columns_.push_back(std::move(c));
    return *this;
}

Schema& Schema::setPrimaryKey(const std::string& name) {
    if (!nameToIdx_.count(name))
        throw std::runtime_error("Schema::setPrimaryKey — unknown column: " + name);
    pk_ = name;
    return *this;
}

int Schema::indexOf(const std::string& name) const {
    auto it = nameToIdx_.find(name);
    return it == nameToIdx_.end() ? -1 : it->second;
}

const Column& Schema::at(int idx) const {
    if (idx < 0 || idx >= (int)columns_.size())
        throw std::out_of_range("Schema::at idx");
    return columns_[idx];
}

const Column& Schema::at(const std::string& name) const {
    int i = indexOf(name);
    if (i < 0) throw std::out_of_range("Schema::at " + name);
    return columns_[i];
}

size_t Schema::rowSize() const {
    size_t s = 0;
    for (const auto& c : columns_) s += c.storageBytes();
    return s;
}

// CRC32 (IEEE polynomial 0xEDB88320). Stable hash dùng cho migration check.
static uint32_t crc32_buf(const uint8_t* data, size_t n, uint32_t seed = 0xFFFFFFFF) {
    static uint32_t table[256];
    static bool inited = false;
    if (!inited) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c >> 1) ^ ((c & 1) ? 0xEDB88320 : 0);
            table[i] = c;
        }
        inited = true;
    }
    uint32_t c = seed;
    for (size_t i = 0; i < n; i++)
        c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c;
}

uint32_t Schema::hash() const {
    uint32_t c = 0xFFFFFFFF;
    for (const auto& col : columns_) {
        c = crc32_buf((const uint8_t*)col.name.data(), col.name.size(), c);
        uint8_t t = (uint8_t)col.type;
        c = crc32_buf(&t, 1, c);
        uint8_t sz[2] = { (uint8_t)(col.size & 0xFF), (uint8_t)(col.size >> 8) };
        c = crc32_buf(sz, 2, c);
    }
    return c ^ 0xFFFFFFFF;
}

} // namespace db
