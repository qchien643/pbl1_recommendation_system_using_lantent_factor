#include "value.h"
#include <ostream>
#include <stdexcept>
#include <cstring>

namespace db {

Value::Value()                          : type_(ColType::INT64),  data_((int64_t)0) {}
Value::Value(int64_t v)                 : type_(ColType::INT64),  data_(v) {}
Value::Value(int v)                     : type_(ColType::INT64),  data_((int64_t)v) {}
Value::Value(double v)                  : type_(ColType::DOUBLE), data_(v) {}
Value::Value(const std::string& s)      : type_(ColType::STR),    data_(s) {}
Value::Value(std::string&& s)           : type_(ColType::STR),    data_(std::move(s)) {}
Value::Value(const char* s)             : type_(ColType::STR),    data_(std::string(s ? s : "")) {}

Value Value::makeBlob(std::string bytes) {
    Value v;
    v.type_ = ColType::BLOB;
    v.data_ = std::move(bytes);
    return v;
}

int64_t Value::asInt() const {
    if (type_ == ColType::INT64)  return std::get<int64_t>(data_);
    if (type_ == ColType::DOUBLE) return (int64_t)std::get<double>(data_);
    throw std::runtime_error("Value::asInt — not numeric");
}

double Value::asDouble() const {
    if (type_ == ColType::DOUBLE) return std::get<double>(data_);
    if (type_ == ColType::INT64)  return (double)std::get<int64_t>(data_);
    throw std::runtime_error("Value::asDouble — not numeric");
}

const std::string& Value::asString() const {
    if (type_ == ColType::STR || type_ == ColType::BLOB)
        return std::get<std::string>(data_);
    throw std::runtime_error("Value::asString — not string");
}

bool Value::operator==(const Value& o) const {
    if (type_ != o.type_) {
        // Numeric coercion
        if ((type_ == ColType::INT64 || type_ == ColType::DOUBLE) &&
            (o.type_ == ColType::INT64 || o.type_ == ColType::DOUBLE))
            return asDouble() == o.asDouble();
        return false;
    }
    return data_ == o.data_;
}

bool Value::operator<(const Value& o) const {
    if (type_ == ColType::INT64 && o.type_ == ColType::INT64)
        return std::get<int64_t>(data_) < std::get<int64_t>(o.data_);
    if (type_ == ColType::DOUBLE && o.type_ == ColType::DOUBLE)
        return std::get<double>(data_) < std::get<double>(o.data_);
    if ((type_ == ColType::INT64 || type_ == ColType::DOUBLE) &&
        (o.type_ == ColType::INT64 || o.type_ == ColType::DOUBLE))
        return asDouble() < o.asDouble();
    if ((type_ == ColType::STR || type_ == ColType::BLOB) &&
        (o.type_ == ColType::STR || o.type_ == ColType::BLOB))
        return std::get<std::string>(data_) < std::get<std::string>(o.data_);
    return (uint8_t)type_ < (uint8_t)o.type_;
}

uint64_t Value::hash() const {
    // FNV-1a 64-bit
    constexpr uint64_t FNV_OFFSET = 1469598103934665603ULL;
    constexpr uint64_t FNV_PRIME  = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;

    auto mix = [&](const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= FNV_PRIME; }
    };

    if (type_ == ColType::INT64) {
        int64_t v = std::get<int64_t>(data_);
        mix((const uint8_t*)&v, sizeof(v));
    } else if (type_ == ColType::DOUBLE) {
        double v = std::get<double>(data_);
        mix((const uint8_t*)&v, sizeof(v));
    } else {
        const std::string& s = std::get<std::string>(data_);
        mix((const uint8_t*)s.data(), s.size());
    }
    return h;
}

std::string Value::debugString() const {
    switch (type_) {
        case ColType::INT64:  return std::to_string(std::get<int64_t>(data_));
        case ColType::DOUBLE: return std::to_string(std::get<double>(data_));
        case ColType::STR:    return "\"" + std::get<std::string>(data_) + "\"";
        case ColType::BLOB:   return "<blob:" + std::to_string(std::get<std::string>(data_).size()) + ">";
    }
    return "?";
}

std::ostream& operator<<(std::ostream& os, const Value& v) {
    return os << v.debugString();
}

} // namespace db
