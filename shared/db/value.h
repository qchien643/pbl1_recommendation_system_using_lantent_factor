#ifndef DB_VALUE_H
#define DB_VALUE_H

#include <cstdint>
#include <string>
#include <variant>
#include <iosfwd>

namespace db {

enum class ColType : uint8_t {
    INT64  = 1,
    DOUBLE = 2,
    STR    = 3,
    BLOB   = 4
};

class Value {
public:
    Value();                                  // null (INT64, 0)
    Value(int64_t v);                         // implicit ok
    Value(int v);
    Value(double v);
    Value(const std::string& s);
    Value(std::string&& s);
    Value(const char* s);

    // BLOB (raw bytes, may contain embedded NUL)
    static Value makeBlob(std::string bytes);

    ColType type() const { return type_; }
    bool    isBlob() const { return type_ == ColType::BLOB; }

    int64_t              asInt()    const;
    double               asDouble() const;
    const std::string&   asString() const;

    bool operator==(const Value& o) const;
    bool operator!=(const Value& o) const { return !(*this == o); }
    bool operator< (const Value& o) const;
    bool operator<=(const Value& o) const { return !(o < *this); }
    bool operator> (const Value& o) const { return o < *this; }
    bool operator>=(const Value& o) const { return !(*this < o); }

    // Hash for HashIndex (FNV-1a 64-bit)
    uint64_t hash() const;

    std::string debugString() const;

private:
    ColType type_;
    std::variant<int64_t, double, std::string> data_;
};

std::ostream& operator<<(std::ostream& os, const Value& v);

} // namespace db

#endif
