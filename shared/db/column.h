#ifndef DB_COLUMN_H
#define DB_COLUMN_H

#include <string>
#include "value.h"

namespace db {

struct Column {
    std::string name;
    ColType     type;
    uint16_t    size;     // STR/BLOB: max bytes (fixed-width on disk); INT64=8; DOUBLE=8
    bool        nullable;

    Column() : type(ColType::INT64), size(8), nullable(true) {}
    Column(std::string n, ColType t, uint16_t sz = 0, bool null = true)
        : name(std::move(n)), type(t), nullable(null)
    {
        if (sz == 0) {
            size = (t == ColType::INT64 || t == ColType::DOUBLE) ? 8 : 32;
        } else {
            size = sz;
        }
    }

    // Bytes occupied on disk for this column (fixed-width)
    size_t storageBytes() const {
        switch (type) {
            case ColType::INT64:  return 8;
            case ColType::DOUBLE: return 8;
            case ColType::STR:    return size;
            case ColType::BLOB:   return size;
        }
        return 0;
    }
};

} // namespace db

#endif
