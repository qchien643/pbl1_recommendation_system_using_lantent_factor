#ifndef DB_INDEX_H
#define DB_INDEX_H

#include "value.h"
#include <vector>
#include <cstdint>

namespace db {

using RowId = int64_t;  // Position trong Table::rows_; >=0 hợp lệ, <0 invalid

enum class IndexKind : uint8_t {
    HASH  = 1,  // O(1) equality
    BTREE = 2,  // O(log N) ordered + range
};

// Abstract base — subclasses: HashIndex, BTreeIndex
class Index {
public:
    virtual ~Index() = default;

    virtual IndexKind kind() const = 0;
    virtual bool      unique() const = 0;

    virtual void      insert(const Value& key, RowId rid) = 0;
    virtual void      erase (const Value& key, RowId rid) = 0;
    virtual void      clear() = 0;

    // Equality lookup. Returns all RowId matching key (>=1 nếu non-unique).
    virtual std::vector<RowId> find(const Value& key) const = 0;

    // Range [lo..hi] inclusive. BTree-only — Hash trả về rỗng + warning.
    virtual std::vector<RowId> range(const Value& lo, const Value& hi) const = 0;

    virtual size_t    size() const = 0;
};

} // namespace db

#endif
