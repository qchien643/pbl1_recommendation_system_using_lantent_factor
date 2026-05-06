#ifndef DB_SCHEMA_H
#define DB_SCHEMA_H

#include <string>
#include <vector>
#include <unordered_map>
#include "column.h"

namespace db {

class Schema {
public:
    Schema() = default;

    Schema& add(const std::string& name, ColType type, uint16_t size = 0, bool nullable = true);
    Schema& setPrimaryKey(const std::string& name);

    int                  indexOf(const std::string& name) const;  // -1 if not found
    const Column&        at(int idx) const;
    const Column&        at(const std::string& name) const;
    int                  columnCount() const { return (int)columns_.size(); }
    const std::vector<Column>& columns() const { return columns_; }

    const std::string&   primaryKey() const { return pk_; }

    // Sum of column.storageBytes() — used for fixed-width row layout
    size_t rowSize() const;

    // CRC32 stable hash of (column name + type + size) sequence — for migration check
    uint32_t hash() const;

private:
    std::vector<Column> columns_;
    std::unordered_map<std::string, int> nameToIdx_;
    std::string pk_;
};

} // namespace db

#endif
