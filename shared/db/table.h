#ifndef DB_TABLE_H
#define DB_TABLE_H

#include "schema.h"
#include "row.h"
#include "index.h"
#include <string>
#include <vector>
#include <memory>

namespace db {

// Table = Schema + Rows + Indexes. Cấu trúc trung tâm của mini-DBMS.
//
// RowId: int64 — position trong rows_, stable trong session (tombstone preserves slot).
// Sau saveToFile + loadFromFile RowId có thể đổi (compact).
class Table {
public:
    Table(std::string name, Schema schema);
    ~Table() = default;

    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    const std::string& name()      const { return name_; }
    const Schema&      schema()    const { return schema_; }
    size_t             size()      const { return aliveCount_; }
    size_t             capacity()  const { return rows_.size(); }

    // -- Index DDL --
    void addIndex(const std::string& column, IndexKind kind, bool unique = false);

    // -- CRUD --
    RowId insert(Row row);                                    // throws UNIQUE violation
    bool  remove(RowId rid);
    bool  update(RowId rid, const std::string& column, Value v);
    void  clear();                                            // truncate all rows + indexes

    // -- Lookups --
    bool         alive(RowId rid) const;
    const Row*   getRow(RowId rid) const;
    Row*         getRow(RowId rid);

    // Tra cứu theo cột có index. Nếu không có index → throw (force người dùng nghĩ).
    RowId               findOne(const std::string& column, const Value& key) const;
    std::vector<RowId>  find   (const std::string& column, const Value& key) const;
    std::vector<RowId>  range  (const std::string& column, const Value& lo, const Value& hi) const;

    // Linear scan — không dùng index
    std::vector<RowId>  scanAll() const;

    // -- Persistence --
    bool saveToFile(const std::string& path) const;
    bool loadFromFile(const std::string& path);  // schema must already match

    // Diagnostics
    bool hasIndex(const std::string& column) const;

private:
    std::string                       name_;
    Schema                            schema_;
    std::vector<Row>                  rows_;
    std::vector<uint8_t>              alive_;     // 0 = tombstone, 1 = alive
    size_t                            aliveCount_;

    struct IndexEntry {
        std::string             column;
        IndexKind               kind;
        bool                    unique;
        std::unique_ptr<Index>  idx;
    };
    std::vector<IndexEntry>           indexes_;

    Index*       findIndex(const std::string& column);
    const Index* findIndex(const std::string& column) const;
    void         rebuildAllIndexes();

    void         writeRowBin(class BinaryWriter& w, const Row& r) const;
    Row          readRowBin (class BinaryReader& r) const;
};

} // namespace db

#endif
