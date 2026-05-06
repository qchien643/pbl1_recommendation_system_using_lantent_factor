#include "table.h"
#include "hash_index.h"
#include "btree_index.h"
#include "codec.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace db {

static const char MAGIC[16] = { 'P','B','L','1','D','B','v','1', 0,0,0,0,0,0,0,0 };

Table::Table(std::string name, Schema schema)
    : name_(std::move(name)), schema_(std::move(schema)), aliveCount_(0) {}

// -- Index management --

void Table::addIndex(const std::string& column, IndexKind kind, bool unique) {
    if (schema_.indexOf(column) < 0)
        throw std::runtime_error("Table::addIndex — unknown column: " + column);
    if (findIndex(column))
        throw std::runtime_error("Table::addIndex — duplicate index: " + column);

    IndexEntry e;
    e.column = column;
    e.kind   = kind;
    e.unique = unique;
    if (kind == IndexKind::HASH)       e.idx = std::make_unique<HashIndex>(unique);
    else if (kind == IndexKind::BTREE) e.idx = std::make_unique<BTreeIndex>();
    else throw std::runtime_error("Table::addIndex — unknown kind");

    // Populate from existing rows
    for (size_t rid = 0; rid < rows_.size(); rid++) {
        if (!alive_[rid]) continue;
        e.idx->insert(rows_[rid].at(column), (RowId)rid);
    }
    indexes_.push_back(std::move(e));
}

Index* Table::findIndex(const std::string& column) {
    for (auto& e : indexes_)
        if (e.column == column) return e.idx.get();
    return nullptr;
}
const Index* Table::findIndex(const std::string& column) const {
    for (const auto& e : indexes_)
        if (e.column == column) return e.idx.get();
    return nullptr;
}

bool Table::hasIndex(const std::string& column) const {
    return findIndex(column) != nullptr;
}

void Table::rebuildAllIndexes() {
    for (auto& e : indexes_) {
        e.idx->clear();
        for (size_t rid = 0; rid < rows_.size(); rid++) {
            if (!alive_[rid]) continue;
            e.idx->insert(rows_[rid].at(e.column), (RowId)rid);
        }
    }
}

// -- CRUD --

RowId Table::insert(Row row) {
    if (row.schema() != &schema_) {
        // Allow schema-less Row but values vector must match
        if (row.values().size() != (size_t)schema_.columnCount())
            throw std::runtime_error("Table::insert — schema mismatch");
    }

    // Validate UNIQUE constraints first (so we don't pollute alive_)
    for (auto& e : indexes_) {
        if (e.unique && e.kind == IndexKind::HASH) {
            int colIdx = schema_.indexOf(e.column);
            const Value& v = row.values()[colIdx];
            if (!e.idx->find(v).empty())
                throw std::runtime_error("Table::insert — UNIQUE violation: " + e.column);
        }
    }

    RowId rid = (RowId)rows_.size();
    rows_.push_back(std::move(row));
    alive_.push_back(1);
    ++aliveCount_;

    for (auto& e : indexes_) {
        e.idx->insert(rows_[rid].at(e.column), rid);
    }
    return rid;
}

bool Table::remove(RowId rid) {
    if (rid < 0 || (size_t)rid >= rows_.size() || !alive_[rid]) return false;
    for (auto& e : indexes_) e.idx->erase(rows_[rid].at(e.column), rid);
    alive_[rid] = 0;
    --aliveCount_;
    return true;
}

void Table::clear() {
    rows_.clear();
    alive_.clear();
    aliveCount_ = 0;
    for (auto& e : indexes_) e.idx->clear();
}

bool Table::update(RowId rid, const std::string& column, Value v) {
    if (rid < 0 || (size_t)rid >= rows_.size() || !alive_[rid]) return false;
    int ci = schema_.indexOf(column);
    if (ci < 0) throw std::runtime_error("Table::update — unknown column: " + column);

    Index* ix = findIndex(column);
    if (ix) {
        if (ix->unique() && ix->kind() == IndexKind::HASH) {
            auto hits = ix->find(v);
            for (RowId other : hits) if (other != rid)
                throw std::runtime_error("Table::update — UNIQUE violation: " + column);
        }
        ix->erase(rows_[rid].at(ci), rid);
    }
    rows_[rid].at(ci) = std::move(v);
    if (ix) ix->insert(rows_[rid].at(ci), rid);
    return true;
}

// -- Lookups --

bool Table::alive(RowId rid) const {
    return rid >= 0 && (size_t)rid < alive_.size() && alive_[rid];
}

const Row* Table::getRow(RowId rid) const {
    if (!alive(rid)) return nullptr;
    return &rows_[rid];
}
Row* Table::getRow(RowId rid) {
    if (!alive(rid)) return nullptr;
    return &rows_[rid];
}

RowId Table::findOne(const std::string& column, const Value& key) const {
    const Index* ix = findIndex(column);
    if (!ix) throw std::runtime_error("Table::findOne — no index on: " + column);
    auto v = ix->find(key);
    return v.empty() ? (RowId)-1 : v.front();
}

std::vector<RowId> Table::find(const std::string& column, const Value& key) const {
    const Index* ix = findIndex(column);
    if (!ix) throw std::runtime_error("Table::find — no index on: " + column);
    return ix->find(key);
}

std::vector<RowId> Table::range(const std::string& column, const Value& lo, const Value& hi) const {
    const Index* ix = findIndex(column);
    if (!ix) throw std::runtime_error("Table::range — no index on: " + column);
    if (ix->kind() != IndexKind::BTREE)
        throw std::runtime_error("Table::range — column needs B-tree index: " + column);
    return ix->range(lo, hi);
}

std::vector<RowId> Table::scanAll() const {
    std::vector<RowId> out;
    out.reserve(aliveCount_);
    for (size_t i = 0; i < rows_.size(); i++)
        if (alive_[i]) out.push_back((RowId)i);
    return out;
}

// -- Persistence --

void Table::writeRowBin(BinaryWriter& w, const Row& r) const {
    for (int i = 0; i < schema_.columnCount(); i++) {
        const Column& c = schema_.at(i);
        const Value&  v = r.at(i);
        switch (c.type) {
            case ColType::INT64:  w.writeI64(v.asInt());                       break;
            case ColType::DOUBLE: w.writeF64(v.asDouble());                    break;
            case ColType::STR:    w.writeFixedStr(v.asString(), c.storageBytes()); break;
            case ColType::BLOB: {
                // Raw bytes, no NUL trim. Pad zeros nếu thiếu.
                const std::string& s = v.asString();
                size_t cap = c.storageBytes();
                size_t n = s.size() < cap ? s.size() : cap;
                if (n) w.writeBytes(s.data(), n);
                if (n < cap) {
                    static const char zeros[64] = {0};
                    size_t pad = cap - n;
                    while (pad >= sizeof(zeros)) { w.writeBytes(zeros, sizeof(zeros)); pad -= sizeof(zeros); }
                    if (pad) w.writeBytes(zeros, pad);
                }
                break;
            }
        }
    }
}

Row Table::readRowBin(BinaryReader& r) const {
    Row row(&schema_);
    for (int i = 0; i < schema_.columnCount(); i++) {
        const Column& c = schema_.at(i);
        switch (c.type) {
            case ColType::INT64:  row.set(i, Value(r.readI64()));                        break;
            case ColType::DOUBLE: row.set(i, Value(r.readF64()));                        break;
            case ColType::STR:    row.set(i, Value(r.readFixedStr(c.storageBytes())));   break;
            case ColType::BLOB: {
                std::string buf(c.storageBytes(), '\0');
                r.readBytes(&buf[0], c.storageBytes());
                row.set(i, Value::makeBlob(std::move(buf)));
                break;
            }
        }
    }
    return row;
}

bool Table::saveToFile(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    out.write(MAGIC, sizeof(MAGIC));
    BinaryWriter w(out);
    w.writeU32(schema_.hash());
    w.writeU32((uint32_t)aliveCount_);
    w.writeU32((uint32_t)schema_.rowSize());

    for (size_t rid = 0; rid < rows_.size(); rid++) {
        if (!alive_[rid]) continue;
        writeRowBin(w, rows_[rid]);
    }
    return w.ok();
}

bool Table::loadFromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    char magic[16];
    in.read(magic, sizeof(magic));
    if (in.gcount() != sizeof(magic) || std::memcmp(magic, MAGIC, 8) != 0) return false;

    BinaryReader r(in);
    uint32_t hash    = r.readU32();
    uint32_t nRows   = r.readU32();
    uint32_t rowSize = r.readU32();
    (void)rowSize;

    if (hash != schema_.hash()) {
        // Schema mismatch — không load để tránh corrupt
        return false;
    }

    rows_.clear();
    alive_.clear();
    aliveCount_ = 0;

    rows_.reserve(nRows);
    alive_.reserve(nRows);
    for (uint32_t i = 0; i < nRows; i++) {
        Row row = readRowBin(r);
        rows_.push_back(std::move(row));
        alive_.push_back(1);
        ++aliveCount_;
    }
    rebuildAllIndexes();
    return r.ok();
}

} // namespace db
