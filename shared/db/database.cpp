#include "database.h"
#include <stdexcept>
#include <fstream>

namespace db {

Database& Database::instance() {
    static Database g;
    return g;
}

Table& Database::createTable(const std::string& name, Schema schema) {
    if (tables_.count(name))
        throw std::runtime_error("Database::createTable — duplicate: " + name);
    auto t = std::make_unique<Table>(name, std::move(schema));
    Table* raw = t.get();
    tables_.emplace(name, std::move(t));
    return *raw;
}

Table& Database::table(const std::string& name) {
    auto it = tables_.find(name);
    if (it == tables_.end()) throw std::runtime_error("Database::table — unknown: " + name);
    return *it->second;
}

const Table& Database::table(const std::string& name) const {
    auto it = tables_.find(name);
    if (it == tables_.end()) throw std::runtime_error("Database::table — unknown: " + name);
    return *it->second;
}

bool Database::has(const std::string& name) const {
    return tables_.count(name) != 0;
}

static std::string joinPath(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
#ifdef _WIN32
    return dir + "\\" + file;
#else
    return dir + "/" + file;
#endif
}

bool Database::openAll(const std::string& dataDir) {
    bool allOk = true;
    for (auto& kv : tables_) {
        std::string path = joinPath(dataDir, kv.first + ".tbl");
        std::ifstream probe(path, std::ios::binary);
        if (!probe) continue;  // bảng chưa có file — coi như mới
        probe.close();
        if (!kv.second->loadFromFile(path)) allOk = false;
    }
    return allOk;
}

bool Database::saveAll(const std::string& dataDir) {
    bool allOk = true;
    for (auto& kv : tables_) {
        std::string path = joinPath(dataDir, kv.first + ".tbl");
        if (!kv.second->saveToFile(path)) allOk = false;
    }
    return allOk;
}

std::vector<std::string> Database::tableNames() const {
    std::vector<std::string> v;
    v.reserve(tables_.size());
    for (auto& kv : tables_) v.push_back(kv.first);
    return v;
}

} // namespace db
