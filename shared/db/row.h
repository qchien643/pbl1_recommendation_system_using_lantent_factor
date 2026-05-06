#ifndef DB_ROW_H
#define DB_ROW_H

#include "schema.h"
#include "value.h"
#include <vector>
#include <string>
#include <initializer_list>
#include <utility>

namespace db {

class Row {
public:
    Row() : schema_(nullptr) {}
    explicit Row(const Schema* schema);
    Row(const Schema* schema, std::initializer_list<std::pair<std::string, Value>> kv);

    const Schema*  schema() const { return schema_; }

    Value&         at(int idx);
    const Value&   at(int idx) const;
    Value&         at(const std::string& name);
    const Value&   at(const std::string& name) const;

    void           set(const std::string& name, Value v);
    void           set(int idx, Value v);

    int64_t        getInt(const std::string& n)    const { return at(n).asInt(); }
    double         getDouble(const std::string& n) const { return at(n).asDouble(); }
    const std::string& getStr(const std::string& n) const { return at(n).asString(); }

    // Whole row as Values vector (debug)
    const std::vector<Value>& values() const { return values_; }
    std::vector<Value>&       mutableValues() { return values_; }

private:
    const Schema*       schema_;
    std::vector<Value>  values_;

    void allocateDefaults();
};

} // namespace db

#endif
