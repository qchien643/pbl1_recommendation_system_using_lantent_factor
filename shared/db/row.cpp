#include "row.h"
#include <stdexcept>

namespace db {

void Row::allocateDefaults() {
    values_.clear();
    if (!schema_) return;
    values_.reserve(schema_->columnCount());
    for (int i = 0; i < schema_->columnCount(); i++) {
        const auto& c = schema_->at(i);
        switch (c.type) {
            case ColType::INT64:  values_.emplace_back((int64_t)0); break;
            case ColType::DOUBLE: values_.emplace_back((double)0.0); break;
            case ColType::STR:
            case ColType::BLOB:   values_.emplace_back(std::string()); break;
        }
    }
}

Row::Row(const Schema* schema) : schema_(schema) {
    allocateDefaults();
}

Row::Row(const Schema* schema, std::initializer_list<std::pair<std::string, Value>> kv)
    : schema_(schema)
{
    allocateDefaults();
    for (const auto& p : kv) set(p.first, p.second);
}

Value& Row::at(int idx) {
    if (idx < 0 || idx >= (int)values_.size())
        throw std::out_of_range("Row::at idx");
    return values_[idx];
}

const Value& Row::at(int idx) const {
    if (idx < 0 || idx >= (int)values_.size())
        throw std::out_of_range("Row::at idx");
    return values_[idx];
}

Value& Row::at(const std::string& name) {
    if (!schema_) throw std::runtime_error("Row::at — no schema");
    int i = schema_->indexOf(name);
    if (i < 0) throw std::out_of_range("Row::at — unknown column: " + name);
    return values_[i];
}

const Value& Row::at(const std::string& name) const {
    if (!schema_) throw std::runtime_error("Row::at — no schema");
    int i = schema_->indexOf(name);
    if (i < 0) throw std::out_of_range("Row::at — unknown column: " + name);
    return values_[i];
}

void Row::set(const std::string& name, Value v) {
    if (!schema_) throw std::runtime_error("Row::set — no schema");
    int i = schema_->indexOf(name);
    if (i < 0) throw std::out_of_range("Row::set — unknown column: " + name);
    values_[i] = std::move(v);
}

void Row::set(int idx, Value v) {
    if (idx < 0 || idx >= (int)values_.size())
        throw std::out_of_range("Row::set idx");
    values_[idx] = std::move(v);
}

} // namespace db
