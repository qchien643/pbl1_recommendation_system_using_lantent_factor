#include "menu_repository.h"
#include "../../shared/db/db_schema.h"

namespace app {

MenuRepository::MenuRepository(db::Database& db)
    : db_(db), table_(db.table(db::tbl::MENU)) {}

MenuItemRecord MenuRepository::rowToRecord(int64_t idx, const db::Row& r) const {
    MenuItemRecord m;
    m.menuIdx  = idx;
    m.code     = r.getStr(db::col::CODE);
    m.name     = r.getStr(db::col::NAME);
    m.price    = r.getDouble(db::col::PRICE);
    m.category = r.getStr(db::col::CATEGORY);
    return m;
}

void MenuRepository::clear() {
    table_.clear();
}

int64_t MenuRepository::save(const MenuItemRecord& item) {
    db::Row row(&table_.schema());
    row.set(db::col::CODE,     item.code);
    row.set(db::col::NAME,     item.name);
    row.set(db::col::PRICE,    item.price);
    row.set(db::col::CATEGORY, item.category);
    try {
        return (int64_t)table_.insert(std::move(row));
    } catch (...) {
        return -1;
    }
}

std::optional<MenuItemRecord> MenuRepository::findByCode(const std::string& code) {
    db::RowId rid = table_.findOne(db::col::CODE, db::Value(code));
    if (rid < 0) return std::nullopt;
    const db::Row* r = table_.getRow(rid);
    return r ? std::optional<MenuItemRecord>(rowToRecord(rid, *r)) : std::nullopt;
}

int64_t MenuRepository::findIndexByCode(const std::string& code) {
    db::RowId rid = table_.findOne(db::col::CODE, db::Value(code));
    return rid;
}

std::optional<MenuItemRecord> MenuRepository::findByIndex(int64_t menuIdx) {
    const db::Row* r = table_.getRow((db::RowId)menuIdx);
    if (!r) return std::nullopt;
    return rowToRecord(menuIdx, *r);
}

std::vector<MenuItemRecord> MenuRepository::findAll() {
    std::vector<MenuItemRecord> out;
    auto rids = table_.scanAll();
    out.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = table_.getRow(rid);
        if (r) out.push_back(rowToRecord(rid, *r));
    }
    return out;
}

int64_t MenuRepository::count() {
    return (int64_t)table_.size();
}

} // namespace app
