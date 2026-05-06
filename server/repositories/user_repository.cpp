#include "user_repository.h"
#include "../../shared/db/db_schema.h"
#include <ctime>

namespace app {

UserRepository::UserRepository(db::Database& db)
    : db_(db), table_(db.table(db::tbl::USERS)) {}

UserRecord UserRepository::rowToRecord(const db::Row& r) const {
    UserRecord u;
    u.userId      = r.getInt(db::col::USER_ID);
    u.phone       = r.getStr(db::col::PHONE);
    u.name        = r.getStr(db::col::NAME);
    u.description = r.getStr(db::col::DESCRIPTION);
    u.totalOrders = r.getInt(db::col::TOTAL_ORDERS);
    u.createdAt   = r.getInt(db::col::CREATED_AT);
    return u;
}

std::optional<UserRecord> UserRepository::findByPhone(const std::string& phone) {
    db::RowId rid = table_.findOne(db::col::PHONE, db::Value(phone));
    if (rid < 0) return std::nullopt;
    const db::Row* r = table_.getRow(rid);
    return r ? std::optional<UserRecord>(rowToRecord(*r)) : std::nullopt;
}

std::optional<UserRecord> UserRepository::findById(int64_t userId) {
    db::RowId rid = table_.findOne(db::col::USER_ID, db::Value(userId));
    if (rid < 0) return std::nullopt;
    const db::Row* r = table_.getRow(rid);
    return r ? std::optional<UserRecord>(rowToRecord(*r)) : std::nullopt;
}

std::vector<UserRecord> UserRepository::findAll() {
    std::vector<UserRecord> out;
    auto rids = table_.scanAll();
    out.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = table_.getRow(rid);
        if (r) out.push_back(rowToRecord(*r));
    }
    return out;
}

int64_t UserRepository::save(const UserRecord& user) {
    // Update path: nếu userId hợp lệ và đã tồn tại
    if (user.userId >= 0) {
        db::RowId rid = table_.findOne(db::col::USER_ID, db::Value(user.userId));
        if (rid >= 0) {
            table_.update(rid, db::col::PHONE,        db::Value(user.phone));
            table_.update(rid, db::col::NAME,         db::Value(user.name));
            table_.update(rid, db::col::DESCRIPTION,  db::Value(user.description));
            table_.update(rid, db::col::TOTAL_ORDERS, db::Value(user.totalOrders));
            return user.userId;
        }
    }
    // Insert path: assign new userId
    int64_t newId = (user.userId >= 0) ? user.userId : nextUserId();
    db::Row row(&table_.schema());
    row.set(db::col::USER_ID,      newId);
    row.set(db::col::PHONE,        user.phone);
    row.set(db::col::NAME,         user.name);
    row.set(db::col::DESCRIPTION,  user.description);
    row.set(db::col::TOTAL_ORDERS, user.totalOrders);
    row.set(db::col::CREATED_AT,   user.createdAt > 0 ? user.createdAt : (int64_t)time(nullptr));
    try { table_.insert(std::move(row)); }
    catch (...) { return -1; }
    return newId;
}

void UserRepository::updateName(int64_t userId, const std::string& name, const std::string& description) {
    db::RowId rid = table_.findOne(db::col::USER_ID, db::Value(userId));
    if (rid < 0) return;
    table_.update(rid, db::col::NAME,        db::Value(name));
    table_.update(rid, db::col::DESCRIPTION, db::Value(description));
}

void UserRepository::incrementTotalOrders(int64_t userId) {
    db::RowId rid = table_.findOne(db::col::USER_ID, db::Value(userId));
    if (rid < 0) return;
    const db::Row* r = table_.getRow(rid);
    if (!r) return;
    int64_t cur = r->getInt(db::col::TOTAL_ORDERS);
    table_.update(rid, db::col::TOTAL_ORDERS, db::Value(cur + 1));
}

int64_t UserRepository::count() {
    return (int64_t)table_.size();
}

int64_t UserRepository::nextUserId() {
    // Tìm max user_id hiện có +1; sequential từ 0 nếu rỗng
    int64_t maxId = -1;
    for (db::RowId rid : table_.scanAll()) {
        const db::Row* r = table_.getRow(rid);
        if (!r) continue;
        int64_t id = r->getInt(db::col::USER_ID);
        if (id > maxId) maxId = id;
    }
    return maxId + 1;
}

} // namespace app
