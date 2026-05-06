#include "session_repository.h"
#include "../../shared/db/db_schema.h"

namespace app {

SessionRepository::SessionRepository(db::Database& db)
    : db_(db), table_(db.table(db::tbl::SESSIONS)) {}

SessionRecord SessionRepository::rowToRecord(const db::Row& r) const {
    SessionRecord s;
    s.code     = r.getStr(db::col::CODE);
    s.openedAt = r.getStr(db::col::OPENED_AT);
    s.closedAt = r.getStr(db::col::CLOSED_AT);
    s.status   = r.getStr(db::col::STATUS);
    return s;
}

void SessionRepository::save(const SessionRecord& s) {
    db::RowId rid = table_.findOne(db::col::CODE, db::Value(s.code));
    if (rid >= 0) {
        table_.update(rid, db::col::OPENED_AT, db::Value(s.openedAt));
        table_.update(rid, db::col::CLOSED_AT, db::Value(s.closedAt));
        table_.update(rid, db::col::STATUS,    db::Value(s.status));
    } else {
        db::Row row(&table_.schema());
        row.set(db::col::CODE,      s.code);
        row.set(db::col::OPENED_AT, s.openedAt);
        row.set(db::col::CLOSED_AT, s.closedAt);
        row.set(db::col::STATUS,    s.status);
        try { table_.insert(std::move(row)); } catch (...) {}
    }
}

std::optional<SessionRecord> SessionRepository::findByCode(const std::string& code) {
    db::RowId rid = table_.findOne(db::col::CODE, db::Value(code));
    if (rid < 0) return std::nullopt;
    const db::Row* r = table_.getRow(rid);
    return r ? std::optional<SessionRecord>(rowToRecord(*r)) : std::nullopt;
}

void SessionRepository::markClosed(const std::string& code, const std::string& closedAt) {
    db::RowId rid = table_.findOne(db::col::CODE, db::Value(code));
    if (rid < 0) return;
    table_.update(rid, db::col::CLOSED_AT, db::Value(closedAt));
    table_.update(rid, db::col::STATUS,    db::Value(std::string("C")));
}

std::vector<SessionRecord> SessionRepository::findAll() {
    std::vector<SessionRecord> out;
    for (db::RowId rid : table_.scanAll()) {
        const db::Row* r = table_.getRow(rid);
        if (r) out.push_back(rowToRecord(*r));
    }
    return out;
}

} // namespace app
