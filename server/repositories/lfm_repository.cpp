#include "lfm_repository.h"
#include "../../shared/db/db_schema.h"
#include "../../shared/constants.h"
#include <cstring>

namespace app {

LfmRepository::LfmRepository(db::Database& db)
    : db_(db),
      pTable_(db.table(db::tbl::LFM_P)),
      qTable_(db.table(db::tbl::LFM_Q)) {}

std::string LfmRepository::packVec(const float* vec, int k) {
    std::string s; s.resize((size_t)k * sizeof(float));
    std::memcpy(&s[0], vec, (size_t)k * sizeof(float));
    return s;
}

std::vector<float> LfmRepository::unpackVec(const std::string& blob, int k) {
    std::vector<float> v(k, 0.0f);
    if (blob.size() >= (size_t)k * sizeof(float))
        std::memcpy(v.data(), blob.data(), (size_t)k * sizeof(float));
    return v;
}

void LfmRepository::saveVector(db::Table& tbl, const std::string& idCol,
                                int64_t id, const float* vec, int k) {
    db::RowId rid = tbl.findOne(idCol, db::Value(id));
    db::Value blob = db::Value::makeBlob(packVec(vec, k));
    if (rid >= 0) {
        tbl.update(rid, db::col::VEC, std::move(blob));
    } else {
        db::Row row(&tbl.schema());
        row.set(idCol, id);
        row.set(db::col::VEC, std::move(blob));
        try { tbl.insert(std::move(row)); } catch (...) {}
    }
}

void LfmRepository::saveUserVector(int64_t userId, const float* vec, int k) {
    saveVector(pTable_, db::col::USER_ID, userId, vec, k);
}

void LfmRepository::saveItemVector(int64_t itemIdx, const float* vec, int k) {
    saveVector(qTable_, db::col::ITEM_IDX, itemIdx, vec, k);
}

std::optional<std::vector<float>> LfmRepository::findVector(db::Table& tbl,
                                                              const std::string& idCol,
                                                              int64_t id, int k) {
    db::RowId rid = tbl.findOne(idCol, db::Value(id));
    if (rid < 0) return std::nullopt;
    const db::Row* r = tbl.getRow(rid);
    if (!r) return std::nullopt;
    return unpackVec(r->getStr(db::col::VEC), k);
}

std::optional<std::vector<float>> LfmRepository::findUserVector(int64_t userId) {
    return findVector(pTable_, db::col::USER_ID, userId, K);
}

std::optional<std::vector<float>> LfmRepository::findItemVector(int64_t itemIdx) {
    return findVector(qTable_, db::col::ITEM_IDX, itemIdx, K);
}

std::vector<LfmVectorRecord> LfmRepository::findAllVectors(db::Table& tbl,
                                                            const std::string& idCol, int k) {
    std::vector<LfmVectorRecord> out;
    auto rids = tbl.scanAll();
    out.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = tbl.getRow(rid);
        if (!r) continue;
        LfmVectorRecord rec;
        rec.id  = r->getInt(idCol);
        rec.vec = unpackVec(r->getStr(db::col::VEC), k);
        out.push_back(std::move(rec));
    }
    return out;
}

std::vector<LfmVectorRecord> LfmRepository::findAllUserVectors() {
    return findAllVectors(pTable_, db::col::USER_ID, K);
}

std::vector<LfmVectorRecord> LfmRepository::findAllItemVectors() {
    return findAllVectors(qTable_, db::col::ITEM_IDX, K);
}

void LfmRepository::clearUsers() { pTable_.clear(); }
void LfmRepository::clearItems() { qTable_.clear(); }

} // namespace app
