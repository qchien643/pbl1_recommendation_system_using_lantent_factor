#include "transaction_repository.h"
#include "../../shared/db/db_schema.h"
#include <algorithm>

namespace app {

TransactionRepository::TransactionRepository(db::Database& db)
    : db_(db),
      txnTable_(db.table(db::tbl::TRANSACTIONS)),
      itemTable_(db.table(db::tbl::TXN_ITEMS)) {}

TransactionRecord TransactionRepository::rowToTxn(const db::Row& r) const {
    TransactionRecord t;
    t.txnId       = r.getInt(db::col::TXN_ID);
    t.userId      = r.getInt(db::col::USER_ID);
    t.sessionCode = r.getStr(db::col::SESSION_CODE);
    t.ts          = r.getStr(db::col::TS);
    t.subtotal    = r.getDouble(db::col::SUBTOTAL);
    t.discount    = r.getDouble(db::col::DISCOUNT);
    t.total       = r.getDouble(db::col::TOTAL);
    return t;
}

TxnItemRecord TransactionRepository::rowToItem(const db::Row& r) const {
    TxnItemRecord it;
    it.txnId    = r.getInt(db::col::TXN_ID);
    it.seq      = r.getInt(db::col::SEQ);
    it.itemCode = r.getStr(db::col::ITEM_CODE);
    it.qty      = r.getInt(db::col::QTY);
    it.price    = r.getDouble(db::col::PRICE);
    return it;
}

void TransactionRepository::loadItemsInto(TransactionRecord& txn) const {
    auto rids = itemTable_.find(db::col::TXN_ID, db::Value(txn.txnId));   // BTree
    txn.items.clear();
    txn.items.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = itemTable_.getRow(rid);
        if (r) txn.items.push_back(rowToItem(*r));
    }
    // Sắp xếp theo seq để giữ thứ tự
    std::sort(txn.items.begin(), txn.items.end(),
              [](const TxnItemRecord& a, const TxnItemRecord& b) { return a.seq < b.seq; });
}

int64_t TransactionRepository::save(const TransactionRecord& txn) {
    int64_t id = (txn.txnId >= 0) ? txn.txnId : nextTxnId();

    // Header
    db::Row hdr(&txnTable_.schema());
    hdr.set(db::col::TXN_ID,       id);
    hdr.set(db::col::USER_ID,      txn.userId);
    hdr.set(db::col::SESSION_CODE, txn.sessionCode);
    hdr.set(db::col::TS,           txn.ts);
    hdr.set(db::col::SUBTOTAL,     txn.subtotal);
    hdr.set(db::col::DISCOUNT,     txn.discount);
    hdr.set(db::col::TOTAL,        txn.total);
    try { txnTable_.insert(std::move(hdr)); } catch (...) { return -1; }

    // Items (1-N)
    for (size_t i = 0; i < txn.items.size(); i++) {
        const auto& it = txn.items[i];
        db::Row item(&itemTable_.schema());
        item.set(db::col::TXN_ID,    id);
        item.set(db::col::SEQ,       (int64_t)i);
        item.set(db::col::ITEM_CODE, it.itemCode);
        item.set(db::col::QTY,       it.qty);
        item.set(db::col::PRICE,     it.price);
        itemTable_.insert(std::move(item));
    }

    return id;
}

std::vector<TransactionRecord> TransactionRepository::findAll() {
    std::vector<TransactionRecord> out;
    auto rids = txnTable_.scanAll();
    out.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = txnTable_.getRow(rid);
        if (!r) continue;
        TransactionRecord t = rowToTxn(*r);
        loadItemsInto(t);
        out.push_back(std::move(t));
    }
    return out;
}

std::vector<TransactionRecord> TransactionRepository::findByUserId(int64_t userId) {
    auto rids = txnTable_.find(db::col::USER_ID, db::Value(userId));   // BTree
    std::vector<TransactionRecord> out;
    out.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = txnTable_.getRow(rid);
        if (!r) continue;
        TransactionRecord t = rowToTxn(*r);
        loadItemsInto(t);
        out.push_back(std::move(t));
    }
    return out;
}

std::vector<TransactionRecord> TransactionRepository::findByDateRange(const std::string& fromTs, const std::string& toTs) {
    auto rids = txnTable_.range(db::col::TS, db::Value(fromTs), db::Value(toTs));  // BTree range
    std::vector<TransactionRecord> out;
    out.reserve(rids.size());
    for (db::RowId rid : rids) {
        const db::Row* r = txnTable_.getRow(rid);
        if (!r) continue;
        TransactionRecord t = rowToTxn(*r);
        loadItemsInto(t);
        out.push_back(std::move(t));
    }
    return out;
}

std::vector<TransactionRecord> TransactionRepository::findBySessionCode(const std::string& sessionCode) {
    // Không có index dedicated trên session_code — linear scan + filter (số session/ca thường nhỏ).
    std::vector<TransactionRecord> out;
    for (db::RowId rid : txnTable_.scanAll()) {
        const db::Row* r = txnTable_.getRow(rid);
        if (!r) continue;
        if (r->getStr(db::col::SESSION_CODE) != sessionCode) continue;
        TransactionRecord t = rowToTxn(*r);
        loadItemsInto(t);
        out.push_back(std::move(t));
    }
    return out;
}

int64_t TransactionRepository::count() {
    return (int64_t)txnTable_.size();
}

int64_t TransactionRepository::nextTxnId() {
    int64_t maxId = -1;
    for (db::RowId rid : txnTable_.scanAll()) {
        const db::Row* r = txnTable_.getRow(rid);
        if (!r) continue;
        int64_t id = r->getInt(db::col::TXN_ID);
        if (id > maxId) maxId = id;
    }
    return maxId + 1;
}

} // namespace app
