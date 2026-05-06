#ifndef APP_TRANSACTION_REPOSITORY_H
#define APP_TRANSACTION_REPOSITORY_H

#include "i_transaction_repository.h"
#include "../../shared/db/database.h"

namespace app {

// Backed by 2 tables (1-N):
//   transactions       — header row per txn, BTree(user_id), BTree(ts) → range queries.
//   transaction_items  — line items, BTree(txn_id) → join nhanh.
class TransactionRepository : public ITransactionRepository {
public:
    explicit TransactionRepository(db::Database& db);

    int64_t                          save(const TransactionRecord& txn) override;
    std::vector<TransactionRecord>   findAll() override;
    std::vector<TransactionRecord>   findByUserId(int64_t userId) override;
    std::vector<TransactionRecord>   findByDateRange(const std::string& fromTs, const std::string& toTs) override;
    std::vector<TransactionRecord>   findBySessionCode(const std::string& sessionCode) override;

    int64_t                          count() override;
    int64_t                          nextTxnId() override;

private:
    db::Database& db_;
    db::Table&    txnTable_;
    db::Table&    itemTable_;

    TransactionRecord       rowToTxn(const db::Row& r) const;
    TxnItemRecord           rowToItem(const db::Row& r) const;
    void                    loadItemsInto(TransactionRecord& txn) const;
};

} // namespace app

#endif
