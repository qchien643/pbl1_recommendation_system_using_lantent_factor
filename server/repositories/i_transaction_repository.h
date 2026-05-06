#ifndef APP_I_TRANSACTION_REPOSITORY_H
#define APP_I_TRANSACTION_REPOSITORY_H

#include <cstdint>
#include <string>
#include <vector>

namespace app {

struct TxnItemRecord {
    int64_t     txnId    = -1;
    int64_t     seq      = 0;
    std::string itemCode;
    int64_t     qty      = 0;
    double      price    = 0.0;
};

struct TransactionRecord {
    int64_t                    txnId       = -1;
    int64_t                    userId      = -1;
    std::string                sessionCode;
    std::string                ts;            // "YYYY-MM-DD HH:MM:SS"
    double                     subtotal    = 0.0;
    double                     discount    = 0.0;
    double                     total       = 0.0;
    std::vector<TxnItemRecord> items;
};

class ITransactionRepository {
public:
    virtual ~ITransactionRepository() = default;

    // Insert; gán txnId tự động (nếu txn.txnId < 0). Returns assigned txnId.
    virtual int64_t                          save(const TransactionRecord& txn) = 0;

    virtual std::vector<TransactionRecord>   findAll() = 0;
    virtual std::vector<TransactionRecord>   findByUserId(int64_t userId) = 0;
    virtual std::vector<TransactionRecord>   findByDateRange(const std::string& fromTs, const std::string& toTs) = 0;
    virtual std::vector<TransactionRecord>   findBySessionCode(const std::string& sessionCode) = 0;

    virtual int64_t                          count() = 0;
    virtual int64_t                          nextTxnId() = 0;
};

} // namespace app

#endif
