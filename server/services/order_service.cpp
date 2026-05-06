#include "order_service.h"
#include "../../shared/utils.h"
#include "../../shared/constants.h"
#include <cstdio>
#include <ctime>

namespace app {

OrderService::OrderService(IUserRepository& userRepo,
                            IMenuRepository& menuRepo,
                            ITransactionRepository& txnRepo,
                            LfmService& lfmService,
                            db::Database& db)
    : userRepo_(userRepo), menuRepo_(menuRepo), txnRepo_(txnRepo),
      lfmService_(lfmService), db_(db) {}

double OrderService::computeDiscount(double subtotal) {
    return subtotal >= DISCOUNT_THRESHOLD ? subtotal * DISCOUNT_RATE : 0.0;
}

OrderService::CreateResult OrderService::create(const CreateRequest& req) {
    CreateResult res;

    if (req.items.empty() || (int)req.items.size() > MAX_ITEMS) {
        res.failReason = "invalid item count";
        return res;
    }
    auto userOpt = userRepo_.findById(req.userId);
    if (!userOpt.has_value()) {
        res.failReason = "user not found";
        return res;
    }

    // Resolve item codes → menu records, build TransactionRecord + accumulate subtotal
    TransactionRecord txn;
    txn.userId      = req.userId;
    txn.sessionCode = req.sessionCode;

    char ts20[24];
    currentTimestamp(ts20, sizeof(ts20));
    txn.ts = ts20;

    std::vector<MenuItemRecord> resolved;
    double sub = 0.0;
    int seq = 0;
    for (const auto& in : req.items) {
        auto m = menuRepo_.findByCode(in.code);
        if (!m.has_value()) {
            res.failReason = "menu code not found: " + in.code;
            return res;
        }
        resolved.push_back(*m);
        TxnItemRecord it;
        it.seq      = seq++;
        it.itemCode = m->code;
        it.qty      = in.qty;
        it.price    = m->price;
        txn.items.push_back(it);
        sub += m->price * in.qty;
    }

    txn.subtotal = sub;
    txn.discount = computeDiscount(sub);
    txn.total    = sub - txn.discount;

    int64_t txnId = txnRepo_.save(txn);
    if (txnId < 0) {
        res.failReason = "txnRepo.save failed";
        return res;
    }
    txn.txnId = txnId;

    // Update user counters + LFM online
    userRepo_.incrementTotalOrders(req.userId);

    std::vector<int> itemIdxs, qtys;
    for (size_t i = 0; i < resolved.size(); i++) {
        itemIdxs.push_back((int)resolved[i].menuIdx);
        qtys.push_back((int)req.items[i].qty);
    }
    lfmService_.onlineUpdate(req.userId, itemIdxs, qtys);

    appendLegacyTextLog(txn, resolved);

    // Persist-on-order: flush .tbl files xuống disk
    db_.saveAll("data");

    res.success  = true;
    res.txnId    = txnId;
    res.subtotal = txn.subtotal;
    res.discount = txn.discount;
    res.total    = txn.total;
    res.ts       = txn.ts;
    return res;
}

void OrderService::appendLegacyTextLog(const TransactionRecord& txn,
                                        const std::vector<MenuItemRecord>& /*items*/) {
    FILE* f = fopen("data/transactions.log", "a");
    if (!f) return;
    auto userOpt = userRepo_.findById(txn.userId);
    const std::string phone = userOpt.has_value() ? userOpt->phone : "";
    fprintf(f, "%s|%s|%s", txn.ts.c_str(), txn.sessionCode.c_str(), phone.c_str());
    for (const auto& it : txn.items) {
        fprintf(f, "|%s,%d", it.itemCode.c_str(), (int)it.qty);
    }
    fprintf(f, "|%.0f|%.0f|%.0f\n", txn.subtotal, txn.discount, txn.total);
    fclose(f);
}

} // namespace app
