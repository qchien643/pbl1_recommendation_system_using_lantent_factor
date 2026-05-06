#ifndef APP_ORDER_SERVICE_H
#define APP_ORDER_SERVICE_H

#include "../repositories/i_user_repository.h"
#include "../repositories/i_menu_repository.h"
#include "../repositories/i_transaction_repository.h"
#include "lfm_service.h"
#include "../../shared/db/database.h"
#include <string>
#include <vector>

namespace app {

class OrderService {
public:
    struct LineItemInput {
        std::string code;
        int         qty;
    };

    struct CreateRequest {
        int64_t                     userId;
        std::string                 sessionCode;
        std::vector<LineItemInput>  items;
    };

    struct CreateResult {
        bool        success    = false;
        int64_t     txnId      = -1;
        double      subtotal   = 0.0;
        double      discount   = 0.0;
        double      total      = 0.0;
        std::string ts;
        std::string failReason;
    };

    OrderService(IUserRepository& userRepo,
                 IMenuRepository& menuRepo,
                 ITransactionRepository& txnRepo,
                 LfmService& lfmService,
                 db::Database& db);

    CreateResult            create(const CreateRequest& req);
    static double           computeDiscount(double subtotal);

private:
    IUserRepository&        userRepo_;
    IMenuRepository&        menuRepo_;
    ITransactionRepository& txnRepo_;
    LfmService&             lfmService_;
    db::Database&           db_;

    void                    appendLegacyTextLog(const TransactionRecord& txn,
                                                  const std::vector<MenuItemRecord>& items);
};

} // namespace app

#endif
