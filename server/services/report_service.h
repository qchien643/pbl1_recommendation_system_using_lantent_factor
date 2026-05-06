#ifndef APP_REPORT_SERVICE_H
#define APP_REPORT_SERVICE_H

#include "../repositories/i_user_repository.h"
#include "../repositories/i_menu_repository.h"
#include "../repositories/i_transaction_repository.h"
#include "session_service.h"
#include <string>

namespace app {

class ReportService {
public:
    ReportService(IUserRepository& userRepo,
                  IMenuRepository& menuRepo,
                  ITransactionRepository& txnRepo,
                  SessionService& sessionService);

    // Ghi báo cáo cuối ca cho session hiện tại vào path. Lọc transactions theo
    // session_code = sessionService_.currentCode().
    bool writeForCurrentSession(const std::string& path);

private:
    IUserRepository& userRepo_;
    IMenuRepository& menuRepo_;
    ITransactionRepository& txnRepo_;
    SessionService& sessionService_;
};

} // namespace app

#endif
