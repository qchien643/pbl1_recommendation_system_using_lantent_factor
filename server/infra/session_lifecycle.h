#ifndef APP_SESSION_LIFECYCLE_H
#define APP_SESSION_LIFECYCLE_H

#include "../services/session_service.h"
#include "../services/menu_service.h"
#include "../services/lfm_service.h"
#include "../services/report_service.h"
#include "../network/tcp_server.h"
#include "../../shared/db/database.h"
#include "event_listener.h"

namespace app {

// Manage session open/close lifecycle: gọi service + broadcast START/STOP +
// trigger save model + write report.
class SessionLifecycle {
public:
    SessionLifecycle(SessionService& session,
                     MenuService& menu,
                     LfmService& lfm,
                     ReportService& report,
                     TcpServer& tcp,
                     db::Database& db,
                     IServerEventListener& events);

    bool start(const std::string& code);
    bool stop (const std::string& code);

private:
    SessionService&      session_;
    MenuService&         menu_;
    LfmService&          lfm_;
    ReportService&       report_;
    TcpServer&           tcp_;
    db::Database&        db_;
    IServerEventListener& events_;
};

} // namespace app

#endif
