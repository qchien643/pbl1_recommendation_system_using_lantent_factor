#ifndef APP_APPLICATION_CONTEXT_H
#define APP_APPLICATION_CONTEXT_H

#include "../../shared/db/database.h"
#include "../repositories/user_repository.h"
#include "../repositories/menu_repository.h"
#include "../repositories/transaction_repository.h"
#include "../repositories/lfm_repository.h"
#include "../repositories/session_repository.h"
#include "../services/auth_service.h"
#include "../services/menu_service.h"
#include "../services/order_service.h"
#include "../services/lfm_service.h"
#include "../services/session_service.h"
#include "../services/report_service.h"
#include "../controllers/auth_controller.h"
#include "../controllers/order_controller.h"
#include "../controllers/heartbeat_controller.h"
#include "../network/tcp_server.h"
#include "../network/message_router.h"
#include "session_lifecycle.h"
#include "event_listener.h"
#include <memory>

namespace app {

// DI container manual: tạo + own + wire tất cả components theo đúng thứ tự
// dependency. Singleton (instance()) để main_server lấy ra dùng.
class ApplicationContext {
public:
    ApplicationContext(db::Database& db,
                        IServerEventListener& events,
                        int port);
    ~ApplicationContext();

    // Top-level access
    TcpServer&            tcpServer()       { return *tcp_; }
    MessageRouter&        router()          { return *router_; }
    SessionLifecycle&     sessionLifecycle() { return *lifecycle_; }
    SessionService&       sessionService()  { return *sessionSvc_; }
    MenuService&          menuService()     { return *menuSvc_; }
    LfmService&           lfmService()      { return *lfmSvc_; }
    AuthService&          authService()     { return *authSvc_; }
    OrderService&         orderService()    { return *orderSvc_; }
    ReportService&        reportService()   { return *reportSvc_; }
    IServerEventListener& events()          { return events_; }
    db::Database&         database()        { return db_; }

private:
    db::Database&         db_;
    IServerEventListener& events_;

    // Repositories
    std::unique_ptr<UserRepository>         userRepo_;
    std::unique_ptr<MenuRepository>         menuRepo_;
    std::unique_ptr<TransactionRepository>  txnRepo_;
    std::unique_ptr<LfmRepository>          lfmRepo_;
    std::unique_ptr<SessionRepository>      sessionRepo_;

    // Services
    std::unique_ptr<MenuService>            menuSvc_;
    std::unique_ptr<AuthService>            authSvc_;
    std::unique_ptr<LfmService>             lfmSvc_;
    std::unique_ptr<SessionService>         sessionSvc_;
    std::unique_ptr<OrderService>           orderSvc_;
    std::unique_ptr<ReportService>          reportSvc_;

    // Controllers
    std::unique_ptr<AuthController>         authCtrl_;
    std::unique_ptr<OrderController>        orderCtrl_;
    std::unique_ptr<HeartbeatController>    heartbeatCtrl_;

    // Network + lifecycle
    std::unique_ptr<TcpServer>              tcp_;
    std::unique_ptr<MessageRouter>          router_;
    std::unique_ptr<SessionLifecycle>       lifecycle_;
};

} // namespace app

#endif
