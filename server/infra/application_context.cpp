#include "application_context.h"
#include <cstdio>

namespace app {

ApplicationContext::ApplicationContext(db::Database& db, IServerEventListener& events, int port)
    : db_(db), events_(events)
{
    // Repositories — lớp thấp nhất, chỉ phụ thuộc db
    userRepo_    = std::make_unique<UserRepository>(db);
    menuRepo_    = std::make_unique<MenuRepository>(db);
    txnRepo_     = std::make_unique<TransactionRepository>(db);
    lfmRepo_     = std::make_unique<LfmRepository>(db);
    sessionRepo_ = std::make_unique<SessionRepository>(db);

    // Services — phụ thuộc repos
    menuSvc_     = std::make_unique<MenuService>(*menuRepo_);
    authSvc_     = std::make_unique<AuthService>(*userRepo_);
    lfmSvc_      = std::make_unique<LfmService>(*lfmRepo_, *userRepo_, *menuRepo_, *txnRepo_);
    sessionSvc_  = std::make_unique<SessionService>(*sessionRepo_);
    orderSvc_    = std::make_unique<OrderService>(*userRepo_, *menuRepo_, *txnRepo_,
                                                   *lfmSvc_, db);
    reportSvc_   = std::make_unique<ReportService>(*userRepo_, *menuRepo_, *txnRepo_, *sessionSvc_);

    // Network
    tcp_         = std::make_unique<TcpServer>(port);

    // Controllers
    authCtrl_      = std::make_unique<AuthController>(*authSvc_, *menuSvc_, *lfmSvc_,
                                                       *sessionSvc_, *tcp_, events_);
    orderCtrl_     = std::make_unique<OrderController>(*orderSvc_, *menuSvc_, *lfmSvc_,
                                                        *sessionSvc_, *tcp_, events_);
    heartbeatCtrl_ = std::make_unique<HeartbeatController>(events_);

    // Router + lifecycle
    router_      = std::make_unique<MessageRouter>(*authCtrl_, *orderCtrl_, *heartbeatCtrl_, events_);
    lifecycle_   = std::make_unique<SessionLifecycle>(*sessionSvc_, *menuSvc_, *lfmSvc_,
                                                       *reportSvc_, *tcp_, db, events_);

    // Wire TcpServer hooks
    tcp_->setOnConnect    ([this](int slot){
        events_.onClientConnect(slot);
        // Replay session START + MENU_DATA cho client connect muộn
        if (sessionSvc_->isOpen()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s|%s",
                          sessionSvc_->currentCode().c_str(),
                          sessionSvc_->startTimestamp().c_str());
            tcp_->sendTo(slot, MSG_START, buf);
            tcp_->sendTo(slot, MSG_MENU_DATA, menuSvc_->serializeForBroadcast());
        }
    });
    tcp_->setOnDisconnect ([this](int slot){ events_.onClientDisconnect(slot); });
    tcp_->setOnLine       ([this](int slot, const std::string& line){ router_->route(slot, line); });
}

ApplicationContext::~ApplicationContext() = default;

} // namespace app
