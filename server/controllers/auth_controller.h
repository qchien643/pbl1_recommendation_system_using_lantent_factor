#ifndef APP_AUTH_CONTROLLER_H
#define APP_AUTH_CONTROLLER_H

#include "../services/auth_service.h"
#include "../services/menu_service.h"
#include "../services/lfm_service.h"
#include "../services/session_service.h"
#include "../network/tcp_server.h"
#include "../infra/event_listener.h"

namespace app {

class AuthController {
public:
    AuthController(AuthService& authService,
                   MenuService& menuService,
                   LfmService& lfmService,
                   SessionService& sessionService,
                   TcpServer& tcpServer,
                   IServerEventListener& events);

    void handleLogin   (int slot, const std::string& payload);
    void handleRegister(int slot, const std::string& payload);

private:
    AuthService&            authService_;
    MenuService&            menuService_;
    LfmService&             lfmService_;
    SessionService&         sessionService_;
    TcpServer&              tcpServer_;
    IServerEventListener&   events_;

    void                    sendUserAck(int slot, int64_t userId, bool isNew,
                                         int64_t orderCount, const std::string& name);
    void                    sendSuggestions(int slot, int64_t userId);
};

} // namespace app

#endif
