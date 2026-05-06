#ifndef APP_ORDER_CONTROLLER_H
#define APP_ORDER_CONTROLLER_H

#include "../services/order_service.h"
#include "../services/menu_service.h"
#include "../services/lfm_service.h"
#include "../services/session_service.h"
#include "../network/tcp_server.h"
#include "../infra/event_listener.h"

namespace app {

class OrderController {
public:
    OrderController(OrderService& orderService,
                    MenuService& menuService,
                    LfmService& lfmService,
                    SessionService& sessionService,
                    TcpServer& tcpServer,
                    IServerEventListener& events);

    void handleItemAdded   (int slot, const std::string& payload);
    void handleOrderSubmit (int slot, const std::string& payload);

private:
    OrderService&           orderService_;
    MenuService&            menuService_;
    LfmService&             lfmService_;
    SessionService&         sessionService_;
    TcpServer&              tcpServer_;
    IServerEventListener&   events_;
};

} // namespace app

#endif
