#ifndef APP_MESSAGE_ROUTER_H
#define APP_MESSAGE_ROUTER_H

#include "../controllers/auth_controller.h"
#include "../controllers/order_controller.h"
#include "../controllers/heartbeat_controller.h"
#include "../infra/event_listener.h"
#include "protocol_codec.h"

namespace app {

// Dispatch raw line đã nhận → parse → đúng controller method.
class MessageRouter {
public:
    MessageRouter(AuthController& auth,
                  OrderController& order,
                  HeartbeatController& heartbeat,
                  IServerEventListener& events);

    void route(int slot, const std::string& line);

private:
    AuthController&        auth_;
    OrderController&       order_;
    HeartbeatController&   heartbeat_;
    IServerEventListener&  events_;
};

} // namespace app

#endif
