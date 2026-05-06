#include "message_router.h"

namespace app {

MessageRouter::MessageRouter(AuthController& auth,
                              OrderController& order,
                              HeartbeatController& heartbeat,
                              IServerEventListener& events)
    : auth_(auth), order_(order), heartbeat_(heartbeat), events_(events) {}

void MessageRouter::route(int slot, const std::string& line) {
    ParsedMsg pm;
    if (!ProtocolCodec::parse(line, pm)) {
        events_.onUnknownMessage(slot, line);
        return;
    }
    std::string payload = pm.payload;
    switch (pm.type) {
        case MSG_USER_LOGIN:    auth_.handleLogin(slot, payload);          break;
        case MSG_USER_REGISTER: auth_.handleRegister(slot, payload);       break;
        case MSG_ITEM_ADDED:    order_.handleItemAdded(slot, payload);     break;
        case MSG_ORDER_SUBMIT:  order_.handleOrderSubmit(slot, payload);   break;
        case MSG_HEARTBEAT:     heartbeat_.handleHeartbeat(slot, payload); break;
        default:                events_.onUnknownMessage(slot, line);     break;
    }
}

} // namespace app
