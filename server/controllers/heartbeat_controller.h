#ifndef APP_HEARTBEAT_CONTROLLER_H
#define APP_HEARTBEAT_CONTROLLER_H

#include "../infra/event_listener.h"
#include <string>

namespace app {

class HeartbeatController {
public:
    explicit HeartbeatController(IServerEventListener& events);
    void handleHeartbeat(int slot, const std::string& payload);

private:
    IServerEventListener& events_;
};

} // namespace app

#endif
