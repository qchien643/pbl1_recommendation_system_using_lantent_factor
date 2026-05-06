#include "heartbeat_controller.h"
#include "../../shared/utils.h"
#include <cstdlib>

namespace app {

HeartbeatController::HeartbeatController(IServerEventListener& events) : events_(events) {}

void HeartbeatController::handleHeartbeat(int slot, const std::string& payload) {
    char tokens[2][256];
    int n = splitByPipe(payload.c_str(), tokens, 2);
    int     clientId = (n >= 1) ? atoi(tokens[0]) : -1;
    int64_t ts       = (n >= 2) ? atoll(tokens[1]) : 0;
    events_.onHeartbeat(slot, clientId, ts);
}

} // namespace app
