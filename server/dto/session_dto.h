#ifndef APP_SESSION_DTO_H
#define APP_SESSION_DTO_H

#include <string>

namespace app {

struct SessionStartedEvent {
    std::string code;
    std::string dateTime;
};

struct SessionStoppedEvent {
    std::string dateTime;
};

struct HeartbeatRequest {
    int      clientId;
    int64_t  timestamp;
};

} // namespace app

#endif
