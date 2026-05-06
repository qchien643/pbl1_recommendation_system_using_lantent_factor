#include "session_service.h"
#include "../../shared/utils.h"
#include <cstring>

namespace app {

SessionService::SessionService(ISessionRepository& repo) : repo_(repo) {}

bool SessionService::open(const std::string& code) {
    if (code.empty() || code.size() >= 10) return false;
    if (open_) return false;

    code_ = code;
    char buf[24];
    currentTimestamp(buf, sizeof(buf));
    startTs_ = buf;
    endTs_.clear();
    open_ = true;

    SessionRecord s;
    s.code     = code_;
    s.openedAt = startTs_;
    s.closedAt = "";
    s.status   = "O";
    repo_.save(s);
    return true;
}

bool SessionService::close(const std::string& code) {
    if (!open_) return false;
    if (code != code_) return false;

    char buf[24];
    currentTimestamp(buf, sizeof(buf));
    endTs_ = buf;

    repo_.markClosed(code_, endTs_);
    open_ = false;
    return true;
}

} // namespace app
