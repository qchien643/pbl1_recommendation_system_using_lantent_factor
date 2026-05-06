#ifndef APP_SESSION_SERVICE_H
#define APP_SESSION_SERVICE_H

#include "../repositories/i_session_repository.h"
#include <string>

namespace app {

class SessionService {
public:
    explicit SessionService(ISessionRepository& repo);

    bool                  open(const std::string& code);     // returns false nếu code rỗng / quá dài / đã mở
    bool                  close(const std::string& code);    // chỉ đóng nếu code khớp
    bool                  isOpen() const { return open_; }
    bool                  matchCode(const std::string& code) const { return code == code_; }

    const std::string&    currentCode()    const { return code_; }
    const std::string&    startTimestamp() const { return startTs_; }
    const std::string&    endTimestamp()   const { return endTs_; }

private:
    ISessionRepository& repo_;
    bool         open_  = false;
    std::string  code_;
    std::string  startTs_;
    std::string  endTs_;
};

} // namespace app

#endif
