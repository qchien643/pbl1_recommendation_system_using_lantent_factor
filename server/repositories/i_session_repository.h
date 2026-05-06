#ifndef APP_I_SESSION_REPOSITORY_H
#define APP_I_SESSION_REPOSITORY_H

#include <optional>
#include <string>
#include <vector>

namespace app {

struct SessionRecord {
    std::string code;
    std::string openedAt;
    std::string closedAt;
    std::string status;   // "O" = open, "C" = closed
};

class ISessionRepository {
public:
    virtual ~ISessionRepository() = default;

    virtual void                          save(const SessionRecord& s) = 0;       // upsert
    virtual std::optional<SessionRecord>  findByCode(const std::string& code) = 0;
    virtual void                          markClosed(const std::string& code, const std::string& closedAt) = 0;
    virtual std::vector<SessionRecord>    findAll() = 0;
};

} // namespace app

#endif
