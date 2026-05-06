#ifndef APP_SESSION_REPOSITORY_H
#define APP_SESSION_REPOSITORY_H

#include "i_session_repository.h"
#include "../../shared/db/database.h"

namespace app {

class SessionRepository : public ISessionRepository {
public:
    explicit SessionRepository(db::Database& db);

    void                          save(const SessionRecord& s) override;
    std::optional<SessionRecord>  findByCode(const std::string& code) override;
    void                          markClosed(const std::string& code, const std::string& closedAt) override;
    std::vector<SessionRecord>    findAll() override;

private:
    db::Database& db_;
    db::Table&    table_;
    SessionRecord rowToRecord(const db::Row& r) const;
};

} // namespace app

#endif
