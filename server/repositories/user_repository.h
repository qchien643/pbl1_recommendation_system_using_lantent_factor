#ifndef APP_USER_REPOSITORY_H
#define APP_USER_REPOSITORY_H

#include "i_user_repository.h"
#include "../../shared/db/database.h"

namespace app {

// Concrete implementation backed by db::Table "users".
// HashIndex(phone, UNIQUE) → findByPhone O(1).
// HashIndex(user_id, UNIQUE) → findById O(1).
class UserRepository : public IUserRepository {
public:
    explicit UserRepository(db::Database& db);

    std::optional<UserRecord> findByPhone(const std::string& phone) override;
    std::optional<UserRecord> findById(int64_t userId) override;
    std::vector<UserRecord>   findAll() override;

    int64_t                   save(const UserRecord& user) override;
    void                      updateName(int64_t userId,
                                          const std::string& name,
                                          const std::string& description) override;
    void                      incrementTotalOrders(int64_t userId) override;

    int64_t                   count() override;
    int64_t                   nextUserId() override;

private:
    db::Database& db_;
    db::Table&    table_;

    UserRecord    rowToRecord(const db::Row& r) const;
};

} // namespace app

#endif
