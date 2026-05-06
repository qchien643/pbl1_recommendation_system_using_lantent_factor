#ifndef APP_I_USER_REPOSITORY_H
#define APP_I_USER_REPOSITORY_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace app {

// Plain record (Entity) — không có method, chỉ mang dữ liệu.
struct UserRecord {
    int64_t     userId      = -1;
    std::string phone;
    std::string name;
    std::string description;
    int64_t     totalOrders = 0;
    int64_t     createdAt   = 0;   // unix timestamp (giây)
};

// Repository interface — abstract base; service phụ thuộc interface chứ không
// phụ thuộc concrete (DIP). Cho phép swap UserRepository → InMemoryUserRepository
// trong test.
class IUserRepository {
public:
    virtual ~IUserRepository() = default;

    virtual std::optional<UserRecord>      findByPhone(const std::string& phone) = 0;
    virtual std::optional<UserRecord>      findById(int64_t userId) = 0;
    virtual std::vector<UserRecord>        findAll() = 0;

    // Insert nếu user mới (userId = -1 hoặc 0 sẽ được assign), update nếu đã có. Trả về userId.
    virtual int64_t                        save(const UserRecord& user) = 0;

    virtual void                           updateName(int64_t userId,
                                                       const std::string& name,
                                                       const std::string& description) = 0;
    virtual void                           incrementTotalOrders(int64_t userId) = 0;

    virtual int64_t                        count() = 0;
    virtual int64_t                        nextUserId() = 0;
};

} // namespace app

#endif
