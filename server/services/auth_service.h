#ifndef APP_AUTH_SERVICE_H
#define APP_AUTH_SERVICE_H

#include "../repositories/i_user_repository.h"
#include <optional>
#include <string>

namespace app {

class AuthService {
public:
    explicit AuthService(IUserRepository& userRepo);

    // Validate phone format: 10 chữ số, bắt đầu '0'.
    static bool                   isValidPhone(const std::string& phone);

    // Tìm user theo SDT; nếu không có → tạo mới (name/desc rỗng) và trả về userId.
    int64_t                       getOrCreate(const std::string& phone);

    std::optional<UserRecord>     findByPhone(const std::string& phone);
    std::optional<UserRecord>     findById(int64_t userId);

    // Gán tên + mô tả cho user (vd lần đầu register).
    void                          registerUser(int64_t userId,
                                                const std::string& name,
                                                const std::string& description);

    // Khi mỗi đơn submitted thành công.
    void                          incrementOrderCount(int64_t userId);

private:
    IUserRepository& userRepo_;
};

} // namespace app

#endif
