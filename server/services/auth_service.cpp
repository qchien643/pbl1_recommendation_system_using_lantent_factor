#include "auth_service.h"
#include <cctype>
#include <ctime>

namespace app {

AuthService::AuthService(IUserRepository& userRepo) : userRepo_(userRepo) {}

bool AuthService::isValidPhone(const std::string& phone) {
    if (phone.size() != 10) return false;
    if (phone[0] != '0') return false;
    for (char c : phone) if (!isdigit((unsigned char)c)) return false;
    return true;
}

int64_t AuthService::getOrCreate(const std::string& phone) {
    auto existing = userRepo_.findByPhone(phone);
    if (existing.has_value()) return existing->userId;

    UserRecord u;
    u.userId      = -1;   // assign automatically
    u.phone       = phone;
    u.name        = "";
    u.description = "";
    u.totalOrders = 0;
    u.createdAt   = (int64_t)time(nullptr);
    return userRepo_.save(u);
}

std::optional<UserRecord> AuthService::findByPhone(const std::string& phone) {
    return userRepo_.findByPhone(phone);
}

std::optional<UserRecord> AuthService::findById(int64_t userId) {
    return userRepo_.findById(userId);
}

void AuthService::registerUser(int64_t userId, const std::string& name, const std::string& description) {
    userRepo_.updateName(userId, name, description);
}

void AuthService::incrementOrderCount(int64_t userId) {
    userRepo_.incrementTotalOrders(userId);
}

} // namespace app
