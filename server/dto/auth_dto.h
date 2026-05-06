#ifndef APP_AUTH_DTO_H
#define APP_AUTH_DTO_H

#include <cstdint>
#include <string>

namespace app {

struct LoginRequest {
    int         clientId;
    std::string phone;
};

struct LoginResponse {
    int64_t     userId;
    bool        isNew;        // true nếu user mới (name rỗng) → cần register
    int64_t     orderCount;
    std::string name;
};

struct RegisterRequest {
    int         clientId;
    std::string phone;
    std::string name;
    std::string description;
};

} // namespace app

#endif
