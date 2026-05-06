#ifndef APP_EVENT_LISTENER_H
#define APP_EVENT_LISTENER_H

#include "../dto/menu_dto.h"
#include <cstdint>
#include <string>
#include <vector>

namespace app {

// Observer interface — text mode + JSON mode subscribe để emit log/event ra stdout.
// Controllers gọi vào đây thay vì printf trực tiếp.
class IServerEventListener {
public:
    virtual ~IServerEventListener() = default;

    virtual void onServerStarted(int port) {}
    virtual void onMenuLoaded(int count) {}

    virtual void onClientConnect(int slot)              {}
    virtual void onClientDisconnect(int slot)           {}

    virtual void onSessionOpened(const std::string& code, const std::string& dt) {}
    virtual void onSessionClosed(const std::string& dt) {}

    virtual void onUserLogin(int slot, int64_t userId,
                              const std::string& phone, const std::string& name,
                              bool isNew, int64_t orderCount) {}
    virtual void onUserRegister(int slot, int64_t userId, const std::string& name) {}

    virtual void onItemAdded(int slot, int64_t userId, const std::string& code, int excluded) {}
    virtual void onOrderSubmitted(int slot, int64_t userId, int64_t orderId,
                                   int itemCount, double total, double discount) {}
    virtual void onOrderRejected(int slot, const std::string& reason) {}

    virtual void onSuggestSent(int slot, int64_t userId,
                                const std::vector<SuggestionItem>& items) {}

    virtual void onHeartbeat(int slot, int clientId, int64_t ts) {}
    virtual void onUnknownMessage(int slot, const std::string& raw) {}
};

// Null impl — dùng làm default khi chưa cần observe.
class NullEventListener : public IServerEventListener {};

} // namespace app

#endif
