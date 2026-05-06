#ifndef APP_TEXT_LOG_LISTENER_H
#define APP_TEXT_LOG_LISTENER_H

#include "event_listener.h"

namespace app {

// Text mode: in log thân thiện với người ra stdout (giống socket_server.cpp cũ).
class TextLogListener : public IServerEventListener {
public:
    void onServerStarted(int port) override;
    void onMenuLoaded(int count) override;

    void onClientConnect(int slot) override;
    void onClientDisconnect(int slot) override;

    void onSessionOpened(const std::string& code, const std::string& dt) override;
    void onSessionClosed(const std::string& dt) override;

    void onUserLogin(int slot, int64_t userId, const std::string& phone, const std::string& name,
                     bool isNew, int64_t orderCount) override;
    void onUserRegister(int slot, int64_t userId, const std::string& name) override;

    void onItemAdded(int slot, int64_t userId, const std::string& code, int excluded) override;
    void onOrderSubmitted(int slot, int64_t userId, int64_t orderId,
                          int itemCount, double total, double discount) override;
    void onOrderRejected(int slot, const std::string& reason) override;

    void onSuggestSent(int slot, int64_t userId,
                        const std::vector<SuggestionItem>& items) override;
    void onHeartbeat(int slot, int clientId, int64_t ts) override;
    void onUnknownMessage(int slot, const std::string& raw) override;
};

} // namespace app

#endif
