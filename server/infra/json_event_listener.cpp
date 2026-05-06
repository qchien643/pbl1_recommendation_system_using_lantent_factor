#include "json_event_listener.h"
#include <cstdio>

namespace app {

void JsonEventListener::onServerStarted(int port) {
    printf("{\"event\":\"server_started\",\"port\":%d}\n", port);
}
void JsonEventListener::onMenuLoaded(int count) {
    printf("{\"event\":\"menu_loaded\",\"count\":%d}\n", count);
}
void JsonEventListener::onClientConnect(int slot) {
    printf("{\"event\":\"client_connect\",\"slot\":%d}\n", slot);
}
void JsonEventListener::onClientDisconnect(int slot) {
    printf("{\"event\":\"client_disconnect\",\"slot\":%d}\n", slot);
}
void JsonEventListener::onSessionOpened(const std::string& code, const std::string& dt) {
    printf("{\"event\":\"session_opened\",\"code\":\"%s\",\"dateTime\":\"%s\"}\n", code.c_str(), dt.c_str());
}
void JsonEventListener::onSessionClosed(const std::string& dt) {
    printf("{\"event\":\"session_closed\",\"dateTime\":\"%s\"}\n", dt.c_str());
}
void JsonEventListener::onUserLogin(int slot, int64_t userId, const std::string& phone,
                                    const std::string& name, bool isNew, int64_t orderCount) {
    printf("{\"event\":\"user_login\",\"slot\":%d,\"userId\":%lld,\"phone\":\"%s\",\"name\":\"%s\",\"isNew\":%s,\"orderCount\":%lld}\n",
           slot, (long long)userId, phone.c_str(), name.c_str(),
           isNew ? "true" : "false", (long long)orderCount);
}
void JsonEventListener::onUserRegister(int slot, int64_t userId, const std::string& name) {
    printf("{\"event\":\"user_register\",\"slot\":%d,\"userId\":%lld,\"name\":\"%s\"}\n",
           slot, (long long)userId, name.c_str());
}
void JsonEventListener::onItemAdded(int slot, int64_t userId, const std::string& code, int excluded) {
    printf("{\"event\":\"item_added\",\"slot\":%d,\"userId\":%lld,\"code\":\"%s\",\"excluded\":%d}\n",
           slot, (long long)userId, code.c_str(), excluded);
}
void JsonEventListener::onOrderSubmitted(int slot, int64_t userId, int64_t orderId,
                                          int itemCount, double total, double discount) {
    printf("{\"event\":\"order_submitted\",\"slot\":%d,\"userId\":%lld,\"orderId\":%lld,\"items\":%d,\"total\":%.0f,\"discount\":%.0f}\n",
           slot, (long long)userId, (long long)orderId, itemCount, total, discount);
}
void JsonEventListener::onOrderRejected(int slot, const std::string& reason) {
    printf("{\"event\":\"order_rejected\",\"slot\":%d,\"reason\":\"%s\"}\n", slot, reason.c_str());
}
void JsonEventListener::onSuggestSent(int slot, int64_t userId,
                                       const std::vector<SuggestionItem>& items) {
    printf("{\"event\":\"suggest\",\"slot\":%d,\"userId\":%lld,\"items\":[", slot, (long long)userId);
    for (size_t i = 0; i < items.size(); i++) {
        if (i) printf(",");
        printf("{\"code\":\"%s\",\"score\":%.3f}", items[i].code.c_str(), items[i].score);
    }
    printf("]}\n");
}
void JsonEventListener::onHeartbeat(int slot, int clientId, int64_t ts) {
    printf("{\"event\":\"heartbeat\",\"slot\":%d,\"clientId\":%d,\"ts\":%lld}\n",
           slot, clientId, (long long)ts);
}
void JsonEventListener::onUnknownMessage(int slot, const std::string& raw) {
    printf("{\"event\":\"unknown\",\"slot\":%d,\"raw\":\"%s\"}\n", slot, raw.c_str());
}

} // namespace app
