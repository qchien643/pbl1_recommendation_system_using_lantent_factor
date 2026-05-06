#include "text_log_listener.h"
#include <cstdio>

namespace app {

void TextLogListener::onServerStarted(int port) {
    printf("[Server] Listening on port %d\n", port);
}
void TextLogListener::onMenuLoaded(int count) {
    printf("[Server] Loaded %d menu items\n", count);
}
void TextLogListener::onClientConnect(int slot) {
    printf("[Server] Client connect slot=%d\n", slot);
}
void TextLogListener::onClientDisconnect(int slot) {
    printf("[Server] Client disconnect slot=%d\n", slot);
}
void TextLogListener::onSessionOpened(const std::string& code, const std::string& dt) {
    printf("[Server] Session OPENED code=%s start=%s\n", code.c_str(), dt.c_str());
}
void TextLogListener::onSessionClosed(const std::string& dt) {
    printf("[Server] Session CLOSED at %s. Report + models saved.\n", dt.c_str());
}
void TextLogListener::onUserLogin(int slot, int64_t userId, const std::string& phone,
                                  const std::string& name, bool isNew, int64_t orderCount) {
    printf("[Server] LOGIN slot=%d userId=%lld phone=%s name=%s isNew=%d orders=%lld\n",
           slot, (long long)userId, phone.c_str(), name.c_str(), (int)isNew, (long long)orderCount);
}
void TextLogListener::onUserRegister(int slot, int64_t userId, const std::string& name) {
    printf("[Server] REGISTER slot=%d userId=%lld name=%s\n", slot, (long long)userId, name.c_str());
}
void TextLogListener::onItemAdded(int slot, int64_t userId, const std::string& code, int excluded) {
    printf("[Server] ITEM_ADDED slot=%d userId=%lld code=%s exclude=%d\n",
           slot, (long long)userId, code.c_str(), excluded);
}
void TextLogListener::onOrderSubmitted(int slot, int64_t userId, int64_t orderId,
                                        int itemCount, double total, double discount) {
    printf("[Server] ORDER_SUBMIT slot=%d userId=%lld oid=%lld items=%d total=%.0f disc=%.0f\n",
           slot, (long long)userId, (long long)orderId, itemCount, total, discount);
}
void TextLogListener::onOrderRejected(int slot, const std::string& reason) {
    printf("[Server] ORDER REJECT slot=%d reason=%s\n", slot, reason.c_str());
}
void TextLogListener::onSuggestSent(int slot, int64_t userId,
                                    const std::vector<SuggestionItem>& items) {
    printf("[Server] SUGGEST slot=%d userId=%lld:", slot, (long long)userId);
    for (const auto& it : items) printf(" %s(%.2f)", it.code.c_str(), it.score);
    printf("\n");
}
void TextLogListener::onHeartbeat(int /*slot*/, int /*clientId*/, int64_t /*ts*/) {
    // Silent — quá nhiều spam
}
void TextLogListener::onUnknownMessage(int slot, const std::string& raw) {
    printf("[Server] Unknown from slot %d: '%s'\n", slot, raw.c_str());
}

} // namespace app
