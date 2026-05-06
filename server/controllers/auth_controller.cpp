#include "auth_controller.h"
#include "../../shared/utils.h"
#include <cstdio>
#include <cstring>

namespace app {

AuthController::AuthController(AuthService& authService,
                                MenuService& menuService,
                                LfmService& lfmService,
                                SessionService& sessionService,
                                TcpServer& tcpServer,
                                IServerEventListener& events)
    : authService_(authService), menuService_(menuService),
      lfmService_(lfmService), sessionService_(sessionService),
      tcpServer_(tcpServer), events_(events) {}

void AuthController::sendUserAck(int slot, int64_t userId, bool isNew, int64_t orderCount,
                                  const std::string& name) {
    char p[200];
    snprintf(p, sizeof(p), "%lld|%s|%lld|%s",
             (long long)userId, isNew ? "true" : "false",
             (long long)orderCount, name.c_str());
    tcpServer_.sendTo(slot, MSG_USER_ACK, p);
}

void AuthController::sendSuggestions(int slot, int64_t userId) {
    auto top = lfmService_.topK(userId, {}, 3);
    std::string p;
    for (size_t i = 0; i < top.size(); i++) {
        if (i) p += "|";
        auto m = menuService_.findByIndex(top[i].first);
        if (!m.has_value()) continue;
        char buf[64];
        snprintf(buf, sizeof(buf), "%s,%.3f", m->code.c_str(), top[i].second);
        p += buf;
    }
    tcpServer_.sendTo(slot, MSG_SUGGEST, p);

    std::vector<SuggestionItem> items;
    for (auto& s : top) {
        auto m = menuService_.findByIndex(s.first);
        if (!m.has_value()) continue;
        items.push_back({ m->code, s.second });
    }
    events_.onSuggestSent(slot, userId, items);
}

void AuthController::handleLogin(int slot, const std::string& payload) {
    if (!sessionService_.isOpen()) return;

    char tokens[4][256];
    int n = splitByPipe(payload.c_str(), tokens, 4);
    if (n < 2) return;
    int clientId = atoi(tokens[0]);
    std::string phone = tokens[1];

    if (!AuthService::isValidPhone(phone)) return;

    int64_t userId = authService_.getOrCreate(phone);
    if (userId < 0) return;

    auto userOpt = authService_.findById(userId);
    if (!userOpt.has_value()) return;

    bool isNew = userOpt->name.empty();
    if (isNew) lfmService_.initUserVector(userId);

    sendUserAck(slot, userId, isNew, userOpt->totalOrders, userOpt->name);
    events_.onUserLogin(slot, userId, phone, userOpt->name, isNew, userOpt->totalOrders);

    if (!isNew) sendSuggestions(slot, userId);
    (void)clientId;
}

void AuthController::handleRegister(int slot, const std::string& payload) {
    if (!sessionService_.isOpen()) return;

    char tokens[5][256];
    int n = splitByPipe(payload.c_str(), tokens, 5);
    if (n < 3) return;
    int clientId = atoi(tokens[0]);
    std::string phone = tokens[1];
    std::string name  = tokens[2];
    std::string desc  = (n >= 4) ? tokens[3] : "";

    auto userOpt = authService_.findByPhone(phone);
    if (!userOpt.has_value()) return;

    authService_.registerUser(userOpt->userId, name, desc);
    sendUserAck(slot, userOpt->userId, false, userOpt->totalOrders, name);
    events_.onUserRegister(slot, userOpt->userId, name);
    sendSuggestions(slot, userOpt->userId);
    (void)clientId;
}

} // namespace app
