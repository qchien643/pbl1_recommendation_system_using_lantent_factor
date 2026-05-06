#include "order_controller.h"
#include "../../shared/utils.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace app {

OrderController::OrderController(OrderService& orderService,
                                  MenuService& menuService,
                                  LfmService& lfmService,
                                  SessionService& sessionService,
                                  TcpServer& tcpServer,
                                  IServerEventListener& events)
    : orderService_(orderService), menuService_(menuService),
      lfmService_(lfmService), sessionService_(sessionService),
      tcpServer_(tcpServer), events_(events) {}

void OrderController::handleItemAdded(int slot, const std::string& payload) {
    if (!sessionService_.isOpen()) return;

    char tokens[8][256];
    int n = splitByPipe(payload.c_str(), tokens, 8);
    if (n < 4) return;
    int     clientId  = atoi(tokens[0]);
    int64_t userId    = atoll(tokens[1]);
    std::string newCode    = tokens[2];
    std::string currentCSV = tokens[3];

    // Parse currentCodes CSV → excluded menuIdx
    std::vector<int64_t> excluded;
    char cur[256]; std::strncpy(cur, currentCSV.c_str(), sizeof(cur) - 1); cur[sizeof(cur)-1]=0;
    char* tok = std::strtok(cur, ",");
    while (tok) {
        int64_t mi = menuService_.findIndexByCode(tok);
        if (mi >= 0) excluded.push_back(mi);
        tok = std::strtok(nullptr, ",");
    }

    auto top = lfmService_.topK(userId, excluded, 3);

    std::string p;
    std::vector<SuggestionItem> items;
    for (size_t i = 0; i < top.size(); i++) {
        auto m = menuService_.findByIndex(top[i].first);
        if (!m.has_value()) continue;
        if (i) p += "|";
        char buf[64]; snprintf(buf, sizeof(buf), "%s,%.3f", m->code.c_str(), top[i].second);
        p += buf;
        items.push_back({ m->code, top[i].second });
    }
    tcpServer_.sendTo(slot, MSG_SUGGEST, p);
    events_.onItemAdded(slot, userId, newCode, (int)excluded.size());
    events_.onSuggestSent(slot, userId, items);
    (void)clientId;
}

void OrderController::handleOrderSubmit(int slot, const std::string& payload) {
    if (!sessionService_.isOpen()) {
        tcpServer_.sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        events_.onOrderRejected(slot, "session not open");
        return;
    }

    char tokens[16][256];
    int n = splitByPipe(payload.c_str(), tokens, 16);
    if (n < 4) {
        tcpServer_.sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        events_.onOrderRejected(slot, "bad payload");
        return;
    }
    int     clientId = atoi(tokens[0]);
    int64_t userId   = atoll(tokens[1]);
    int     itemsEnd = n - 2;   // last 2 = total, discount

    OrderService::CreateRequest req;
    req.userId      = userId;
    req.sessionCode = sessionService_.currentCode();

    for (int t = 2; t < itemsEnd; t++) {
        char code[8] = {0}; int qty = 0;
        if (sscanf(tokens[t], "%7[^,],%d", code, &qty) != 2) continue;
        if (qty <= 0) continue;
        req.items.push_back({ code, qty });
    }

    auto res = orderService_.create(req);
    if (!res.success) {
        tcpServer_.sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        events_.onOrderRejected(slot, res.failReason);
        return;
    }

    char ack[64]; snprintf(ack, sizeof(ack), "%lld|OK", (long long)res.txnId + 1);
    tcpServer_.sendTo(slot, MSG_ORDER_ACK, ack);
    events_.onOrderSubmitted(slot, userId, res.txnId + 1, (int)req.items.size(),
                              res.total, res.discount);
    (void)clientId;
}

} // namespace app
