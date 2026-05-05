#include "socket_server.h"
#include "../shared/net.h"
#include "../shared/state.h"
#include "../shared/constants.h"
#include "../shared/protocol.h"
#include "../shared/utils.h"
#include "menu.h"
#include "user_store.h"
#include "order_store.h"
#include "transaction_store.h"
#include "lfm.h"
#include "phone_validator.h"
#include "session.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

static SOCKET listenSock = INVALID_SOCKET;
static SOCKET clientSock[MAX_CLIENTS];
static bool   clientActive[MAX_CLIENTS];
static char   clientRecvBuf[MAX_CLIENTS][2048];
static int    clientRecvLen[MAX_CLIENTS];
static int    clientUserId[MAX_CLIENTS];
static const SrvHooks* g_hooks = NULL;

void srvSetHooks(const SrvHooks* h) { g_hooks = h; }

static void slotReset(int slot) {
    if (clientSock[slot] != INVALID_SOCKET) {
        closesocket(clientSock[slot]);
    }
    clientSock[slot] = INVALID_SOCKET;
    clientActive[slot] = false;
    clientRecvLen[slot] = 0;
    clientUserId[slot] = -1;
}

bool srvStart(int port) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        clientSock[i] = INVALID_SOCKET;
        clientActive[i] = false;
        clientRecvLen[i] = 0;
        clientUserId[i] = -1;
    }

    listenSock = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSock == INVALID_SOCKET) return false;

    int opt = 1;
    setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)port);

    if (bind(listenSock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(listenSock); listenSock = INVALID_SOCKET; return false;
    }
    if (listen(listenSock, MAX_CLIENTS) == SOCKET_ERROR) {
        closesocket(listenSock); listenSock = INVALID_SOCKET; return false;
    }
    if (g_hooks && g_hooks->onReady) g_hooks->onReady(port);
    else printf("[Server] Listening on port %d\n", port);
    return true;
}

void srvStop() {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clientActive[i]) slotReset(i);
    }
    if (listenSock != INVALID_SOCKET) {
        closesocket(listenSock);
        listenSock = INVALID_SOCKET;
    }
}

int srvClientCount() {
    int n = 0;
    for (int i = 0; i < MAX_CLIENTS; i++) if (clientActive[i]) n++;
    return n;
}

static bool sendTo(int slot, MsgType type, const char* payload) {
    if (!clientActive[slot]) return false;
    char buf[2048];
    int n = buildMessage(type, payload, buf, sizeof(buf));
    return netSendAll(clientSock[slot], buf, n);
}

static void broadcast(MsgType type, const char* payload) {
    char buf[2048];
    int n = buildMessage(type, payload, buf, sizeof(buf));
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clientActive[i]) netSendAll(clientSock[i], buf, n);
    }
}

void srvBroadcastStart(const char* sessCode, const char* dt) {
    char payload[256];
    snprintf(payload, sizeof(payload), "%s|%s", sessCode, dt);
    broadcast(MSG_START, payload);
}

void srvBroadcastStop(const char* dt) {
    broadcast(MSG_STOP, dt);
}

void srvBroadcastMenu() {
    char payload[1536];
    serializeMenu(payload, sizeof(payload));
    broadcast(MSG_MENU_DATA, payload);
}

static void sendSuggest(int slot, int userId, const int* excluded, int exCount) {
    int topIdx[3]; float topScore[3];
    int n = lfmGetTopK(userId, excluded, exCount, topIdx, topScore, 3);
    char payload[512];
    payload[0] = '\0';
    int o = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) o += snprintf(payload + o, sizeof(payload) - o, "|");
        o += snprintf(payload + o, sizeof(payload) - o, "%s,%.3f",
                      menuCode[topIdx[i]], topScore[i]);
    }
    sendTo(slot, MSG_SUGGEST, payload);
}

// Sanitize name/desc ve ASCII printable (32-126), loai ky tu ngoai ASCII (BR16).
// Truncate toi maxLen-1 ky tu.
static void sanitizeAscii(const char* src, char* dst, int maxLen) {
    int o = 0;
    for (int i = 0; src && src[i] && o < maxLen - 1; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c >= 32 && c <= 126) dst[o++] = (char)c;
    }
    dst[o] = '\0';
}

static void handleUserLogin(int slot, const char* payload) {
    // Guard: khach khong the dang nhap truoc khi thu ngan mo ca.
    // Client o WAITING state nen khong the gui; day la defense in depth.
    if (!isSessionOpen()) {
        printf("[Server] REJECT USER_LOGIN slot=%d: session not open\n", slot);
        return;
    }
    char tokens[4][256];
    int n = splitByPipe(payload, tokens, 4);
    if (n < 2 || !isValidPhone(tokens[1])) {
        sendTo(slot, MSG_USER_ACK, "0|false|0|");
        sendSuggest(slot, 0, NULL, 0);
        return;
    }
    int userId = getOrCreateUser(tokens[1]);
    if (userId < 0) {
        sendTo(slot, MSG_USER_ACK, "0|false|0|");
        sendSuggest(slot, 0, NULL, 0);
        return;
    }
    clientUserId[slot] = userId;
    // isNew = chua co ten (khach moi), BAT KE so don.
    bool isNew = (userName[userId][0] == '\0');
    char ack[256];
    snprintf(ack, sizeof(ack), "%d|%s|%d|%s",
             userId, isNew ? "true" : "false",
             userTotalOrders[userId], userName[userId]);
    sendTo(slot, MSG_USER_ACK, ack);
    // Chi gui SUGGEST khi user khong phai new (new user phai dang ky truoc)
    if (!isNew) sendSuggest(slot, userId, NULL, 0);
    if (g_hooks && g_hooks->onUserLogin)
        g_hooks->onUserLogin(slot, tokens[1], userId, isNew, userTotalOrders[userId]);
    else printf("[Server] USER_LOGIN slot=%d phone=%s userId=%d (%s)\n",
                slot, tokens[1], userId, isNew ? "new" : "returning");
}

static void handleUserRegister(int slot, const char* payload) {
    // Guard: khach khong the dang ky truoc khi thu ngan mo ca.
    if (!isSessionOpen()) {
        printf("[Server] REJECT USER_REGISTER slot=%d: session not open\n", slot);
        return;
    }
    // Format: clientId|phone|name|desc
    char tokens[5][256];
    int n = splitByPipe(payload, tokens, 5);
    if (n < 3 || !isValidPhone(tokens[1])) {
        sendTo(slot, MSG_USER_ACK, "0|false|0|");
        return;
    }
    int userId = findUser(tokens[1]);
    if (userId < 0) userId = getOrCreateUser(tokens[1]);
    if (userId < 0) {
        sendTo(slot, MSG_USER_ACK, "0|false|0|");
        return;
    }
    char cleanName[NAME_LEN];
    char cleanDesc[DESC_LEN];
    sanitizeAscii(tokens[2], cleanName, NAME_LEN);
    sanitizeAscii(n >= 4 ? tokens[3] : "", cleanDesc, DESC_LEN);
    // Fallback: neu name rong sau sanitize -> dat "Khach"
    if (cleanName[0] == '\0') {
        strncpy(cleanName, "Khach", NAME_LEN - 1);
        cleanName[NAME_LEN - 1] = '\0';
    }
    setUserName(userId, cleanName, cleanDesc);
    saveUsers("data/users.dat");

    clientUserId[slot] = userId;
    // Gui lai USER_ACK voi isNew=false (da dang ky xong) + SUGGEST
    char ack[256];
    snprintf(ack, sizeof(ack), "%d|false|%d|%s",
             userId, userTotalOrders[userId], cleanName);
    sendTo(slot, MSG_USER_ACK, ack);
    sendSuggest(slot, userId, NULL, 0);

    if (g_hooks && g_hooks->onUserRegister)
        g_hooks->onUserRegister(slot, tokens[1], userId, cleanName);
    else printf("[Server] USER_REGISTER slot=%d phone=%s userId=%d name=%s\n",
                slot, tokens[1], userId, cleanName);
}

static void handleItemAdded(int slot, const char* payload) {
    if (!isSessionOpen()) return;
    char tokens[6][256];
    int n = splitByPipe(payload, tokens, 6);
    if (n < 3) return;
    int userId = clientUserId[slot];
    if (userId < 0) return;

    int excluded[MAX_ITEMS]; int exCount = 0;
    if (n >= 4) {
        char* p = tokens[3];
        while (*p && exCount < MAX_ITEMS) {
            char code[4] = {0}; int c = 0;
            while (*p && *p != ',' && c < 3) code[c++] = *p++;
            code[c] = '\0';
            int idx = findMenuIndex(code);
            if (idx >= 0) excluded[exCount++] = idx;
            if (*p == ',') p++;
        }
    }
    sendSuggest(slot, userId, excluded, exCount);
    const char* lastCode = (n >= 3) ? tokens[2] : "";
    if (g_hooks && g_hooks->onItemAdded)
        g_hooks->onItemAdded(slot, userId, lastCode, exCount);
    else printf("[Server] ITEM_ADDED slot=%d userId=%d exclude=%d\n", slot, userId, exCount);
}

static void handleOrderSubmit(int slot, const char* payload) {
    if (!isSessionOpen()) {
        sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        printf("[Server] REJECT ORDER_SUBMIT slot=%d: session not open\n", slot);
        return;
    }
    char tokens[16][256];
    int n = splitByPipe(payload, tokens, 16);
    if (n < 4) {
        sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        return;
    }
    int clientId = atoi(tokens[0]);
    int userId = atoi(tokens[1]);
    int itemsEnd = n - 2;  // last 2 fields = total, discount

    OrderInput in;
    memset(&in, 0, sizeof(in));
    in.userId = userId;
    in.clientId = clientId;
    if (userId >= 0 && userId < userCount) {
        strncpy(in.phone, userPhone[userId], 10);
        in.phone[10] = '\0';
    }

    in.itemCount = 0;
    for (int t = 2; t < itemsEnd && in.itemCount < MAX_ITEMS; t++) {
        char code[8] = {0}; int qty = 0;
        if (sscanf(tokens[t], "%7[^,],%d", code, &qty) == 2) {
            int idx = findMenuIndex(code);
            if (idx >= 0) {
                in.itemIdx[in.itemCount] = idx;
                in.qty[in.itemCount] = qty;
                in.itemCount++;
            }
        }
    }
    if (in.itemCount == 0) {
        sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        return;
    }

    int oid = createOrder(&in);
    if (oid < 0) {
        sendTo(slot, MSG_ORDER_ACK, "0|FAIL");
        return;
    }
    lfmOnlineUpdate(userId, in.itemIdx, in.qty, in.itemCount);

    // Persist ngay sau moi order submit de dashboard / restart doc duoc:
    //   - transactions.dat: chua txn moi
    //   - users.dat: userTotalOrders[userId] vua tang
    saveTransactions("data/transactions.dat");
    saveUsers("data/users.dat");

    char ack[64];
    snprintf(ack, sizeof(ack), "%d|OK", oid);
    sendTo(slot, MSG_ORDER_ACK, ack);
    if (g_hooks && g_hooks->onOrderSubmitted)
        g_hooks->onOrderSubmitted(slot, userId, oid, in.itemCount,
                                  orderTotal[oid - 1], orderDiscount[oid - 1]);
    else printf("[Server] ORDER_SUBMIT slot=%d userId=%d oid=%d items=%d total=%.0f\n",
                slot, userId, oid, in.itemCount, orderTotal[oid - 1]);
}

static void onLine(int slot, const char* line, void* ctx) {
    (void)ctx;
    ParsedMsg pm;
    if (!parseMessage(line, &pm)) {
        printf("[Server] Unknown from slot %d: '%s'\n", slot, line);
        return;
    }
    switch (pm.type) {
        case MSG_USER_LOGIN:    handleUserLogin(slot, pm.payload); break;
        case MSG_USER_REGISTER: handleUserRegister(slot, pm.payload); break;
        case MSG_ITEM_ADDED:    handleItemAdded(slot, pm.payload); break;
        case MSG_ORDER_SUBMIT:  handleOrderSubmit(slot, pm.payload); break;
        case MSG_HEARTBEAT:
            if (g_hooks && g_hooks->onHeartbeat) g_hooks->onHeartbeat(slot);
            break;
        default:
            printf("[Server] Unhandled %s from slot %d\n", msgTypeName(pm.type), slot);
    }
}

void srvPoll(int timeoutMs) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listenSock, &rfds);
    SOCKET maxSock = listenSock;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clientActive[i]) {
            FD_SET(clientSock[i], &rfds);
            if (clientSock[i] > maxSock) maxSock = clientSock[i];
        }
    }
    timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    int n = select((int)maxSock + 1, &rfds, NULL, NULL, &tv);
    if (n <= 0) return;

    if (FD_ISSET(listenSock, &rfds)) {
        sockaddr_in peer; socklen_t plen = sizeof(peer);
        SOCKET cs = accept(listenSock, (sockaddr*)&peer, &plen);
        if (cs != INVALID_SOCKET) {
            int slot = -1;
            for (int i = 0; i < MAX_CLIENTS; i++) {
                if (!clientActive[i]) { slot = i; break; }
            }
            if (slot < 0) {
                closesocket(cs);
                if (!g_hooks) printf("[Server] No free slot, rejected\n");
            } else {
                clientSock[slot] = cs;
                clientActive[slot] = true;
                clientRecvLen[slot] = 0;
                clientUserId[slot] = -1;
                if (g_hooks && g_hooks->onClientJoined) g_hooks->onClientJoined(slot);
                else printf("[Server] Accepted slot=%d\n", slot);
                // Neu ca dang mo, gui START + MENU_DATA cho client vua ket noi
                if (isSessionOpen()) {
                    char sp[256];
                    snprintf(sp, sizeof(sp), "%s|%s", sessionCode, sessionStart);
                    sendTo(slot, MSG_START, sp);
                    char menuPayload[1536];
                    serializeMenu(menuPayload, sizeof(menuPayload));
                    sendTo(slot, MSG_MENU_DATA, menuPayload);
                }
            }
        }
    }

    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (!clientActive[i]) continue;
        if (!FD_ISSET(clientSock[i], &rfds)) continue;
        int space = 2048 - clientRecvLen[i];
        if (space <= 0) { slotReset(i); continue; }
        int got = recv(clientSock[i], clientRecvBuf[i] + clientRecvLen[i], space, 0);
        if (got <= 0) {
            if (g_hooks && g_hooks->onClientLeft) g_hooks->onClientLeft(i);
            else printf("[Server] Slot %d disconnected\n", i);
            slotReset(i);
            continue;
        }
        clientRecvLen[i] += got;
        netDrainBuffer(clientRecvBuf[i], &clientRecvLen[i], i, onLine, NULL);
    }
}
