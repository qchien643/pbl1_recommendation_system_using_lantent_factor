// main_client.cpp — Entry --client <IP> [clientId] [--json]
// Default: text mode state machine (Phase 2)
// --json: IPC mode cho React Ink UI (Phase 3), emit JSON events, read JSON cmds

#include "socket_client.h"
#include "order_builder.h"
#include "input_handler.h"
#include "display.h"
#include "../shared/net.h"
#include "../shared/protocol.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include "../shared/constants.h"
#include "../shared/json.h"
#include "../shared/db/database.h"
#include "../shared/db/db_schema.h"
#include "../server/menu.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <ctime>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <string>
#include <chrono>

// Parse MENU_DATA payload va populate menu Table (source of truth) + parallel arrays cache.
// Server đã đăng ký schema; phía client cũng phải initSchema để có menu table với HashIndex.
static void applyMenuData(const char* payload) {
    db::initRestaurantSchema();
    db::Table& menuT = db::Database::instance().table(db::tbl::MENU);
    menuT.clear();

    menuCount = 0;
    const char* p = payload;
    while (*p && menuCount < MAX_MENU) {
        char code[8] = {0};
        char name[64] = {0};
        float price = 0;
        int n = sscanf(p, "%7[^,],%63[^,],%f", code, name, &price);
        if (n == 3) {
            int clen = (int)strlen(code); if (clen > 3) clen = 3;
            memcpy(menuCode[menuCount], code, clen);
            menuCode[menuCount][clen] = '\0';
            strncpy(menuName[menuCount], name, 49);
            menuName[menuCount][49] = '\0';
            menuPrice[menuCount] = price;

            // Insert vào menu table — findMenuIndex(code) sẽ dùng HashIndex
            db::Row row(&menuT.schema());
            row.set(db::col::CODE,     std::string(menuCode[menuCount]));
            row.set(db::col::NAME,     std::string(menuName[menuCount]));
            row.set(db::col::PRICE,    (double)price);
            row.set(db::col::CATEGORY, std::string(""));
            try { menuT.insert(std::move(row)); } catch (...) {}

            menuCount++;
        }
        const char* nxt = strchr(p, '|');
        if (!nxt) break;
        p = nxt + 1;
    }
}

// ============================================================================
// TEXT MODE (Phase 2) — giu nguyen logic cu
// ============================================================================

static int runTextLoop(const char* serverIp, int clientId) {
    printf("[Client %d] Connecting to %s:%d...\n", clientId, serverIp, DEFAULT_PORT);
    if (!cliConnect(serverIp, DEFAULT_PORT)) {
        printf("[Client] Cannot connect\n");
        return 1;
    }
    printf("[Client %d] Connected\n", clientId);

    int type;
    char payload[2048];
    char sessCode[16] = {0};
    bool sessionStarted = false;

    while (!sessionStarted || menuCount == 0) {
        if (!cliRecvMessage(&type, payload, sizeof(payload))) {
            printf("[Client] Lost connection during handshake\n");
            cliDisconnect(); return 1;
        }
        if (type == MSG_START) {
            char tokens[4][256];
            splitByPipe(payload, tokens, 4);
            strncpy(sessCode, tokens[0], 15);
            printf("[Client %d] Session STARTED code=%s at %s\n",
                   clientId, tokens[0], tokens[1]);
            sessionStarted = true;
        } else if (type == MSG_MENU_DATA) {
            applyMenuData(payload);
            printf("[Client %d] Loaded %d menu items\n", clientId, menuCount);
        } else if (type == MSG_STOP) {
            printf("[Client] Session closed before starting\n");
            cliDisconnect(); return 0;
        }
    }

    bool keepRunning = true;
    while (keepRunning) {
        printf("\n==================================================\n");
        printf("   NHA HANG VIET PHONG - BAN %02d\n", clientId);
        printf("==================================================\n");
        printf("  Vui long nhap so dien thoai (10 chu so):\n> ");
        char phone[16];
        if (!readPhone(phone)) { printf("\n[Client] No more input. Bye.\n"); break; }

        char loginPayload[64];
        snprintf(loginPayload, sizeof(loginPayload), "%d|%s", clientId, phone);
        if (!cliSend(MSG_USER_LOGIN, loginPayload)) break;

        int userId = -1;
        bool gotSuggest = false;
        while (!gotSuggest) {
            if (!cliRecvMessage(&type, payload, sizeof(payload))) {
                printf("[Client] Lost connection\n");
                keepRunning = false; break;
            }
            if (type == MSG_USER_ACK) {
                char tokens[4][256];
                splitByPipe(payload, tokens, 4);
                userId = atoi(tokens[0]);
                showUserAck(payload);
            } else if (type == MSG_SUGGEST) {
                showMenu();
                showSuggestions(payload);
                gotSuggest = true;
            } else if (type == MSG_STOP) {
                printf("\n[Client] Ca ket thuc\n");
                keepRunning = false; break;
            }
        }
        if (!keepRunning) break;
        if (userId < 0) continue;

        ClientOrder order;
        orderInit(&order);
        while (order.count < MAX_ITEMS) {
            showOrderStatus(&order);
            printf("  Nhap MA MON va SO LUONG (VD: P01 2) - 00/Enter = Xong\n> ");
            char code[8] = {0};
            int qty = 0;
            if (!readItemAndQty(code, &qty)) break;
            if (!orderAddItem(&order, code, qty)) {
                printf("  Ma mon '%s' khong ton tai\n", code);
                continue;
            }
            char curCodes[128];
            orderCurrentCodes(&order, curCodes, sizeof(curCodes));
            char ip2[256];
            snprintf(ip2, sizeof(ip2), "%d|%d|%s|%s", clientId, userId, code, curCodes);
            cliSend(MSG_ITEM_ADDED, ip2);
            if (cliRecvMessage(&type, payload, sizeof(payload))) {
                if (type == MSG_SUGGEST) showSuggestions(payload);
                else if (type == MSG_STOP) { keepRunning = false; break; }
            }
        }
        if (!keepRunning) break;
        if (order.count == 0) { printf("  Don trong, bo qua.\n"); continue; }

        orderFinalize(&order);
        showInvoice(&order, clientId, phone, sessCode);
        printf("\n  Xac nhan gui len Server? (Y=Co / N=Huy) [Y]:\n> ");
        if (readYesNo()) {
            char submit[512];
            orderSerializeForSubmit(&order, clientId, userId, submit, sizeof(submit));
            cliSend(MSG_ORDER_SUBMIT, submit);
            if (cliRecvMessage(&type, payload, sizeof(payload))) {
                if (type == MSG_ORDER_ACK) {
                    printf("  Server ACK: %s. Cam on!\n", payload);
                } else if (type == MSG_STOP) {
                    printf("\n[Client] Ca ket thuc\n");
                    break;
                }
            }
        } else {
            printf("  Da huy don.\n");
        }
    }
    cliDisconnect();
    return 0;
}

// ============================================================================
// JSON MODE (Phase 3) — stdio IPC voi React Ink parent
// ============================================================================

static ClientOrder          jsonOrder;
static int                  jsonClientId  = 1;
static int                  jsonUserId    = -1;
static std::atomic<bool>    jsonRunning{true};

static std::mutex           jsonCmdMutex;
static std::queue<std::string> jsonCmdQueue;
static std::atomic<bool>    jsonStdinEof{false};

static void jsonEmit(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    fflush(stdout);
}

static void jsonStdinThread() {
    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        std::lock_guard<std::mutex> lg(jsonCmdMutex);
        jsonCmdQueue.push(std::string(line));
    }
    jsonStdinEof = true;
}

static void emitMenuEvent(const char* payload) {
    // "P01,Pho Bo Tai,65000|B01,..." -> JSON array
    printf("{\"event\":\"menu\",\"items\":[");
    const char* p = payload;
    bool first = true;
    while (*p) {
        char code[8] = {0}, name[64] = {0};
        float price = 0;
        if (sscanf(p, "%7[^,],%63[^,],%f", code, name, &price) == 3) {
            char esc[96];
            jsonEscape(name, esc, sizeof(esc));
            if (!first) printf(",");
            printf("{\"code\":\"%s\",\"name\":\"%s\",\"price\":%.0f}", code, esc, price);
            first = false;
        }
        const char* nxt = strchr(p, '|');
        if (!nxt) break;
        p = nxt + 1;
    }
    printf("]}\n");
    fflush(stdout);
}

static void emitSuggestEvent(const char* payload) {
    printf("{\"event\":\"suggest\",\"items\":[");
    const char* p = payload;
    bool first = true;
    while (*p) {
        char code[8] = {0};
        float score = 0;
        if (sscanf(p, "%7[^,],%f", code, &score) == 2) {
            int idx = findMenuIndex(code);
            const char* name = (idx >= 0) ? menuName[idx] : "";
            char esc[96];
            jsonEscape(name, esc, sizeof(esc));
            if (!first) printf(",");
            printf("{\"code\":\"%s\",\"name\":\"%s\",\"score\":%.3f}", code, esc, score);
            first = false;
        }
        const char* nxt = strchr(p, '|');
        if (!nxt) break;
        p = nxt + 1;
    }
    printf("]}\n");
    fflush(stdout);
}

static void emitInvoiceReady() {
    printf("{\"event\":\"invoice_ready\",\"items\":[");
    for (int i = 0; i < jsonOrder.count; i++) {
        if (i > 0) printf(",");
        char esc[96];
        jsonEscape(jsonOrder.names[i], esc, sizeof(esc));
        printf("{\"code\":\"%s\",\"name\":\"%s\",\"qty\":%d,\"price\":%.0f,\"subtotal\":%.0f}",
               jsonOrder.codes[i], esc, jsonOrder.qtys[i], jsonOrder.prices[i],
               jsonOrder.prices[i] * jsonOrder.qtys[i]);
    }
    printf("],\"subtotal\":%.0f,\"discount\":%.0f,\"total\":%.0f}\n",
           jsonOrder.subtotal, jsonOrder.discount, jsonOrder.total);
    fflush(stdout);
}

static void handleServerLine(int slot, const char* line, void* ctx) {
    (void)slot; (void)ctx;
    ParsedMsg pm;
    if (!parseMessage(line, &pm)) return;
    char tokens[16][256];
    int n = splitByPipe(pm.payload, tokens, 16);

    switch (pm.type) {
        case MSG_START: {
            char esc1[32], esc2[32];
            jsonEscape(n > 0 ? tokens[0] : "", esc1, sizeof(esc1));
            jsonEscape(n > 1 ? tokens[1] : "", esc2, sizeof(esc2));
            jsonEmit("{\"event\":\"session_start\",\"code\":\"%s\",\"dateTime\":\"%s\"}",
                     esc1, esc2);
            break;
        }
        case MSG_MENU_DATA:
            applyMenuData(pm.payload);
            emitMenuEvent(pm.payload);
            break;
        case MSG_USER_ACK: {
            int uid = (n > 0) ? atoi(tokens[0]) : 0;
            const char* isNew = (n > 1) ? tokens[1] : "false";
            int cnt = (n > 2) ? atoi(tokens[2]) : 0;
            const char* name = (n > 3) ? tokens[3] : "";
            jsonUserId = uid;
            char nameEsc[96]; jsonEscape(name, nameEsc, sizeof(nameEsc));
            jsonEmit("{\"event\":\"user_ack\",\"userId\":%d,\"isNew\":%s,"
                     "\"orderCount\":%d,\"name\":\"%s\"}",
                     uid, isNew, cnt, nameEsc);
            break;
        }
        case MSG_SUGGEST:
            emitSuggestEvent(pm.payload);
            break;
        case MSG_ORDER_ACK: {
            int oid = (n > 0) ? atoi(tokens[0]) : 0;
            const char* status = (n > 1) ? tokens[1] : "OK";
            char esc[32]; jsonEscape(status, esc, sizeof(esc));
            jsonEmit("{\"event\":\"order_ack\",\"orderId\":%d,\"status\":\"%s\"}", oid, esc);
            orderInit(&jsonOrder);
            break;
        }
        case MSG_STOP: {
            char esc[32];
            jsonEscape(n > 0 ? tokens[0] : "", esc, sizeof(esc));
            jsonEmit("{\"event\":\"session_stop\",\"dateTime\":\"%s\"}", esc);
            jsonRunning = false;
            break;
        }
        default: break;
    }
}

static void handleJsonCmd(const char* cmdLine) {
    char cmd[32] = {0};
    if (!jsonGetString(cmdLine, "cmd", cmd, sizeof(cmd))) return;

    if (strcmp(cmd, "login") == 0) {
        char phone[16] = {0};
        if (!jsonGetString(cmdLine, "phone", phone, sizeof(phone))) return;
        orderInit(&jsonOrder);
        char p[64];
        snprintf(p, sizeof(p), "%d|%s", jsonClientId, phone);
        cliSend(MSG_USER_LOGIN, p);
    }
    else if (strcmp(cmd, "register") == 0) {
        char phone[16] = {0};
        char name[64] = {0};
        char desc[128] = {0};
        if (!jsonGetString(cmdLine, "phone", phone, sizeof(phone))) return;
        jsonGetString(cmdLine, "name", name, sizeof(name));
        jsonGetString(cmdLine, "desc", desc, sizeof(desc));
        char p[256];
        snprintf(p, sizeof(p), "%d|%s|%s|%s", jsonClientId, phone, name, desc);
        cliSend(MSG_USER_REGISTER, p);
    }
    else if (strcmp(cmd, "add_item") == 0) {
        char code[8] = {0};
        int qty = 1;
        if (!jsonGetString(cmdLine, "code", code, sizeof(code))) return;
        jsonGetInt(cmdLine, "qty", &qty);
        if (orderAddItem(&jsonOrder, code, qty)) {
            char curCodes[128];
            orderCurrentCodes(&jsonOrder, curCodes, sizeof(curCodes));
            char p[256];
            snprintf(p, sizeof(p), "%d|%d|%s|%s",
                     jsonClientId, jsonUserId, code, curCodes);
            cliSend(MSG_ITEM_ADDED, p);
        } else {
            jsonEmit("{\"event\":\"error\",\"message\":\"invalid_code_or_full\"}");
        }
    }
    else if (strcmp(cmd, "finish") == 0) {
        orderFinalize(&jsonOrder);
        emitInvoiceReady();
    }
    else if (strcmp(cmd, "confirm") == 0) {
        char submit[512];
        orderSerializeForSubmit(&jsonOrder, jsonClientId, jsonUserId, submit, sizeof(submit));
        cliSend(MSG_ORDER_SUBMIT, submit);
    }
    else if (strcmp(cmd, "cancel") == 0) {
        orderInit(&jsonOrder);
        jsonEmit("{\"event\":\"cancelled\"}");
    }
    else if (strcmp(cmd, "quit") == 0) {
        jsonRunning = false;
    }
}

static int runJsonLoop(const char* serverIp, int clientId) {
    jsonClientId = clientId;
    orderInit(&jsonOrder);
    setbuf(stdout, NULL);

    if (!cliConnect(serverIp, DEFAULT_PORT)) {
        jsonEmit("{\"event\":\"error\",\"message\":\"cannot_connect\"}");
        return 1;
    }
    jsonEmit("{\"event\":\"connected\"}");

    std::thread t(jsonStdinThread);
    t.detach();

    char recvBuf[4096];
    int recvLen = 0;
    auto lastHeartbeat = std::chrono::steady_clock::now();
    const int HEARTBEAT_INTERVAL_MS = 5000;

    while (jsonRunning) {
        // Gui HEARTBEAT dinh ky (§BR mo ta moi 5s)
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastHeartbeat).count();
        if (elapsed >= HEARTBEAT_INTERVAL_MS) {
            char payload[48];
            snprintf(payload, sizeof(payload), "%d|%lld", jsonClientId, (long long)time(NULL));
            cliSend(MSG_HEARTBEAT, payload);
            lastHeartbeat = now;
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        SOCKET sk = cliGetSocket();
        if (sk == INVALID_SOCKET) break;
        FD_SET(sk, &rfds);
        timeval tv;
        tv.tv_sec = 0; tv.tv_usec = 50000;
        int n = select((int)sk + 1, &rfds, nullptr, nullptr, &tv);

        if (n > 0 && FD_ISSET(sk, &rfds)) {
            int space = (int)sizeof(recvBuf) - recvLen;
            if (space <= 0) { jsonRunning = false; break; }
            int got = cliRecvOnce(recvBuf + recvLen, space);
            if (got <= 0) {
                jsonEmit("{\"event\":\"disconnected\"}");
                jsonRunning = false;
                break;
            }
            recvLen += got;
            netDrainBuffer(recvBuf, &recvLen, 0, handleServerLine, nullptr);
        }

        std::string cmdLine;
        {
            std::lock_guard<std::mutex> lg(jsonCmdMutex);
            if (!jsonCmdQueue.empty()) {
                cmdLine = jsonCmdQueue.front();
                jsonCmdQueue.pop();
            }
        }
        if (!cmdLine.empty()) {
            handleJsonCmd(cmdLine.c_str());
        } else if (jsonStdinEof) {
            // Node parent dong stdin → client thoat
            jsonRunning = false;
        }
    }
    cliDisconnect();
    return 0;
}

// ============================================================================
// ENTRY
// ============================================================================

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    const char* serverIp = "127.0.0.1";
    int clientId = 1;
    bool jsonMode = false;

    if (argc < 2 || strcmp(argv[1], "--client") != 0) {
        fprintf(stderr, "Usage: %s --client <IP> [clientId] [--json]\n", argv[0]);
        return 1;
    }
    if (argc >= 3) serverIp = argv[2];
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = true;
        else {
            int v = atoi(argv[i]);
            if (v > 0) clientId = v;
        }
    }

    if (!netInit()) {
        fprintf(stderr, "netInit failed\n");
        return 1;
    }

    int rc;
    if (jsonMode) rc = runJsonLoop(serverIp, clientId);
    else          rc = runTextLoop(serverIp, clientId);

    netCleanup();
    return rc;
}
