// main_server.cpp — Entry --server [--json]
// Text mode: printf logs + stdin nhap ma so ca
// JSON mode: emit JSON events tren stdout, doc JSON cmds tu stdin (cho React Ink Server UI)

#include "socket_server.h"
#include "session.h"
#include "menu.h"
#include "user_store.h"
#include "transaction_store.h"
#include "lfm.h"
#include "../shared/net.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include "../shared/constants.h"
#include "../shared/json.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <string>

// ============================================================================
// Stdin queue (dung chung cho ca 2 mode)
// ============================================================================
static std::mutex cmdMutex;
static std::queue<std::string> cmdQueue;
static std::atomic<bool> stdinEof{false};

static void stdinReader() {
    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;
        std::lock_guard<std::mutex> lg(cmdMutex);
        cmdQueue.push(std::string(line));
    }
    stdinEof = true;
}

static bool popCmd(std::string& out) {
    std::lock_guard<std::mutex> lg(cmdMutex);
    if (cmdQueue.empty()) return false;
    out = cmdQueue.front();
    cmdQueue.pop();
    return true;
}

// ============================================================================
// JSON MODE event emitters (set lam hooks cho socket_server)
// ============================================================================
static void jsonEmit(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    fflush(stdout);
}

static int  g_totalOrdersToday = 0;
static float g_revenueToday = 0;
static float g_discountToday = 0;
static int   g_discountedOrders = 0;

static void emitStats() {
    jsonEmit("{\"event\":\"stats\",\"totalOrdersToday\":%d,\"revenueToday\":%.0f,"
             "\"discountToday\":%.0f,\"discountedOrders\":%d,"
             "\"clientsConnected\":%d,\"usersKnown\":%d}",
             g_totalOrdersToday, g_revenueToday, g_discountToday,
             g_discountedOrders, srvClientCount(), userCount);
}

static void jsonOnReady(int port) {
    jsonEmit("{\"event\":\"ready\",\"port\":%d,\"menuItems\":%d,\"savedUsers\":%d}",
             port, menuCount, userCount);
}

static void jsonOnClientJoined(int slot) {
    jsonEmit("{\"event\":\"client_joined\",\"slot\":%d}", slot);
    emitStats();
}

static void jsonOnClientLeft(int slot) {
    jsonEmit("{\"event\":\"client_left\",\"slot\":%d}", slot);
    emitStats();
}

static void jsonOnUserLogin(int slot, const char* phone, int userId,
                             bool isNew, int orderCount) {
    char esc[32]; jsonEscape(phone, esc, sizeof(esc));
    jsonEmit("{\"event\":\"user_login\",\"slot\":%d,\"phone\":\"%s\","
             "\"userId\":%d,\"isNew\":%s,\"orderCount\":%d}",
             slot, esc, userId, isNew ? "true" : "false", orderCount);
}

static void jsonOnItemAdded(int slot, int userId, const char* code, int exCount) {
    char esc[16]; jsonEscape(code, esc, sizeof(esc));
    jsonEmit("{\"event\":\"item_added\",\"slot\":%d,\"userId\":%d,"
             "\"code\":\"%s\",\"excluded\":%d}",
             slot, userId, esc, exCount);
}

static void jsonOnOrderSubmitted(int slot, int userId, int orderId,
                                  int itemCount, float total, float discount) {
    jsonEmit("{\"event\":\"order_submitted\",\"slot\":%d,\"userId\":%d,"
             "\"orderId\":%d,\"itemCount\":%d,\"total\":%.0f,\"discount\":%.0f}",
             slot, userId, orderId, itemCount, total, discount);
    g_totalOrdersToday++;
    g_revenueToday += total;
    if (discount > 0) { g_discountToday += discount; g_discountedOrders++; }
    emitStats();
}

static void jsonOnHeartbeat(int slot) {
    jsonEmit("{\"event\":\"heartbeat\",\"slot\":%d}", slot);
}

static void jsonOnUserRegister(int slot, const char* phone, int userId, const char* name) {
    char pesc[32]; jsonEscape(phone, pesc, sizeof(pesc));
    char nesc[96]; jsonEscape(name, nesc, sizeof(nesc));
    jsonEmit("{\"event\":\"user_register\",\"slot\":%d,\"phone\":\"%s\","
             "\"userId\":%d,\"name\":\"%s\"}",
             slot, pesc, userId, nesc);
}

static const SrvHooks JSON_HOOKS = {
    jsonOnReady, jsonOnClientJoined, jsonOnClientLeft,
    jsonOnUserLogin, jsonOnItemAdded, jsonOnOrderSubmitted, jsonOnHeartbeat,
    jsonOnUserRegister
};

// ============================================================================
// TEXT MODE (Phase 2 — chay default neu khong co --json)
// ============================================================================
static int runTextMode() {
    printf("[Server] Ready. Nhap MA SO (1-9 chu so) de MO CA:\n");
    std::thread t(stdinReader); t.detach();

    bool sessionFlag = false;
    while (true) {
        srvPoll(100);
        std::string cmd;
        if (!popCmd(cmd)) {
            if (stdinEof && !sessionFlag) break;
            continue;
        }
        if (!sessionFlag) {
            if (openSession(cmd.c_str())) {
                sessionFlag = true;
                srvBroadcastStart(sessionCode, sessionStart);
                srvBroadcastMenu();
                printf("[Server] Session OPENED code=%s start=%s\n",
                       sessionCode, sessionStart);
            } else {
                printf("[Server] Ma so khong hop le (1-9 ky tu)\n");
            }
        } else {
            if (matchSessionCode(cmd.c_str())) {
                char dt[20]; currentTimestamp(dt, sizeof(dt));
                srvBroadcastStop(dt);
                closeSession(cmd.c_str(), "data");
                printf("[Server] Session CLOSED. Report + models saved.\n");
                break;
            } else {
                printf("[Server] Ma so khong khop. Nhap lai:\n");
            }
        }
    }
    return 0;
}

// ============================================================================
// JSON MODE — React Ink Server UI
// ============================================================================
static int runJsonMode() {
    // srvStart da goi onReady truoc do (hooks da set trong main)
    std::thread t(stdinReader); t.detach();
    bool sessionFlag = false;

    while (true) {
        srvPoll(100);
        std::string cmdLine;
        if (!popCmd(cmdLine)) {
            if (stdinEof) break;
            continue;
        }

        char cmd[32] = {0};
        if (!jsonGetString(cmdLine.c_str(), "cmd", cmd, sizeof(cmd))) continue;

        if (strcmp(cmd, "open_session") == 0) {
            char code[16] = {0};
            if (!jsonGetString(cmdLine.c_str(), "code", code, sizeof(code))) continue;
            if (openSession(code)) {
                sessionFlag = true;
                srvBroadcastStart(sessionCode, sessionStart);
                srvBroadcastMenu();
                char esc1[16], esc2[32];
                jsonEscape(sessionCode, esc1, sizeof(esc1));
                jsonEscape(sessionStart, esc2, sizeof(esc2));
                jsonEmit("{\"event\":\"session_opened\",\"code\":\"%s\",\"dateTime\":\"%s\"}",
                         esc1, esc2);
                emitStats();
            } else {
                jsonEmit("{\"event\":\"error\",\"message\":\"invalid_session_code\"}");
            }
        }
        else if (strcmp(cmd, "close_session") == 0) {
            char code[16] = {0};
            if (!jsonGetString(cmdLine.c_str(), "code", code, sizeof(code))) continue;
            if (matchSessionCode(code)) {
                char dt[20]; currentTimestamp(dt, sizeof(dt));
                srvBroadcastStop(dt);
                closeSession(code, "data");
                char esc[32]; jsonEscape(dt, esc, sizeof(esc));
                jsonEmit("{\"event\":\"session_closed\",\"dateTime\":\"%s\","
                         "\"totalOrders\":%d,\"totalRevenue\":%.0f,"
                         "\"totalDiscount\":%.0f,\"discountedOrders\":%d,\"uniqueUsers\":%d}",
                         esc, g_totalOrdersToday, g_revenueToday,
                         g_discountToday, g_discountedOrders, userCount);
                sessionFlag = false;
                break;
            } else {
                jsonEmit("{\"event\":\"error\",\"message\":\"code_mismatch\"}");
            }
        }
        else if (strcmp(cmd, "get_stats") == 0) {
            emitStats();
        }
        else if (strcmp(cmd, "quit") == 0) {
            break;
        }
    }
    return 0;
}

// ============================================================================
// ENTRY
// ============================================================================
int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "--server";
    if (strcmp(mode, "--server") != 0) {
        fprintf(stderr, "Usage: %s --server [--json]\n", argv[0]);
        return 1;
    }
    bool jsonMode = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = true;
    }

    setbuf(stdout, NULL);

    if (!netInit()) { fprintf(stderr, "netInit failed\n"); return 1; }

    if (!loadMenu("data/menu.txt")) {
        fprintf(stderr, "[Server] Cannot load data/menu.txt\n");
        netCleanup(); return 1;
    }

    if (loadUsers("data/users.dat") && !jsonMode) {
        printf("[Server] Loaded %d existing users\n", userCount);
    }

    if (loadTransactions("data/transactions.dat")) {
        rebuildOrderHistory();
        if (!jsonMode) printf("[Server] Loaded %d transactions, orderHistory rebuilt\n", txnCount);
    } else {
        if (!jsonMode) printf("[Server] No transactions.dat found (fresh start)\n");
    }

    lfmInit((unsigned int)time(NULL));
    if (lfmLoadModels("data/lfm_P.dat", "data/lfm_Q.dat")) {
        if (!jsonMode) printf("[Server] Loaded LFM model from disk\n");
    } else {
        if (!jsonMode) printf("[Server] Using fresh random init\n");
    }

    if (jsonMode) srvSetHooks(&JSON_HOOKS);

    if (!srvStart(DEFAULT_PORT)) {
        fprintf(stderr, "[Server] Failed to start listener on port %d\n", DEFAULT_PORT);
        netCleanup(); return 1;
    }

    int rc = jsonMode ? runJsonMode() : runTextMode();

    srvStop();
    netCleanup();
    return rc;
}
