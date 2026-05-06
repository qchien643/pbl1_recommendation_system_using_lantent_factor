// main_server.cpp — Entry --server [--json]
//
// Sau Spring-style refactor: main chỉ bootstrap + event loop. Mọi business logic
// nằm trong app::ApplicationContext (DI container).

#include "../shared/net.h"
#include "../shared/db/database.h"
#include "../shared/db/db_schema.h"
#include "../shared/json.h"
#include "../shared/utils.h"
#include "../shared/constants.h"
#include "infra/application_context.h"
#include "infra/text_log_listener.h"
#include "infra/json_event_listener.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <queue>
#include <string>
#include <memory>

// ---- Stdin queue (text+json mode dùng chung) ----
static std::mutex                cmdMutex;
static std::queue<std::string>   cmdQueue;
static std::atomic<bool>         stdinEof{false};

static void stdinReader() {
    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        std::lock_guard<std::mutex> lg(cmdMutex);
        cmdQueue.push(std::string(line));
    }
    stdinEof = true;
}

static bool popCmd(std::string& out) {
    std::lock_guard<std::mutex> lg(cmdMutex);
    if (cmdQueue.empty()) return false;
    out = cmdQueue.front(); cmdQueue.pop();
    return true;
}

// ---- Text mode: stdin = mã ca. Lần 1 mở, lần 2 đóng (phải khớp). ----
static int runTextMode(app::ApplicationContext& ctx) {
    printf("[Server] Ready. Nhap MA SO (1-9 chu so) de MO CA:\n");
    std::thread t(stdinReader); t.detach();

    while (true) {
        std::string cmd;
        if (!popCmd(cmd)) {
            if (stdinEof && !ctx.sessionService().isOpen()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        if (!ctx.sessionService().isOpen()) {
            if (!ctx.sessionLifecycle().start(cmd)) {
                printf("[Server] Ma so khong hop le (1-9 ky tu)\n");
            }
        } else {
            if (ctx.sessionService().matchCode(cmd)) {
                ctx.sessionLifecycle().stop(cmd);
                break;
            } else {
                printf("[Server] Ma so khong khop. Nhap lai:\n");
            }
        }
    }
    return 0;
}

// ---- JSON mode: stdin = lệnh JSON {"cmd":"open_session","code":"1234"}, etc. ----
static int runJsonMode(app::ApplicationContext& ctx) {
    std::thread t(stdinReader); t.detach();

    while (true) {
        std::string line;
        if (!popCmd(line)) {
            if (stdinEof) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        char cmd[32] = {0};
        if (!jsonGetString(line.c_str(), "cmd", cmd, sizeof(cmd))) continue;

        if (strcmp(cmd, "open_session") == 0) {
            char code[16] = {0};
            if (!jsonGetString(line.c_str(), "code", code, sizeof(code))) continue;
            if (!ctx.sessionLifecycle().start(code))
                printf("{\"event\":\"error\",\"message\":\"invalid_session_code\"}\n");
        }
        else if (strcmp(cmd, "close_session") == 0) {
            char code[16] = {0};
            if (!jsonGetString(line.c_str(), "code", code, sizeof(code))) continue;
            if (ctx.sessionLifecycle().stop(code)) break;
            else printf("{\"event\":\"error\",\"message\":\"code_mismatch\"}\n");
        }
        else if (strcmp(cmd, "quit") == 0) break;
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "--server";
    if (strcmp(mode, "--server") != 0) {
        fprintf(stderr, "Usage: %s --server [--json]\n", argv[0]);
        return 1;
    }
    bool jsonMode = false;
    for (int i = 2; i < argc; i++) if (strcmp(argv[i], "--json") == 0) jsonMode = true;

    setbuf(stdout, NULL);
    if (!netInit()) { fprintf(stderr, "netInit failed\n"); return 1; }

    db::initRestaurantSchema();
    db::Database::instance().openAll("data");

    // Event listener (text vs json)
    std::unique_ptr<app::IServerEventListener> listener;
    if (jsonMode) listener = std::make_unique<app::JsonEventListener>();
    else          listener = std::make_unique<app::TextLogListener>();

    // ApplicationContext = DI container — tạo + wire repositories/services/controllers/network.
    app::ApplicationContext ctx(db::Database::instance(), *listener, DEFAULT_PORT);

    if (!ctx.menuService().loadFromFile("data/menu.txt")) {
        fprintf(stderr, "[Server] Cannot load data/menu.txt\n");
        netCleanup(); return 1;
    }
    listener->onMenuLoaded((int)ctx.menuService().count());

    // LFM init: random first, sau đó override nếu tìm thấy vector trong .tbl.
    ctx.lfmService().initRandom((unsigned int)time(NULL));
    if (ctx.lfmService().loadFromRepository() && !jsonMode) {
        printf("[Server] Loaded LFM model from .tbl\n");
    }

    ctx.lfmService().rebuildOrderHistory();
    if (!jsonMode) {
        int64_t cnt = ctx.database().table(db::tbl::TRANSACTIONS).size();
        printf("[Server] orderHistory rebuilt from %lld transactions\n", (long long)cnt);
    }

    if (!ctx.tcpServer().start()) {
        fprintf(stderr, "[Server] Failed to start listener on port %d\n", DEFAULT_PORT);
        netCleanup(); return 1;
    }
    listener->onServerStarted(DEFAULT_PORT);

    int rc = jsonMode ? runJsonMode(ctx) : runTextMode(ctx);

    ctx.tcpServer().stop();
    netCleanup();
    return rc;
}
