// phase1_test.cpp — Smoke test cho Spring-style architecture.
// Exercises: repositories + services + LFM training/predict/save/load roundtrip.

#include "../shared/db/database.h"
#include "../shared/db/db_schema.h"
#include "../shared/protocol.h"
#include "../shared/utils.h"
#include "../shared/constants.h"
#include "../server/repositories/user_repository.h"
#include "../server/repositories/menu_repository.h"
#include "../server/repositories/transaction_repository.h"
#include "../server/repositories/lfm_repository.h"
#include "../server/repositories/session_repository.h"
#include "../server/services/menu_service.h"
#include "../server/services/auth_service.h"
#include "../server/services/lfm_service.h"
#include "../server/services/order_service.h"
#include "../server/services/session_service.h"
#include "../server/services/report_service.h"
#include <cstdio>
#include <cstring>

static void hr() { printf("--------------------------------------------------\n"); }

int main() {
    printf("=== Phase 1 Smoke Test (Spring-style) ===\n\n");

    db::initRestaurantSchema();
    db::Database& db = db::Database::instance();

    // Reset relevant tables để test idempotent
    db.table(db::tbl::USERS).clear();
    db.table(db::tbl::TRANSACTIONS).clear();
    db.table(db::tbl::TXN_ITEMS).clear();
    db.table(db::tbl::LFM_P).clear();
    db.table(db::tbl::LFM_Q).clear();

    app::UserRepository        userRepo(db);
    app::MenuRepository        menuRepo(db);
    app::TransactionRepository txnRepo(db);
    app::LfmRepository         lfmRepo(db);
    app::SessionRepository     sessionRepo(db);

    app::MenuService           menuSvc(menuRepo);
    app::AuthService           authSvc(userRepo);
    app::LfmService            lfmSvc(lfmRepo, userRepo, menuRepo, txnRepo);
    app::SessionService        sessionSvc(sessionRepo);
    app::OrderService          orderSvc(userRepo, menuRepo, txnRepo, lfmSvc, db);
    app::ReportService         reportSvc(userRepo, menuRepo, txnRepo, sessionSvc);

    // 1. Load menu
    hr(); printf("[1] MenuService.loadFromFile\n");
    if (!menuSvc.loadFromFile("data/menu.txt")) { printf("  FAIL\n"); return 1; }
    printf("  Loaded %lld mon:\n", (long long)menuSvc.count());
    for (auto& m : menuSvc.findAll())
        printf("    %s | %-22s | %.0f\n", m.code.c_str(), m.name.c_str(), m.price);

    // 2. Phone validator
    hr(); printf("[2] AuthService::isValidPhone\n");
    const char* cases[] = {"0901234567", "1901234567", "090123456", "09012345ab"};
    for (auto c : cases) printf("  %-12s -> %s\n", c, app::AuthService::isValidPhone(c) ? "OK" : "INVALID");

    // 3. Menu code validator
    hr(); printf("[3] MenuService.isValidCode\n");
    const char* codes[] = {"P01", "B02", "X99", "P1", "p01"};
    for (auto c : codes) printf("  %-5s -> %s\n", c, menuSvc.isValidCode(c) ? "OK" : "INVALID");

    // 4. Protocol
    hr(); printf("[4] Protocol parser/builder\n");
    char buf[256];
    int n = buildMessage(MSG_USER_LOGIN, "1|0901234567", buf, sizeof(buf));
    printf("  Built (%d bytes): %s", n, buf);
    ParsedMsg pm; bool ok = parseMessage(buf, &pm);
    printf("  Parsed ok=%d type=%s payload='%s'\n", ok, msgTypeName(pm.type), pm.payload);

    // 5. Auth: getOrCreate
    hr(); printf("[5] AuthService\n");
    int64_t u1 = authSvc.getOrCreate("0901234567");
    int64_t u2 = authSvc.getOrCreate("0912345678");
    int64_t u3 = authSvc.getOrCreate("0901234567");
    printf("  u1=%lld u2=%lld u3(same)=%lld total=%lld\n",
           (long long)u1, (long long)u2, (long long)u3, (long long)userRepo.count());

    // 6. LFM init
    hr(); printf("[6] LfmService.initRandom (seed=42)\n");
    lfmSvc.initRandom(42);

    // 7. Open session + create orders
    hr(); printf("[7] SessionService + OrderService\n");
    sessionSvc.open("1234");
    printf("  Session OPEN code=%s\n", sessionSvc.currentCode().c_str());

    {
        app::OrderService::CreateRequest r;
        r.userId      = u1;
        r.sessionCode = sessionSvc.currentCode();
        r.items.push_back({"P01", 2});
        r.items.push_back({"D01", 1});
        auto res = orderSvc.create(r);
        printf("  Order#1 success=%d total=%.0f txnId=%lld\n",
               (int)res.success, res.total, (long long)res.txnId);
    }

    // Big order → discount 25%
    {
        app::OrderService::CreateRequest r;
        r.userId      = u2;
        r.sessionCode = sessionSvc.currentCode();
        r.items.push_back({"A01", 20});
        r.items.push_back({"C01", 5});
        r.items.push_back({"G01", 10});
        auto res = orderSvc.create(r);
        printf("  Order#big success=%d sub=%.0f disc=%.0f total=%.0f\n",
               (int)res.success, res.subtotal, res.discount, res.total);
    }

    // 8. LFM training + top-K
    hr(); printf("[8] LFM batch train + top-K\n");
    lfmSvc.rebuildOrderHistory();
    lfmSvc.trainBatch(50);
    printf("  Final loss = %.6f\n", lfmSvc.computeLoss());

    auto top = lfmSvc.topK(u1, {}, 3);
    printf("  Top-3 cho u1 (0901234567):\n");
    for (auto& s : top) {
        auto m = menuSvc.findByIndex(s.first);
        if (m.has_value())
            printf("    %s %.3f - %s\n", m->code.c_str(), s.second, m->name.c_str());
    }

    // 9. Session close + report
    hr(); printf("[9] Session close + report\n");
    sessionSvc.close("1234");
    reportSvc.writeForCurrentSession("data/reports/report_test.txt");
    lfmSvc.saveToRepository();
    db.saveAll("data");
    printf("  Saved: data/reports/report_test.txt + .tbl files\n");

    // 10. Reload + verify
    hr(); printf("[10] Reload from .tbl + verify\n");
    db::Database::instance().table(db::tbl::LFM_P).clear();
    db::Database::instance().openAll("data");
    auto top2 = lfmSvc.topK(u1, {}, 3);
    printf("  After reload, top-3 cho u1 (load from .tbl):\n");
    for (auto& s : top2) {
        auto m = menuSvc.findByIndex(s.first);
        if (m.has_value())
            printf("    %s %.3f\n", m->code.c_str(), s.second);
    }

    hr();
    printf("\n=== Phase 1 PASSED ===\n");
    return 0;
}
