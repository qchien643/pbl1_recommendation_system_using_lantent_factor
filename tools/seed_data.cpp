// seed_data.cpp — Sinh data mẫu (10 personas + per-order txns + LFM trained).
// Sau Spring-style refactor: dùng repositories + services trực tiếp (không cần network).

#include "../shared/db/database.h"
#include "../shared/db/db_schema.h"
#include "../shared/constants.h"
#include "../shared/utils.h"
#include "../server/repositories/user_repository.h"
#include "../server/repositories/menu_repository.h"
#include "../server/repositories/transaction_repository.h"
#include "../server/repositories/lfm_repository.h"
#include "../server/services/menu_service.h"
#include "../server/services/auth_service.h"
#include "../server/services/lfm_service.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

struct PersonaAggregate { const char* code; int count; };
struct Persona {
    const char* phone;
    const char* name;
    const char* desc;
    int totalOrders;
    PersonaAggregate history[8];
};

static const Persona PERSONAS[] = {
    {"0901234567", "Anh Nam",  "Dan van phong, sang Pho Bo + Tra Da",      22,
     {{"P01",15},{"D01",15},{"B01",2},{NULL,0}}},
    {"0912345678", "Chi Lan",  "Sinh vien, trua Com Tam + Nuoc Ngot",      20,
     {{"C01",10},{"D02",10},{NULL,0}}},
    {"0923456789", "Bac Hung", "Khach cuoi tuan, thich Bun + Che",         18,
     {{"B01",6},{"B02",4},{"T01",5},{"D01",3},{NULL,0}}},
    {"0934567890", "Chi Mai",  "Di doan, chuyen Goi Cuon + Cha Gio",       21,
     {{"G01",8},{"A01",6},{"D02",7},{NULL,0}}},
    {"0945678901", "Anh Minh", "Gia dinh, dat nhieu mon pha tron",         23,
     {{"P01",4},{"P02",3},{"C01",5},{"D01",7},{"D02",4},{NULL,0}}},
    {"0956789012", "Co Tu",    "Di mot minh, Pho Ga + Che Ba Mau",         20,
     {{"P02",12},{"T01",8},{NULL,0}}},
    {"0967890123", "Anh Tuan", "Van phong, luan phien Pho va Bun",         18,
     {{"P01",5},{"B01",7},{"D01",6},{NULL,0}}},
    {"0978901234", "Chi Hoa",  "Cap doi, Com Chien + Goi Cuon",            15,
     {{"C02",6},{"G01",5},{"D02",4},{NULL,0}}},
    {"0989012345", "Bac Sau",  "Khach quen mien Tay, Com Tam + Cha Gio",   21,
     {{"C01",9},{"A01",7},{"D01",5},{NULL,0}}},
    {"0990123456", "Anh Khoa", "KHACH MOI (cold-start, chi 3 don)",         3,
     {{"P01",2},{"D01",1},{NULL,0}}},
};
static const int NUM_PERSONAS = (int)(sizeof(PERSONAS) / sizeof(PERSONAS[0]));

static void formatTs(time_t t, char* out20) {
    struct tm* tm = localtime(&t);
    strftime(out20, 20, "%Y-%m-%d %H:%M:%S", tm);
}

static void generateOrders(app::TransactionRepository& txnRepo,
                           app::MenuService& menuSvc,
                           int64_t userId, int userIdx, const Persona& P) {
    char units[500][4]; int unitCount = 0;
    for (int i = 0; P.history[i].code != NULL && unitCount < 500; i++) {
        for (int k = 0; k < P.history[i].count && unitCount < 500; k++) {
            std::strncpy(units[unitCount], P.history[i].code, 3);
            units[unitCount][3] = '\0'; unitCount++;
        }
    }
    for (int i = unitCount - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp[4]; std::strcpy(tmp, units[i]);
        std::strcpy(units[i], units[j]); std::strcpy(units[j], tmp);
    }

    int N = P.totalOrders;
    if (N <= 0) return;
    int basePerOrder = unitCount / N;
    int extra        = unitCount % N;
    int unitIdx      = 0;
    time_t nowT = time(NULL);

    for (int ord = 0; ord < N; ord++) {
        int cnt = basePerOrder + (ord < extra ? 1 : 0);
        if (cnt <= 0) cnt = 1;

        char  codes[MAX_ITEMS][4]; int qtys[MAX_ITEMS]; int itemCount = 0;
        for (int k = 0; k < cnt && unitIdx < unitCount; k++) {
            const char* code = units[unitIdx++];
            int existing = -1;
            for (int x = 0; x < itemCount; x++) if (std::strcmp(codes[x], code) == 0) { existing = x; break; }
            if (existing >= 0) qtys[existing]++;
            else if (itemCount < MAX_ITEMS) {
                std::strncpy(codes[itemCount], code, 3); codes[itemCount][3] = '\0';
                qtys[itemCount] = 1; itemCount++;
            } else qtys[0]++;
        }
        if (itemCount == 0) continue;

        double subtotal = 0;
        std::vector<app::TxnItemRecord> items;
        for (int x = 0; x < itemCount; x++) {
            auto m = menuSvc.findByCode(codes[x]);
            if (!m.has_value()) continue;
            subtotal += m->price * qtys[x];
            app::TxnItemRecord it;
            it.itemCode = codes[x]; it.qty = qtys[x]; it.price = m->price; it.seq = x;
            items.push_back(it);
        }
        double discount = subtotal >= DISCOUNT_THRESHOLD ? subtotal * DISCOUNT_RATE : 0.0;
        double total    = subtotal - discount;

        int daysAgo = (90 * (N - 1 - ord)) / (N > 1 ? (N - 1) : 1);
        int hour    = 9 + ((ord + userIdx * 3) % 12);
        int minute  = (ord * 17 + userIdx * 13) % 60;
        time_t ts = nowT - (time_t)daysAgo * 86400;
        struct tm* tm = localtime(&ts);
        tm->tm_hour = hour; tm->tm_min = minute; tm->tm_sec = 0;
        time_t finalTs = mktime(tm);
        char ts20[20]; formatTs(finalTs, ts20);

        char sessCode[10]; std::snprintf(sessCode, sizeof(sessCode), "SEED%03d", ord + 1);

        app::TransactionRecord txn;
        txn.userId      = userId;
        txn.sessionCode = sessCode;
        txn.ts          = ts20;
        txn.subtotal    = subtotal;
        txn.discount    = discount;
        txn.total       = total;
        txn.items       = items;
        txnRepo.save(txn);
    }
}

int main() {
    printf("=== SEED DATA GENERATOR (Spring-style) ===\n\n");

    db::initRestaurantSchema();
    db::Database& db = db::Database::instance();

    // Reset all relevant tables
    db.table(db::tbl::USERS).clear();
    db.table(db::tbl::TRANSACTIONS).clear();
    db.table(db::tbl::TXN_ITEMS).clear();
    db.table(db::tbl::LFM_P).clear();
    db.table(db::tbl::LFM_Q).clear();

    // Build repos + services manually (no full ApplicationContext for tools)
    app::UserRepository         userRepo(db);
    app::MenuRepository         menuRepo(db);
    app::TransactionRepository  txnRepo(db);
    app::LfmRepository          lfmRepo(db);

    app::MenuService            menuSvc(menuRepo);
    app::AuthService            authSvc(userRepo);
    app::LfmService             lfmSvc(lfmRepo, userRepo, menuRepo, txnRepo);

    if (!menuSvc.loadFromFile("data/menu.txt")) {
        fprintf(stderr, "ERROR: Khong mo duoc data/menu.txt\n"); return 1;
    }
    printf("[1] Loaded %lld menu items\n", (long long)menuSvc.count());

    srand(42);
    lfmSvc.initRandom(42);
    printf("[2] LFM initialized (K=%d, LR=%.3f, REG=%.3f)\n", K, LR, REG);

    for (int p = 0; p < NUM_PERSONAS; p++) {
        const Persona& P = PERSONAS[p];
        int64_t uid = authSvc.getOrCreate(P.phone);
        if (uid < 0) { fprintf(stderr, "ERROR: getOrCreate %s\n", P.phone); continue; }

        if (std::strcmp(P.phone, "0990123456") != 0) {
            authSvc.registerUser(uid, P.name, P.desc);
        }
        int before = (int)txnRepo.count();
        generateOrders(txnRepo, menuSvc, uid, p, P);
        int generated = (int)txnRepo.count() - before;

        // Set totalOrders cho user (vì bypass OrderService nên không tự inc)
        for (int x = 0; x < generated; x++) userRepo.incrementTotalOrders(uid);
        printf("  + %s (%-10s): sinh %d txns\n", P.phone, P.name, generated);
    }
    printf("[3] Created %lld users, %lld transactions\n",
           (long long)userRepo.count(), (long long)txnRepo.count());

    lfmSvc.rebuildOrderHistory();
    int totalPairs = 0;
    int64_t userCount = userRepo.count();
    int64_t menuCount = menuSvc.count();
    for (int u = 0; u < userCount; u++)
        for (int i = 0; i < menuCount; i++)
            if (lfmSvc.getOrderCount(u, i) > 0) totalPairs++;
    printf("[4] Rebuilt orderHistory: %d (user,item) pairs\n", totalPairs);

    printf("[5] Training LFM (max %d iter, patience=%d)...\n", MAX_ITER * 6, PATIENCE);
    lfmSvc.trainBatch(MAX_ITER * 6);
    printf("    Final loss = %.6f\n", lfmSvc.computeLoss());

    lfmSvc.saveToRepository();
    if (!db.saveAll("data")) {
        fprintf(stderr, "ERROR: Khong ghi duoc *.tbl files\n"); return 1;
    }
    printf("[6] Saved: users.tbl, transactions.tbl, transaction_items.tbl, lfm_p.tbl, lfm_q.tbl\n");

    printf("\n=== TOP-3 GOI Y PREVIEW ===\n");
    for (int p = 0; p < NUM_PERSONAS; p++) {
        auto u = authSvc.findByPhone(PERSONAS[p].phone);
        if (!u.has_value()) continue;
        auto top = lfmSvc.topK(u->userId, {}, 3);
        printf("  %s (%-10s):", PERSONAS[p].phone, PERSONAS[p].name);
        for (auto& s : top) {
            auto m = menuSvc.findByIndex(s.first);
            if (m.has_value()) printf(" %s %.2f,", m->code.c_str(), s.second);
        }
        printf("\n");
    }
    printf("\n=== SEED COMPLETE ===\n");
    return 0;
}
