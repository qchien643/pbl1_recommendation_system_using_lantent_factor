// migrate_legacy.cpp — Đọc legacy .dat → ghi .tbl mới qua repositories.
//
// Sau Spring-style refactor: dùng repositories + services trực tiếp.

#include "../shared/db/database.h"
#include "../shared/db/db_schema.h"
#include "../shared/constants.h"
#include "../server/repositories/user_repository.h"
#include "../server/repositories/menu_repository.h"
#include "../server/repositories/transaction_repository.h"
#include "../server/repositories/lfm_repository.h"
#include "../server/services/menu_service.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

// Legacy users.dat parallel-array layout.
struct LegacyUser {
    char  phone[11];
    char  name[NAME_LEN];
    char  desc[DESC_LEN];
    int   totalOrders;
};

static std::vector<LegacyUser> readLegacyUsers(const std::string& path) {
    std::vector<LegacyUser> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    int cnt = 0;
    if (fread(&cnt, sizeof(int), 1, f) != 1 || cnt < 0 || cnt > MAX_USERS) {
        fclose(f); return out;
    }
    std::vector<char> phones((size_t)cnt * 11);
    std::vector<char> names ((size_t)cnt * NAME_LEN);
    std::vector<char> descs ((size_t)cnt * DESC_LEN);
    std::vector<int>  totals((size_t)cnt);
    fread(phones.data(), 1, phones.size(), f);
    fread(names.data(),  1, names.size(),  f);
    fread(descs.data(),  1, descs.size(),  f);
    fread(totals.data(), sizeof(int), cnt, f);
    fclose(f);

    out.resize(cnt);
    for (int i = 0; i < cnt; i++) {
        std::memcpy(out[i].phone, phones.data() + i * 11, 11);
        std::memcpy(out[i].name,  names.data()  + i * NAME_LEN, NAME_LEN);
        std::memcpy(out[i].desc,  descs.data()  + i * DESC_LEN, DESC_LEN);
        out[i].totalOrders = totals[i];
    }
    return out;
}

struct LegacyTxn {
    int   userIdx;
    char  ts[20];
    char  sess[10];
    int   itemCount;
    char  codes[MAX_ITEMS][4];
    int   qtys [MAX_ITEMS];
    float subtotal, discount, total;
};

static std::vector<LegacyTxn> readLegacyTxns(const std::string& path) {
    std::vector<LegacyTxn> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    int cnt = 0;
    if (fread(&cnt, sizeof(int), 1, f) != 1 || cnt < 0 || cnt > MAX_TXN) {
        fclose(f); return out;
    }
    out.resize(cnt);
    std::vector<int>   userIdx(cnt);
    std::vector<char>  ts((size_t)cnt * 20);
    std::vector<char>  sess((size_t)cnt * 10);
    std::vector<int>   itemCount(cnt);
    std::vector<char>  codes((size_t)cnt * MAX_ITEMS * 4);
    std::vector<int>   qtys((size_t)cnt * MAX_ITEMS);
    std::vector<float> sub(cnt), disc(cnt), tot(cnt);
    fread(userIdx.data(), sizeof(int), cnt, f);
    fread(ts.data(),       1, ts.size(),    f);
    fread(sess.data(),     1, sess.size(),  f);
    fread(itemCount.data(),sizeof(int), cnt, f);
    fread(codes.data(),    1, codes.size(), f);
    fread(qtys.data(),     sizeof(int), (size_t)cnt * MAX_ITEMS, f);
    fread(sub.data(),      sizeof(float), cnt, f);
    fread(disc.data(),     sizeof(float), cnt, f);
    fread(tot.data(),      sizeof(float), cnt, f);
    fclose(f);

    for (int i = 0; i < cnt; i++) {
        out[i].userIdx   = userIdx[i];
        std::memcpy(out[i].ts,   ts.data()   + i * 20, 20);
        std::memcpy(out[i].sess, sess.data() + i * 10, 10);
        out[i].itemCount = itemCount[i];
        for (int k = 0; k < MAX_ITEMS; k++) {
            std::memcpy(out[i].codes[k], codes.data() + (i * MAX_ITEMS + k) * 4, 4);
            out[i].qtys[k] = qtys[i * MAX_ITEMS + k];
        }
        out[i].subtotal = sub[i];
        out[i].discount = disc[i];
        out[i].total    = tot[i];
    }
    return out;
}

static bool readLegacyLfmP(const std::string& path, std::vector<std::vector<float>>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int cnt = 0, k = 0;
    if (fread(&cnt, sizeof(int), 1, f) != 1 || fread(&k, sizeof(int), 1, f) != 1 || k != K) {
        fclose(f); return false;
    }
    out.resize(cnt, std::vector<float>(K, 0.0f));
    for (int i = 0; i < cnt; i++) fread(out[i].data(), sizeof(float), K, f);
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    const char* dataDir = (argc > 1) ? argv[1] : "data";

    db::initRestaurantSchema();
    db::Database& db = db::Database::instance();

    app::UserRepository        userRepo(db);
    app::MenuRepository        menuRepo(db);
    app::TransactionRepository txnRepo(db);
    app::LfmRepository         lfmRepo(db);
    app::MenuService           menuSvc(menuRepo);

    // Load menu (cần để map item code → price)
    std::string menuTxt = std::string(dataDir) + "/menu.txt";
    if (!menuSvc.loadFromFile(menuTxt.c_str())) {
        fprintf(stderr, "[migrate] WARN: cannot load %s\n", menuTxt.c_str());
    } else {
        printf("[migrate] menu.txt → menu table: %lld items\n", (long long)menuSvc.count());
    }

    // Users
    std::string uPath = std::string(dataDir) + "/users.dat";
    auto users = readLegacyUsers(uPath);
    if (!users.empty()) {
        db.table(db::tbl::USERS).clear();
        for (size_t i = 0; i < users.size(); i++) {
            app::UserRecord u;
            u.userId      = (int64_t)i;
            u.phone       = users[i].phone;
            u.name        = users[i].name;
            u.description = users[i].desc;
            u.totalOrders = users[i].totalOrders;
            u.createdAt   = (int64_t)time(nullptr);
            userRepo.save(u);
        }
        printf("[migrate] users.dat → users table: %d users\n", (int)users.size());
    } else {
        printf("[migrate] users.dat absent — skip\n");
    }

    // Transactions + items
    std::string tPath = std::string(dataDir) + "/transactions.dat";
    auto txns = readLegacyTxns(tPath);
    if (!txns.empty()) {
        db.table(db::tbl::TRANSACTIONS).clear();
        db.table(db::tbl::TXN_ITEMS).clear();
        for (size_t i = 0; i < txns.size(); i++) {
            app::TransactionRecord t;
            t.txnId       = (int64_t)i;
            t.userId      = txns[i].userIdx;
            t.sessionCode = txns[i].sess;
            t.ts          = txns[i].ts;
            t.subtotal    = txns[i].subtotal;
            t.discount    = txns[i].discount;
            t.total       = txns[i].total;
            for (int k = 0; k < txns[i].itemCount; k++) {
                app::TxnItemRecord it;
                it.seq      = k;
                it.itemCode = txns[i].codes[k];
                it.qty      = txns[i].qtys[k];
                auto m = menuSvc.findByCode(it.itemCode);
                it.price = m.has_value() ? m->price : 0.0;
                t.items.push_back(it);
            }
            txnRepo.save(t);
        }
        printf("[migrate] transactions.dat → transactions+items: %d txns\n", (int)txns.size());
    } else {
        printf("[migrate] transactions.dat absent — skip\n");
    }

    // LFM matrices
    std::vector<std::vector<float>> P, Q;
    bool gotP = readLegacyLfmP(std::string(dataDir) + "/lfm_P.dat", P);
    bool gotQ = readLegacyLfmP(std::string(dataDir) + "/lfm_Q.dat", Q);
    if (gotP || gotQ) {
        lfmRepo.clearUsers(); lfmRepo.clearItems();
        for (size_t i = 0; i < P.size(); i++) lfmRepo.saveUserVector((int64_t)i, P[i].data(), K);
        for (size_t i = 0; i < Q.size(); i++) lfmRepo.saveItemVector((int64_t)i, Q[i].data(), K);
        printf("[migrate] lfm_P/Q.dat → lfm_p/lfm_q tables\n");
    }

    if (db.saveAll(dataDir)) {
        printf("[migrate] OK — wrote *.tbl files vào %s/\n", dataDir);
        return 0;
    } else {
        fprintf(stderr, "[migrate] FAIL — saveAll error\n");
        return 1;
    }
}
