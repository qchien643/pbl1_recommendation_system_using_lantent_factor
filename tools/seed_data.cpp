// seed_data.cpp — Sinh du lieu mau PER-ORDER cho demo:
//   - data/users.dat         : 10 SDT + ten + mo ta (format moi)
//   - data/transactions.dat  : tung lan dat mon cu the cua moi persona
//   - data/lfm_P.dat         : User latent matrix da train
//   - data/lfm_Q.dat         : Item latent matrix da train
//   - data/personas.txt      : Ban mo ta + list orders dang readable
//
// Chay sau khi build:   ./build/seed_data  (tu goc du an)

#include "../shared/state.h"
#include "../shared/constants.h"
#include "../shared/utils.h"
#include "../server/menu.h"
#include "../server/user_store.h"
#include "../server/transaction_store.h"
#include "../server/lfm.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

struct PersonaAggregate {
    const char* code;
    int count;  // tong so units trong lich su persona (se duoc chia nho thanh cac don rieng)
};

struct Persona {
    const char* phone;
    const char* name;                // ASCII no-diacritic
    const char* desc;                // mo ta
    int totalOrders;                 // so don — se sinh chinh xac con so nay
    PersonaAggregate history[8];     // ket thuc bang code=NULL
};

// 10 persona da dang de LFM hoc duoc thi hieu ca nhan hoa.
// Moi persona: "totalOrders" don voi tong units = sum(history[i].count).
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

// Format timestamp YYYY-MM-DD HH:MM:SS tu time_t
static void formatTs(time_t t, char* out20) {
    struct tm* tm = localtime(&t);
    strftime(out20, 20, "%Y-%m-%d %H:%M:%S", tm);
}

// Sinh totalOrders don cho persona bang cach chia tong units thanh cac don rieng.
// Thoi gian rai deu trong 90 ngay qua, session code "SEEDxxx".
static void generateOrdersForPersona(int userIdx, const Persona& P) {
    // 1. Build danh sach units tu history[] aggregate
    //    Max: neu 1 persona co 50 units = 22 don * 2.5 units/don, cap cung < 500
    char units[500][4];
    int unitCount = 0;
    for (int i = 0; P.history[i].code != NULL && unitCount < 500; i++) {
        for (int k = 0; k < P.history[i].count && unitCount < 500; k++) {
            strncpy(units[unitCount], P.history[i].code, 3);
            units[unitCount][3] = '\0';
            unitCount++;
        }
    }

    // 2. Shuffle de cac don co variety (Fisher-Yates)
    for (int i = unitCount - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char tmp[4]; strcpy(tmp, units[i]); strcpy(units[i], units[j]); strcpy(units[j], tmp);
    }

    // 3. Chia tong unitCount units thanh totalOrders don
    int N = P.totalOrders;
    if (N <= 0) return;
    int basePerOrder = unitCount / N;        // don binh thuong nhan so nay
    int extra = unitCount % N;               // `extra` don dau duoc 1 unit nhieu hon
    int unitIdx = 0;

    time_t nowT = time(NULL);

    for (int ord = 0; ord < N; ord++) {
        int cnt = basePerOrder + (ord < extra ? 1 : 0);
        if (cnt <= 0) cnt = 1;  // toi thieu 1 unit / don

        // Gom units cua don nay, merge duplicate cung code
        char codes[MAX_ITEMS][4];
        int  qtys[MAX_ITEMS];
        int  itemCount = 0;
        for (int k = 0; k < cnt && unitIdx < unitCount; k++) {
            const char* code = units[unitIdx++];
            int existing = -1;
            for (int x = 0; x < itemCount; x++) {
                if (strcmp(codes[x], code) == 0) { existing = x; break; }
            }
            if (existing >= 0) {
                qtys[existing]++;
            } else if (itemCount < MAX_ITEMS) {
                strncpy(codes[itemCount], code, 3);
                codes[itemCount][3] = '\0';
                qtys[itemCount] = 1;
                itemCount++;
            } else {
                // Qua MAX_ITEMS — cong them vao item dau
                qtys[0]++;
            }
        }
        if (itemCount == 0) continue;

        // Tinh tien
        float subtotal = 0.0f;
        for (int x = 0; x < itemCount; x++) {
            int m = findMenuIndex(codes[x]);
            if (m >= 0) subtotal += menuPrice[m] * qtys[x];
        }
        float discount = subtotal >= DISCOUNT_THRESHOLD ? subtotal * DISCOUNT_RATE : 0.0f;
        float total    = subtotal - discount;

        // Fake timestamp rai deu 90 ngay
        // don dau tien = 90 ngay truoc, don cuoi cung = hom nay
        int daysAgo = (90 * (N - 1 - ord)) / (N > 1 ? (N - 1) : 1);
        // Them bien doi trong ngay: 9h-20h, moi persona mot pattern
        int hour = 9 + ((ord + userIdx * 3) % 12);
        int minute = (ord * 17 + userIdx * 13) % 60;
        time_t ts = nowT - (time_t)daysAgo * 86400;
        struct tm* tm = localtime(&ts);
        tm->tm_hour = hour;
        tm->tm_min  = minute;
        tm->tm_sec  = 0;
        time_t finalTs = mktime(tm);

        char ts20[20];
        formatTs(finalTs, ts20);

        // Fake session code SEED + 3 digits
        char sessCode[10];
        snprintf(sessCode, sizeof(sessCode), "SEED%03d", ord + 1);

        appendTransaction(userIdx, ts20, sessCode, itemCount,
                          codes, qtys, subtotal, discount, total);
    }
}

// Viet personas.txt voi format per-order moi
static void writePersonasTxt(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# DU LIEU MAU — 10 personas (per-order)\n");
    fprintf(f, "# Sinh boi tools/seed_data.cpp. Dung de test va demo.\n");
    fprintf(f, "# ==========================================================\n\n");

    for (int p = 0; p < NUM_PERSONAS; p++) {
        const Persona& P = PERSONAS[p];
        int uid = findUser(P.phone);
        fprintf(f, "%s | %-10s | %2d don | %s\n",
                P.phone, P.name, P.totalOrders, P.desc);

        // Liet ke tat ca txn cua user nay, theo thu tu thoi gian
        int txnIdxSorted[1000];
        int nT = 0;
        for (int t = 0; t < txnCount && nT < 1000; t++) {
            if (txnUserIdx[t] == uid) txnIdxSorted[nT++] = t;
        }
        // Insertion sort theo txnTime
        for (int i = 1; i < nT; i++) {
            int cur = txnIdxSorted[i]; int j = i - 1;
            while (j >= 0 && strcmp(txnTime[txnIdxSorted[j]], txnTime[cur]) > 0) {
                txnIdxSorted[j + 1] = txnIdxSorted[j]; j--;
            }
            txnIdxSorted[j + 1] = cur;
        }
        for (int i = 0; i < nT; i++) {
            int t = txnIdxSorted[i];
            fprintf(f, "   Don %2d (%s): ", i + 1, txnTime[t]);
            for (int k = 0; k < txnItemCount[t]; k++) {
                if (k > 0) fprintf(f, ", ");
                fprintf(f, "%s x%d", txnItemCode[t][k], txnItemQty[t][k]);
            }
            fprintf(f, " = %.0fd", txnTotal[t]);
            if (txnDiscount[t] > 0) fprintf(f, " (-%.0f)", txnDiscount[t]);
            fprintf(f, "\n");
        }

        if (uid >= 0) {
            int idx[3]; float sc[3];
            int n = lfmGetTopK(uid, NULL, 0, idx, sc, 3);
            fprintf(f, "   Goi y LFM:  ");
            for (int i = 0; i < n; i++) {
                if (i > 0) fprintf(f, ", ");
                fprintf(f, "%s (%.3f)", menuCode[idx[i]], sc[i]);
            }
            fprintf(f, "\n\n");
        }
    }
    fclose(f);
}

int main() {
    printf("=== SEED DATA GENERATOR (per-order) ===\n\n");

    if (!loadMenu("data/menu.txt")) {
        fprintf(stderr, "ERROR: Khong mo duoc data/menu.txt (chay tu goc du an)\n");
        return 1;
    }
    printf("[1] Loaded %d menu items\n", menuCount);

    // Reset state
    userCount = 0;
    txnCount = 0;
    memset(orderHistory, 0, sizeof(orderHistory));
    memset(userTotalOrders, 0, sizeof(userTotalOrders));

    // Seed random deterministically (reproducible demo data)
    srand(42);
    lfmInit(42);
    printf("[2] LFM initialized (K=%d, LR=%.3f, REG=%.3f)\n", K, LR, REG);

    // Tao user + sinh transactions cho moi persona
    for (int p = 0; p < NUM_PERSONAS; p++) {
        const Persona& P = PERSONAS[p];
        int uid = getOrCreateUser(P.phone);
        if (uid < 0) { fprintf(stderr, "ERROR: getOrCreateUser %s\n", P.phone); continue; }

        // Khach mau: set ten + desc ngay (nhung khach 'Anh Khoa' la KHACH MOI de test flow register)
        if (strcmp(P.phone, "0990123456") != 0) {
            setUserName(uid, P.name, P.desc);
        }
        userTotalOrders[uid] = P.totalOrders;

        int txnBefore = txnCount;
        generateOrdersForPersona(uid, P);
        printf("  + %s (%-10s): sinh %d txns\n",
               P.phone, P.name, txnCount - txnBefore);
    }
    printf("[3] Created %d users, %d transactions\n", userCount, txnCount);

    // Rebuild aggregate orderHistory tu txn*[]
    rebuildOrderHistory();
    int totalPairs = 0;
    for (int u = 0; u < userCount; u++) {
        for (int i = 0; i < menuCount; i++) {
            if (orderHistory[u][i] > 0) totalPairs++;
        }
    }
    printf("[4] Rebuilt orderHistory: %d (user,item) pairs\n", totalPairs);

    // Train LFM
    printf("[5] Training LFM (max %d iter, patience=%d)...\n",
           MAX_ITER * 6, PATIENCE);
    lfmTrainFromHistory(MAX_ITER * 6);
    float loss = lfmComputeLoss();
    printf("    Final loss = %.6f\n", loss);

    // Save
    if (!saveUsers("data/users.dat")) {
        fprintf(stderr, "ERROR: Khong ghi duoc data/users.dat\n");
        return 1;
    }
    if (!saveTransactions("data/transactions.dat")) {
        fprintf(stderr, "ERROR: Khong ghi duoc data/transactions.dat\n");
        return 1;
    }
    if (!lfmSaveModels("data/lfm_P.dat", "data/lfm_Q.dat")) {
        fprintf(stderr, "ERROR: Khong ghi duoc data/lfm_*.dat\n");
        return 1;
    }
    printf("[6] Saved: users.dat, transactions.dat, lfm_P.dat, lfm_Q.dat\n");

    writePersonasTxt("data/personas.txt");
    printf("[7] Wrote data/personas.txt (per-order listing)\n");

    // Preview top-3 moi persona
    printf("\n=== TOP-3 GOI Y PREVIEW ===\n");
    for (int p = 0; p < NUM_PERSONAS; p++) {
        int uid = findUser(PERSONAS[p].phone);
        if (uid < 0) continue;
        int idx[3]; float sc[3];
        int n = lfmGetTopK(uid, NULL, 0, idx, sc, 3);
        printf("  %s (%-10s): ", PERSONAS[p].phone, PERSONAS[p].name);
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(", ");
            printf("%s %.2f", menuCode[idx[i]], sc[i]);
        }
        printf("\n");
    }

    printf("\n=== SEED COMPLETE ===\n");
    printf("Khoi dong server (./build/server --server) → tu dong load.\n");
    return 0;
}
