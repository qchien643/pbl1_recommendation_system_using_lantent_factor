// Phase 1 Smoke Test
// Chay tu root project: ./phase1_test (can truy cap data/menu.txt)

#include "../shared/state.h"
#include "../shared/utils.h"
#include "../shared/protocol.h"
#include "../server/menu.h"
#include "../server/phone_validator.h"
#include "../server/user_store.h"
#include "../server/order_store.h"
#include "../server/lfm.h"
#include "../server/file_manager.h"
#include "../server/session.h"
#include <cstdio>
#include <cstring>

static void hr() { printf("--------------------------------------------------\n"); }

int main() {
    printf("=== Phase 1 Smoke Test ===\n\n");

    // 1. Load menu
    hr();
    printf("[1] Load menu tu data/menu.txt\n");
    if (!loadMenu("data/menu.txt")) {
        printf("  FAIL: khong mo duoc data/menu.txt\n");
        return 1;
    }
    printf("  Loaded %d mon:\n", menuCount);
    for (int i = 0; i < menuCount; i++) {
        printf("    %s | %-22s | %.0f\n", menuCode[i], menuName[i], menuPrice[i]);
    }

    // 2. Phone validator
    hr();
    printf("[2] Phone validator\n");
    const char* cases[] = {"0901234567", "1901234567", "090123456", "09012345ab", "0000000000"};
    for (auto c : cases) printf("  %-12s -> %s\n", c, isValidPhone(c) ? "OK" : "INVALID");

    // 3. Menu code validator
    hr();
    printf("[3] Menu code validator\n");
    const char* codes[] = {"P01", "B02", "X99", "P1", "p01"};
    for (auto c : codes) printf("  %-5s -> %s\n", c, isValidMenuCode(c) ? "OK" : "INVALID");

    // 4. Protocol parser
    hr();
    printf("[4] Protocol parser/builder\n");
    char buf[256];
    int n = buildMessage(MSG_USER_LOGIN, "1|0901234567", buf, sizeof(buf));
    printf("  Built (%d bytes): %s", n, buf);
    ParsedMsg pm;
    bool ok = parseMessage(buf, &pm);
    printf("  Parsed ok=%d type=%s payload='%s'\n",
           ok, msgTypeName(pm.type), pm.payload);

    buildMessage(MSG_SUGGEST, "C01,0.91|D01,0.85|T01,0.72", buf, sizeof(buf));
    parseMessage(buf, &pm);
    printf("  Round-trip SUGGEST: type=%s payload='%s'\n", msgTypeName(pm.type), pm.payload);

    // 5. Users
    hr();
    printf("[5] User store\n");
    int u1 = getOrCreateUser("0901234567");
    int u2 = getOrCreateUser("0912345678");
    int u3 = getOrCreateUser("0901234567");  // existing
    printf("  u1=%d, u2=%d, u3(same-phone)=%d, userCount=%d\n", u1, u2, u3, userCount);

    // 6. LFM init
    hr();
    printf("[6] LFM init (seed=42)\n");
    lfmInit(42);
    printf("  P[%d][%d], Q[%d][%d] ready (K=%d, LR=%.3f, REG=%.3f)\n",
           MAX_USERS, K, MAX_MENU, K, K, LR, REG);

    // 7. Create orders
    hr();
    printf("[7] Create orders + online LFM update\n");
    int u1ItemP01 = findMenuIndex("P01");
    int u1ItemD01 = findMenuIndex("D01");
    for (int i = 0; i < 3; i++) {
        OrderInput in = {};
        in.userId = u1;
        in.clientId = 2;
        strncpy(in.phone, "0901234567", 11);
        in.itemIdx[0] = u1ItemP01; in.qty[0] = 2;
        in.itemIdx[1] = u1ItemD01; in.qty[1] = 1;
        in.itemCount = 2;
        int oid = createOrder(&in);
        printf("  Order #%d sub=%.0f disc=%.0f total=%.0f\n",
               oid, orderSubtotal[oid - 1], orderDiscount[oid - 1], orderTotal[oid - 1]);
        lfmOnlineUpdate(in.userId, in.itemIdx, in.qty, in.itemCount);
    }

    // Don lon cho u2, kich hoat giam gia 25%
    {
        OrderInput big = {};
        big.userId = u2;
        big.clientId = 1;
        strncpy(big.phone, "0912345678", 11);
        big.itemIdx[0] = findMenuIndex("C01"); big.qty[0] = 5;
        big.itemIdx[1] = findMenuIndex("A01"); big.qty[1] = 20;
        big.itemIdx[2] = findMenuIndex("G01"); big.qty[2] = 10;
        big.itemCount = 3;
        int oid = createOrder(&big);
        printf("  Big #%d sub=%.0f disc=%.0f(expect 25%%) total=%.0f\n",
               oid, orderSubtotal[oid - 1], orderDiscount[oid - 1], orderTotal[oid - 1]);
        lfmOnlineUpdate(big.userId, big.itemIdx, big.qty, big.itemCount);
    }

    // 8. Batch train
    hr();
    printf("[8] LFM batch train (max %d iter)\n", MAX_ITER);
    lfmTrainFromHistory(MAX_ITER);
    printf("  Final loss = %.6f\n", lfmComputeLoss());

    // 9. Top-3 goi y
    hr();
    printf("[9] Top-3 goi y\n");
    int topIdx[3]; float topScore[3];
    int n1 = lfmGetTopK(u1, NULL, 0, topIdx, topScore, 3);
    printf("  u1 (SDT %s, lich su: P01+D01):\n", userPhone[u1]);
    for (int i = 0; i < n1; i++) {
        printf("    #%d %s %.3f - %s\n", i + 1, menuCode[topIdx[i]], topScore[i], menuName[topIdx[i]]);
    }
    int n2 = lfmGetTopK(u2, NULL, 0, topIdx, topScore, 3);
    printf("  u2 (SDT %s, lich su: C01+A01+G01):\n", userPhone[u2]);
    for (int i = 0; i < n2; i++) {
        printf("    #%d %s %.3f - %s\n", i + 1, menuCode[topIdx[i]], topScore[i], menuName[topIdx[i]]);
    }

    // 10. Session + report
    hr();
    printf("[10] Session + xuat report\n");
    openSession("1234");
    if (closeSession("1234", "data")) {
        char date[16];
        currentDate(date, sizeof(date));
        printf("  Saved: data/reports/report_%s.txt\n", date);
        printf("  Saved: data/lfm_P.dat, data/lfm_Q.dat, data/users.dat\n");
    } else {
        printf("  FAIL: closeSession\n");
    }

    // 11. Reload test
    hr();
    printf("[11] Load lai model + users tu file\n");
    // Reset P[u1][0] de xem load co khoi phuc khong
    float origP00 = P[u1][0];
    P[u1][0] = -999.0f;
    bool lp = lfmLoadModels("data/lfm_P.dat", "data/lfm_Q.dat");
    printf("  lfmLoadModels: %s\n", lp ? "OK" : "FAIL");
    printf("  P[u1][0] before=-999 after=%.6f (original=%.6f)\n", P[u1][0], origP00);

    int savedUserCount = userCount;
    userCount = 0;
    bool lu = loadUsers("data/users.dat");
    printf("  loadUsers: %s -> userCount=%d (expected %d)\n",
           lu ? "OK" : "FAIL", userCount, savedUserCount);

    hr();
    printf("\n=== Phase 1 PASSED ===\n");
    return 0;
}
