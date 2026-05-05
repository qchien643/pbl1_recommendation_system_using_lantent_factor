#include "display.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include "../server/menu.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

void showMenu() {
    printf("\n+----------+------------------------+------------+\n");
    printf("| MA MON   | TEN MON                | GIA        |\n");
    printf("+----------+------------------------+------------+\n");
    for (int i = 0; i < menuCount; i++) {
        char pb[32];
        formatMoney(menuPrice[i], pb, sizeof(pb));
        printf("| %-8s | %-22s | %10s |\n", menuCode[i], menuName[i], pb);
    }
    printf("+----------+------------------------+------------+\n");
}

void showSuggestions(const char* payload) {
    if (!payload || !*payload) return;
    printf("\n+-- GOI Y CHO BAN (LFM) ---------------------------+\n");
    const char* p = payload;
    while (*p) {
        char code[8] = {0};
        float score = 0;
        if (sscanf(p, "%7[^,],%f", code, &score) == 2) {
            int idx = findMenuIndex(code);
            const char* name = (idx >= 0) ? menuName[idx] : "?";
            char bar[12] = {0};
            int n = (int)(score * 9.0f);
            if (n < 0) n = 0;
            if (n > 9) n = 9;
            for (int i = 0; i < 9; i++) bar[i] = (i < n) ? '#' : '.';
            bar[9] = '\0';
            printf("| %-3s %-22s %s %.3f |\n", code, name, bar, score);
        }
        const char* nxt = strchr(p, '|');
        if (!nxt) break;
        p = nxt + 1;
    }
    printf("+--------------------------------------------------+\n");
}

void showOrderStatus(const ClientOrder* o) {
    printf("\n  Da chon:");
    if (o->count == 0) printf(" (chua co mon)");
    for (int i = 0; i < o->count; i++) printf(" [%s x%d]", o->codes[i], o->qtys[i]);
    printf("  Con lai: %d mon\n", MAX_ITEMS - o->count);
}

void showUserAck(const char* payload) {
    char tokens[4][256];
    int n = splitByPipe(payload, tokens, 4);
    if (n < 3) return;
    int userId = atoi(tokens[0]);
    bool isNew = (strcmp(tokens[1], "true") == 0);
    int cnt = atoi(tokens[2]);
    if (userId == 0 && !isNew && cnt == 0) {
        printf("\n  [Server] SDT khong hop le hoac da day.\n");
        return;
    }
    if (isNew) {
        printf("\n  Chao mung khach moi! userId=%d\n", userId);
    } else {
        printf("\n  Chao mung tro lai! userId=%d - Ban da dat %d don truoc.\n",
               userId, cnt);
    }
}

void showInvoice(const ClientOrder* o, int clientId, const char* phone,
                 const char* sessCode) {
    printf("\n+----------------------------------------------------------+\n");
    printf("|                  HOA DON - BAN %02d                        |\n", clientId);
    printf("|    Ma GD: %-4s                SDT: %-14s          |\n", sessCode, phone);
    printf("+-----+-----+--------------------+----+---------+-----------+\n");
    printf("| STT | Ma  | Ten                | SL | Don gia | T.tien    |\n");
    printf("+-----+-----+--------------------+----+---------+-----------+\n");
    for (int i = 0; i < o->count; i++) {
        char pb[32], sb[32];
        formatMoney(o->prices[i], pb, sizeof(pb));
        formatMoney(o->prices[i] * o->qtys[i], sb, sizeof(sb));
        printf("|  %-2d | %-3s | %-18s | %-2d | %7s | %9s |\n",
               i + 1, o->codes[i], o->names[i], o->qtys[i], pb, sb);
    }
    char sBuf[32], dBuf[32], tBuf[32];
    formatMoney(o->subtotal, sBuf, sizeof(sBuf));
    formatMoney(o->discount, dBuf, sizeof(dBuf));
    formatMoney(o->total,    tBuf, sizeof(tBuf));
    printf("+-----+-----+--------------------+----+---------+-----------+\n");
    printf("|                                 Tam tinh:      %9s |\n", sBuf);
    printf("|                                 Giam gia:      %9s |\n", dBuf);
    printf("|                                 TONG CONG:     %9s |\n", tBuf);
    printf("+----------------------------------------------------------+\n");
}
