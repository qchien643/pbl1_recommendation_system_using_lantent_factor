#include "file_manager.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include "menu.h"
#include <cstdio>
#include <cstring>

bool writeReport(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return false;

    char date[16];
    currentDate(date, sizeof(date));

    fprintf(f, "==============================================================\n");
    fprintf(f, "   BAO CAO NGAY %s\n", date);
    fprintf(f, "   Ma giao dich : %s\n", sessionCode);
    fprintf(f, "   Ca lam viec  : %s - %s\n", sessionStart, sessionEnd);
    fprintf(f, "==============================================================\n\n");

    float totalRev = 0, totalDisc = 0;
    int   discOrders = 0;
    int   itemCount[MAX_MENU] = {0};
    bool  phoneSeen[MAX_USERS] = {false};
    int   uniquePhones = 0;

    for (int o = 0; o < totalOrders; o++) {
        fprintf(f, "DON #%03d | Ban %02d | SDT: %s | %s\n",
                orderId[o], orderClientId[o], orderPhone[o], orderTime[o]);
        fprintf(f, "--------------------------------------------------------------\n");

        for (int i = 0; i < orderItemCount[o]; i++) {
            int mi = findMenuIndex(orderItemCode[o][i]);
            const char* name = mi >= 0 ? menuName[mi] : "?";
            char priceBuf[32], subBuf[32];
            formatMoney(orderItemPrice[o][i], priceBuf, sizeof(priceBuf));
            float sub = orderItemPrice[o][i] * orderItemQty[o][i];
            formatMoney(sub, subBuf, sizeof(subBuf));
            fprintf(f, "  %s  %-22s x%-3d %10s = %12s\n",
                    orderItemCode[o][i], name, orderItemQty[o][i], priceBuf, subBuf);
            if (mi >= 0) itemCount[mi] += orderItemQty[o][i];
        }

        char sBuf[32], dBuf[32], tBuf[32];
        formatMoney(orderSubtotal[o], sBuf, sizeof(sBuf));
        formatMoney(orderDiscount[o], dBuf, sizeof(dBuf));
        formatMoney(orderTotal[o],    tBuf, sizeof(tBuf));
        fprintf(f, "  Tam tinh: %s | Giam: %s | Tong: %s\n\n", sBuf, dBuf, tBuf);

        totalRev  += orderTotal[o];
        totalDisc += orderDiscount[o];
        if (orderDiscount[o] > 0) discOrders++;
        int uid = orderUserId[o];
        if (uid >= 0 && uid < MAX_USERS && !phoneSeen[uid]) {
            phoneSeen[uid] = true;
            uniquePhones++;
        }
    }

    char revBuf[32], discBuf[32];
    formatMoney(totalRev,  revBuf,  sizeof(revBuf));
    formatMoney(totalDisc, discBuf, sizeof(discBuf));
    fprintf(f, "==============================================================\n");
    fprintf(f, "TONG KET NGAY\n");
    fprintf(f, "  Tong so don        : %d\n", totalOrders);
    fprintf(f, "  Tong doanh thu     : %s\n", revBuf);
    fprintf(f, "  Tong giam gia      : %s\n", discBuf);
    fprintf(f, "  Don duoc giam      : %d / %d\n", discOrders, totalOrders);
    fprintf(f, "  So SDT khac nhau   : %d\n", uniquePhones);

    // Top 3 mon ban chay
    fprintf(f, "  Mon ban chay       : ");
    int top3[3] = {-1, -1, -1};
    for (int r = 0; r < 3; r++) {
        int best = -1, bestC = 0;
        for (int i = 0; i < menuCount; i++) {
            bool taken = false;
            for (int t = 0; t < r; t++) if (top3[t] == i) { taken = true; break; }
            if (taken) continue;
            if (itemCount[i] > bestC) { bestC = itemCount[i]; best = i; }
        }
        if (best < 0) break;
        top3[r] = best;
        if (r > 0) fprintf(f, ", ");
        fprintf(f, "%s (%d lan)", menuCode[best], itemCount[best]);
    }
    fprintf(f, "\n");
    fprintf(f, "==============================================================\n");
    fprintf(f, "LFM MODEL STATS\n");
    fprintf(f, "  Tong users da hoc  : %d\n", userCount);
    fprintf(f, "  Latent dimensions  : K=%d\n", K);
    fprintf(f, "==============================================================\n");

    fclose(f);
    return true;
}
