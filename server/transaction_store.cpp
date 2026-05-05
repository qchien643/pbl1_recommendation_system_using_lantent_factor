#include "transaction_store.h"
#include "../shared/state.h"
#include <cstdio>
#include <cstring>

static int findMenuIdxByCode(const char* code) {
    if (!code) return -1;
    for (int i = 0; i < menuCount; i++) {
        if (strcmp(menuCode[i], code) == 0) return i;
    }
    return -1;
}

int appendTransaction(
    int userIdx,
    const char* time20,
    const char* sessionCode10,
    int itemCount,
    const char itemCodes[][4],
    const int* qtys,
    float subtotal,
    float discount,
    float total)
{
    if (txnCount >= MAX_TXN) return -1;
    if (itemCount <= 0 || itemCount > MAX_ITEMS) return -1;

    int t = txnCount;
    txnUserIdx[t] = userIdx;

    if (time20) {
        strncpy(txnTime[t], time20, 19);
        txnTime[t][19] = '\0';
    } else {
        txnTime[t][0] = '\0';
    }

    if (sessionCode10) {
        strncpy(txnSessionCode[t], sessionCode10, 9);
        txnSessionCode[t][9] = '\0';
    } else {
        txnSessionCode[t][0] = '\0';
    }

    txnItemCount[t] = itemCount;
    for (int i = 0; i < itemCount; i++) {
        strncpy(txnItemCode[t][i], itemCodes[i], 3);
        txnItemCode[t][i][3] = '\0';
        txnItemQty[t][i] = qtys[i];
    }
    // Zero-fill slot con lai
    for (int i = itemCount; i < MAX_ITEMS; i++) {
        txnItemCode[t][i][0] = '\0';
        txnItemQty[t][i] = 0;
    }

    txnSubtotal[t] = subtotal;
    txnDiscount[t] = discount;
    txnTotal[t]    = total;

    // KHONG cap nhat orderHistory o day — lfmOnlineUpdate() se lam o runtime,
    // rebuildOrderHistory() lam o startup va sau seeding.

    txnCount++;
    return t;
}

// Binary format:
//   [int txnCount]
//   [int    txnUserIdx[N]]
//   [char[20]  txnTime[N]]
//   [char[10]  txnSessionCode[N]]
//   [int    txnItemCount[N]]
//   [char[4]   txnItemCode[N][MAX_ITEMS]]
//   [int    txnItemQty [N][MAX_ITEMS]]
//   [float  txnSubtotal[N]]
//   [float  txnDiscount[N]]
//   [float  txnTotal[N]]
bool saveTransactions(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;
    fwrite(&txnCount, sizeof(int), 1, f);
    if (txnCount > 0) {
        fwrite(txnUserIdx,     sizeof(int),   (size_t)txnCount,                     f);
        fwrite(txnTime,        sizeof(char),  (size_t)txnCount * 20,                f);
        fwrite(txnSessionCode, sizeof(char),  (size_t)txnCount * 10,                f);
        fwrite(txnItemCount,   sizeof(int),   (size_t)txnCount,                     f);
        fwrite(txnItemCode,    sizeof(char),  (size_t)txnCount * MAX_ITEMS * 4,     f);
        fwrite(txnItemQty,     sizeof(int),   (size_t)txnCount * MAX_ITEMS,         f);
        fwrite(txnSubtotal,    sizeof(float), (size_t)txnCount,                     f);
        fwrite(txnDiscount,    sizeof(float), (size_t)txnCount,                     f);
        fwrite(txnTotal,       sizeof(float), (size_t)txnCount,                     f);
    }
    fclose(f);
    return true;
}

bool loadTransactions(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { txnCount = 0; return false; }
    int cnt = 0;
    if (fread(&cnt, sizeof(int), 1, f) != 1) { fclose(f); txnCount = 0; return false; }
    if (cnt < 0 || cnt > MAX_TXN) { fclose(f); txnCount = 0; return false; }
    txnCount = cnt;
    if (cnt > 0) {
        fread(txnUserIdx,     sizeof(int),   (size_t)cnt,                     f);
        fread(txnTime,        sizeof(char),  (size_t)cnt * 20,                f);
        fread(txnSessionCode, sizeof(char),  (size_t)cnt * 10,                f);
        fread(txnItemCount,   sizeof(int),   (size_t)cnt,                     f);
        fread(txnItemCode,    sizeof(char),  (size_t)cnt * MAX_ITEMS * 4,     f);
        fread(txnItemQty,     sizeof(int),   (size_t)cnt * MAX_ITEMS,         f);
        fread(txnSubtotal,    sizeof(float), (size_t)cnt,                     f);
        fread(txnDiscount,    sizeof(float), (size_t)cnt,                     f);
        fread(txnTotal,       sizeof(float), (size_t)cnt,                     f);
    }
    fclose(f);
    return true;
}

void rebuildOrderHistory() {
    // Reset cache
    for (int u = 0; u < userCount; u++) {
        for (int i = 0; i < MAX_MENU; i++) orderHistory[u][i] = 0;
    }
    // Accumulate tu tung transaction
    for (int t = 0; t < txnCount; t++) {
        int u = txnUserIdx[t];
        if (u < 0 || u >= userCount) continue;
        int n = txnItemCount[t];
        for (int i = 0; i < n; i++) {
            int m = findMenuIdxByCode(txnItemCode[t][i]);
            if (m >= 0) orderHistory[u][m] += txnItemQty[t][i];
        }
    }
}
