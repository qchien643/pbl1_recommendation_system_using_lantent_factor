#include "user_store.h"
#include "../shared/state.h"
#include <cstdio>
#include <cstring>

int findUser(const char* phone) {
    if (!phone) return -1;
    for (int i = 0; i < userCount; i++) {
        if (strcmp(userPhone[i], phone) == 0) return i;
    }
    return -1;
}

int getOrCreateUser(const char* phone) {
    int idx = findUser(phone);
    if (idx >= 0) return idx;
    if (userCount >= MAX_USERS) return -1;
    strncpy(userPhone[userCount], phone, 10);
    userPhone[userCount][10] = '\0';
    userName[userCount][0] = '\0';
    userDesc[userCount][0] = '\0';
    userTotalOrders[userCount] = 0;
    for (int k = 0; k < MAX_MENU; k++) orderHistory[userCount][k] = 0;
    return userCount++;
}

void incrementHistory(int userId, int itemIdx, int qty) {
    if (userId < 0 || userId >= userCount) return;
    if (itemIdx < 0 || itemIdx >= menuCount) return;
    orderHistory[userId][itemIdx] += qty;
}

void setUserName(int userId, const char* name, const char* desc) {
    if (userId < 0 || userId >= userCount) return;
    if (name) {
        strncpy(userName[userId], name, NAME_LEN - 1);
        userName[userId][NAME_LEN - 1] = '\0';
    }
    if (desc) {
        strncpy(userDesc[userId], desc, DESC_LEN - 1);
        userDesc[userId][DESC_LEN - 1] = '\0';
    }
}

// Format moi (xoa orderHistory, them name + desc):
//   [int userCount]
//   [char[11]  userPhone[N]]
//   [char[40]  userName[N]]
//   [char[80]  userDesc[N]]
//   [int       userTotalOrders[N]]
bool saveUsers(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;
    fwrite(&userCount, sizeof(int), 1, f);
    fwrite(userPhone,       sizeof(char), (size_t)userCount * 11,       f);
    fwrite(userName,        sizeof(char), (size_t)userCount * NAME_LEN, f);
    fwrite(userDesc,        sizeof(char), (size_t)userCount * DESC_LEN, f);
    fwrite(userTotalOrders, sizeof(int),  (size_t)userCount,            f);
    fclose(f);
    return true;
}

bool loadUsers(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    int cnt = 0;
    if (fread(&cnt, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (cnt < 0 || cnt > MAX_USERS) { fclose(f); return false; }
    userCount = cnt;
    if (cnt > 0) {
        fread(userPhone,       sizeof(char), (size_t)cnt * 11,       f);
        fread(userName,        sizeof(char), (size_t)cnt * NAME_LEN, f);
        fread(userDesc,        sizeof(char), (size_t)cnt * DESC_LEN, f);
        fread(userTotalOrders, sizeof(int),  (size_t)cnt,            f);
    }
    // orderHistory se duoc rebuild tu transactions.dat
    for (int u = 0; u < cnt; u++) {
        for (int i = 0; i < MAX_MENU; i++) orderHistory[u][i] = 0;
    }
    fclose(f);
    return true;
}
