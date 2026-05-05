#ifndef TRANSACTION_STORE_H
#define TRANSACTION_STORE_H

// Per-order transactions persistent xuyen ca. Source of truth cho lich su dat mon.
// orderHistory[u][i] la cache aggregate duoc rebuild tu txn*[] o startup va sau moi append.

// Ghi 1 transaction vao txn*[] arrays (khong ghi file ngay — goi saveTransactions() o cho khac).
// Return index cua txn moi (>=0) hoac -1 neu loi / het slot.
int appendTransaction(
    int userIdx,
    const char* time20,          // "YYYY-MM-DD HH:MM:SS"
    const char* sessionCode10,   // ma ca
    int itemCount,
    const char itemCodes[][4],   // [itemCount][4]
    const int* qtys,             // [itemCount]
    float subtotal,
    float discount,
    float total);

bool saveTransactions(const char* filename);
bool loadTransactions(const char* filename);

// Scan qua txn*[] de dung lai aggregate orderHistory[u][i].
// Goi sau khi loadUsers + loadTransactions o startup.
void rebuildOrderHistory();

#endif
