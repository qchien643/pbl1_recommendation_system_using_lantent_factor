#include "order_store.h"
#include "transaction_store.h"
#include "../shared/state.h"
#include "../shared/utils.h"
#include <cstring>
#include <cstdio>

// Append 1 dong vao data/transactions.log:
// TIMESTAMP|SESSION_CODE|PHONE|CODE1,QTY1|...|CODEn,QTYn|SUBTOTAL|DISCOUNT|TOTAL
// Du lieu nay persist xuyen ca — khac voi orderHistory (chi la so dem cho LFM).
static void appendTransactionLog(int orderIdx) {
    FILE* f = fopen("data/transactions.log", "a");
    if (!f) return;
    fprintf(f, "%s|%s|%s",
            orderTime[orderIdx], sessionCode, orderPhone[orderIdx]);
    for (int i = 0; i < orderItemCount[orderIdx]; i++) {
        fprintf(f, "|%s,%d", orderItemCode[orderIdx][i], orderItemQty[orderIdx][i]);
    }
    fprintf(f, "|%.0f|%.0f|%.0f\n",
            orderSubtotal[orderIdx], orderDiscount[orderIdx], orderTotal[orderIdx]);
    fclose(f);
}

float computeDiscount(float subtotal) {
    return subtotal >= DISCOUNT_THRESHOLD ? subtotal * DISCOUNT_RATE : 0.0f;
}

int createOrder(const OrderInput* in) {
    if (!in) return -1;
    if (totalOrders >= MAX_ORDERS) return -1;
    if (in->itemCount <= 0 || in->itemCount > MAX_ITEMS) return -1;

    int idx = totalOrders;
    orderId[idx]       = idx + 1;
    orderUserId[idx]   = in->userId;
    strncpy(orderPhone[idx], in->phone, 10);
    orderPhone[idx][10] = '\0';
    orderClientId[idx] = in->clientId;
    currentTimestamp(orderTime[idx], 20);

    float sub = 0.0f;
    orderItemCount[idx] = in->itemCount;
    for (int i = 0; i < in->itemCount; i++) {
        int mi = in->itemIdx[i];
        if (mi < 0 || mi >= menuCount) return -1;
        strncpy(orderItemCode[idx][i], menuCode[mi], 3);
        orderItemCode[idx][i][3] = '\0';
        orderItemQty[idx][i]   = in->qty[i];
        orderItemPrice[idx][i] = menuPrice[mi];
        sub += menuPrice[mi] * in->qty[i];
    }
    orderSubtotal[idx] = sub;
    orderDiscount[idx] = computeDiscount(sub);
    orderTotal[idx]    = sub - orderDiscount[idx];

    if (in->userId >= 0 && in->userId < userCount) {
        userTotalOrders[in->userId]++;
    }

    appendTransactionLog(idx);

    // Persist vao binary transactions.dat thong qua parallel arrays txn*[]
    appendTransaction(
        in->userId,
        orderTime[idx],
        sessionCode,
        orderItemCount[idx],
        orderItemCode[idx],
        orderItemQty[idx],
        orderSubtotal[idx],
        orderDiscount[idx],
        orderTotal[idx]
    );

    totalOrders++;
    return orderId[idx];
}
