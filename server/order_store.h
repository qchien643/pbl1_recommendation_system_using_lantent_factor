#ifndef ORDER_STORE_H
#define ORDER_STORE_H

#include "../shared/constants.h"

struct OrderInput {
    int  userId;
    int  clientId;
    char phone[11];
    int  itemIdx[MAX_ITEMS];
    int  qty[MAX_ITEMS];
    int  itemCount;
};

int   createOrder(const OrderInput* in);   // tra ve orderId (>= 1), -1 neu loi
float computeDiscount(float subtotal);

#endif
