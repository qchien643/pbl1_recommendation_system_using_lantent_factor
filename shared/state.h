#ifndef STATE_H
#define STATE_H

#include "constants.h"

// Menu (parallel arrays)
extern char  menuCode[MAX_MENU][4];
extern char  menuName[MAX_MENU][50];
extern float menuPrice[MAX_MENU];
extern char  menuCategory[MAX_MENU];
extern int   menuCount;

// Users
extern char  userPhone[MAX_USERS][11];
extern char  userName[MAX_USERS][NAME_LEN];   // Ten khach (ASCII, set khi dang ky lan dau)
extern char  userDesc[MAX_USERS][DESC_LEN];   // Mo ta ban than (optional)
extern int   userTotalOrders[MAX_USERS];
extern int   userCount;
extern int   orderHistory[MAX_USERS][MAX_MENU];  // Aggregate cache, derived tu txn*[]

// Per-order transactions persistent xuyen ca
// Khac orderId[]/orderItemCode[][] (session-only, reset moi ca)
extern int   txnCount;
extern int   txnUserIdx[MAX_TXN];
extern char  txnTime[MAX_TXN][20];
extern char  txnSessionCode[MAX_TXN][10];
extern int   txnItemCount[MAX_TXN];
extern char  txnItemCode[MAX_TXN][MAX_ITEMS][4];
extern int   txnItemQty[MAX_TXN][MAX_ITEMS];
extern float txnSubtotal[MAX_TXN];
extern float txnDiscount[MAX_TXN];
extern float txnTotal[MAX_TXN];

// Latent Factor matrices
extern float P[MAX_USERS][K];
extern float Q[MAX_MENU][K];

// Orders
extern int   orderId[MAX_ORDERS];
extern int   orderUserId[MAX_ORDERS];
extern char  orderPhone[MAX_ORDERS][11];
extern int   orderClientId[MAX_ORDERS];
extern char  orderTime[MAX_ORDERS][20];
extern char  orderItemCode[MAX_ORDERS][MAX_ITEMS][4];
extern int   orderItemQty[MAX_ORDERS][MAX_ITEMS];
extern float orderItemPrice[MAX_ORDERS][MAX_ITEMS];
extern int   orderItemCount[MAX_ORDERS];
extern float orderSubtotal[MAX_ORDERS];
extern float orderDiscount[MAX_ORDERS];
extern float orderTotal[MAX_ORDERS];
extern int   totalOrders;

// Session
extern char  sessionCode[10];
extern char  sessionStart[20];
extern char  sessionEnd[20];

#endif
