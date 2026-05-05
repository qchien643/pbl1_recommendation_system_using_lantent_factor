#include "state.h"

// Menu
char  menuCode[MAX_MENU][4];
char  menuName[MAX_MENU][50];
float menuPrice[MAX_MENU];
char  menuCategory[MAX_MENU];
int   menuCount = 0;

// Users
char  userPhone[MAX_USERS][11];
char  userName[MAX_USERS][NAME_LEN];
char  userDesc[MAX_USERS][DESC_LEN];
int   userTotalOrders[MAX_USERS];
int   userCount = 0;
int   orderHistory[MAX_USERS][MAX_MENU];

// Per-order transactions persistent
int   txnCount = 0;
int   txnUserIdx[MAX_TXN];
char  txnTime[MAX_TXN][20];
char  txnSessionCode[MAX_TXN][10];
int   txnItemCount[MAX_TXN];
char  txnItemCode[MAX_TXN][MAX_ITEMS][4];
int   txnItemQty[MAX_TXN][MAX_ITEMS];
float txnSubtotal[MAX_TXN];
float txnDiscount[MAX_TXN];
float txnTotal[MAX_TXN];

// LFM
float P[MAX_USERS][K];
float Q[MAX_MENU][K];

// Orders
int   orderId[MAX_ORDERS];
int   orderUserId[MAX_ORDERS];
char  orderPhone[MAX_ORDERS][11];
int   orderClientId[MAX_ORDERS];
char  orderTime[MAX_ORDERS][20];
char  orderItemCode[MAX_ORDERS][MAX_ITEMS][4];
int   orderItemQty[MAX_ORDERS][MAX_ITEMS];
float orderItemPrice[MAX_ORDERS][MAX_ITEMS];
int   orderItemCount[MAX_ORDERS];
float orderSubtotal[MAX_ORDERS];
float orderDiscount[MAX_ORDERS];
float orderTotal[MAX_ORDERS];
int   totalOrders = 0;

// Session
char  sessionCode[10];
char  sessionStart[20];
char  sessionEnd[20];
