// shared/state.h — Cấu trúc dữ liệu parallel arrays (trích lược)

const int MAX_MENU    = 20;     // tối đa 20 món
const int MAX_ORDERS  = 1000;   // tối đa 1000 đơn / ca
const int MAX_ITEMS   = 5;      // tối đa 5 món / đơn (BR01)
const int MAX_USERS   = 1000;   // tối đa 1000 SDT
const int MAX_TXN     = 5000;   // transactions persistent xuyên ca
const int K           = 10;     // số chiều latent

// Menu (parallel arrays)
extern char  menuCode[MAX_MENU][4];
extern char  menuName[MAX_MENU][50];
extern float menuPrice[MAX_MENU];
extern int   menuCount;

// Users (mapping SDT → userId)
extern char  userPhone[MAX_USERS][11];
extern char  userName[MAX_USERS][NAME_LEN];     // tên khách
extern char  userDesc[MAX_USERS][DESC_LEN];     // mô tả tùy chọn
extern int   userTotalOrders[MAX_USERS];
extern int   userCount;
extern int   orderHistory[MAX_USERS][MAX_MENU]; // aggregate cache cho LFM

// Latent Factor matrices (P, Q)
extern float P[MAX_USERS][K];
extern float Q[MAX_MENU][K];

// Per-order transactions (persistent xuyên ca)
extern int   txnCount;
extern int   txnUserIdx[MAX_TXN];
extern char  txnTime[MAX_TXN][20];
extern int   txnItemCount[MAX_TXN];
extern char  txnItemCode[MAX_TXN][MAX_ITEMS][4];
extern int   txnItemQty[MAX_TXN][MAX_ITEMS];
extern float txnSubtotal[MAX_TXN];
extern float txnDiscount[MAX_TXN];
extern float txnTotal[MAX_TXN];
