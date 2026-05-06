#ifndef DB_SCHEMA_DEF_H
#define DB_SCHEMA_DEF_H

#include "database.h"

namespace db {

// Khởi tạo toàn bộ schema cho dự án nhà hàng vào Database singleton.
// Idempotent — gọi nhiều lần OK (chỉ tạo lần đầu).
// Bảng tạo:
//   menu               (PK code,  HASH idx code)
//   users              (PK user_id INT64, HASH idx user_id, HASH idx phone UNIQUE)
//   transactions       (PK txn_id, HASH idx txn_id, BTREE idx user_id, BTREE idx ts)
//   transaction_items  (BTREE idx txn_id, BTREE idx item_code)
//   lfm_p              (HASH idx user_id UNIQUE)
//   lfm_q              (HASH idx item_idx UNIQUE)
//   sessions           (HASH idx code UNIQUE, BTREE idx opened_at)
void initRestaurantSchema();

// Tên file table tương ứng (dùng khi save/load):
//   data/<table_name>.tbl
// Migration cũ → mới: tools/migrate_legacy.cpp
// Constants for table names (single source of truth).
namespace tbl {
    constexpr const char* MENU         = "menu";
    constexpr const char* USERS        = "users";
    constexpr const char* TRANSACTIONS = "transactions";
    constexpr const char* TXN_ITEMS    = "transaction_items";
    constexpr const char* LFM_P        = "lfm_p";
    constexpr const char* LFM_Q        = "lfm_q";
    constexpr const char* SESSIONS     = "sessions";
}

namespace col {
    // menu
    constexpr const char* CODE         = "code";
    constexpr const char* NAME         = "name";
    constexpr const char* PRICE        = "price";
    constexpr const char* CATEGORY     = "category";

    // users
    constexpr const char* USER_ID      = "user_id";
    constexpr const char* PHONE        = "phone";
    constexpr const char* DESCRIPTION  = "description";
    constexpr const char* TOTAL_ORDERS = "total_orders";
    constexpr const char* CREATED_AT   = "created_at";

    // transactions
    constexpr const char* TXN_ID       = "txn_id";
    constexpr const char* SESSION_CODE = "session_code";
    constexpr const char* TS           = "ts";
    constexpr const char* SUBTOTAL     = "subtotal";
    constexpr const char* DISCOUNT     = "discount";
    constexpr const char* TOTAL        = "total";

    // transaction_items
    constexpr const char* SEQ          = "seq";
    constexpr const char* ITEM_CODE    = "item_code";
    constexpr const char* QTY          = "qty";

    // lfm
    constexpr const char* ITEM_IDX     = "item_idx";
    constexpr const char* VEC          = "vec";

    // sessions
    constexpr const char* OPENED_AT    = "opened_at";
    constexpr const char* CLOSED_AT    = "closed_at";
    constexpr const char* STATUS       = "status";
}

} // namespace db

#endif
