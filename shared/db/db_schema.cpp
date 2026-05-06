#include "db_schema.h"
#include "../constants.h"

namespace db {

void initRestaurantSchema() {
    Database& d = Database::instance();
    if (d.has(tbl::MENU)) return;  // idempotent

    // -- menu --
    {
        Schema s;
        s.add(col::CODE,     ColType::STR, 4)
         .add(col::NAME,     ColType::STR, 50)
         .add(col::PRICE,    ColType::DOUBLE)
         .add(col::CATEGORY, ColType::STR, 2)
         .setPrimaryKey(col::CODE);
        Table& t = d.createTable(tbl::MENU, std::move(s));
        t.addIndex(col::CODE, IndexKind::HASH, /*unique=*/true);
    }

    // -- users --
    {
        Schema s;
        s.add(col::USER_ID,      ColType::INT64)
         .add(col::PHONE,        ColType::STR, 11)
         .add(col::NAME,         ColType::STR, NAME_LEN)
         .add(col::DESCRIPTION,  ColType::STR, DESC_LEN)
         .add(col::TOTAL_ORDERS, ColType::INT64)
         .add(col::CREATED_AT,   ColType::INT64)
         .setPrimaryKey(col::USER_ID);
        Table& t = d.createTable(tbl::USERS, std::move(s));
        t.addIndex(col::USER_ID, IndexKind::HASH, /*unique=*/true);
        t.addIndex(col::PHONE,   IndexKind::HASH, /*unique=*/true);
    }

    // -- transactions --
    {
        Schema s;
        s.add(col::TXN_ID,       ColType::INT64)
         .add(col::USER_ID,      ColType::INT64)
         .add(col::SESSION_CODE, ColType::STR, 10)
         .add(col::TS,           ColType::STR, 20)
         .add(col::SUBTOTAL,     ColType::DOUBLE)
         .add(col::DISCOUNT,     ColType::DOUBLE)
         .add(col::TOTAL,        ColType::DOUBLE)
         .setPrimaryKey(col::TXN_ID);
        Table& t = d.createTable(tbl::TRANSACTIONS, std::move(s));
        t.addIndex(col::TXN_ID,   IndexKind::HASH,  /*unique=*/true);
        t.addIndex(col::USER_ID,  IndexKind::BTREE);
        t.addIndex(col::TS,       IndexKind::BTREE);
    }

    // -- transaction_items --
    {
        Schema s;
        s.add(col::TXN_ID,    ColType::INT64)
         .add(col::SEQ,       ColType::INT64)
         .add(col::ITEM_CODE, ColType::STR, 4)
         .add(col::QTY,       ColType::INT64)
         .add(col::PRICE,     ColType::DOUBLE);
        Table& t = d.createTable(tbl::TXN_ITEMS, std::move(s));
        t.addIndex(col::TXN_ID,    IndexKind::BTREE);
        t.addIndex(col::ITEM_CODE, IndexKind::BTREE);
    }

    // -- lfm_p --
    {
        Schema s;
        s.add(col::USER_ID, ColType::INT64)
         .add(col::VEC,     ColType::BLOB, K * sizeof(float))
         .setPrimaryKey(col::USER_ID);
        Table& t = d.createTable(tbl::LFM_P, std::move(s));
        t.addIndex(col::USER_ID, IndexKind::HASH, /*unique=*/true);
    }

    // -- lfm_q --
    {
        Schema s;
        s.add(col::ITEM_IDX, ColType::INT64)
         .add(col::VEC,      ColType::BLOB, K * sizeof(float))
         .setPrimaryKey(col::ITEM_IDX);
        Table& t = d.createTable(tbl::LFM_Q, std::move(s));
        t.addIndex(col::ITEM_IDX, IndexKind::HASH, /*unique=*/true);
    }

    // -- sessions --
    {
        Schema s;
        s.add(col::CODE,      ColType::STR, 10)
         .add(col::OPENED_AT, ColType::STR, 20)
         .add(col::CLOSED_AT, ColType::STR, 20)
         .add(col::STATUS,    ColType::STR, 2)
         .setPrimaryKey(col::CODE);
        Table& t = d.createTable(tbl::SESSIONS, std::move(s));
        t.addIndex(col::CODE,      IndexKind::HASH, /*unique=*/true);
        t.addIndex(col::OPENED_AT, IndexKind::BTREE);
    }
}

} // namespace db
