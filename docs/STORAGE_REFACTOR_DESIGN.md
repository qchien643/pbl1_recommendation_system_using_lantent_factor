# Storage Refactor — OOP Tables + Indexed Storage

**Status:** Design v1 — chờ duyệt
**Mục tiêu:** Thay parallel-arrays + flat files bằng **mini-DBMS** OOP có schema kiểu SQL, index B-tree / Hash / Fenwick.

---

## 1. Động cơ

### Vấn đề hiện tại
| Operation | Hiện tại | Cost |
|---|---|---|
| Login by phone | linear scan `userPhone[]` | **O(N)** mỗi lần đăng nhập |
| Tra menu code | linear scan `menuCode[]` | **O(M)** mỗi item parse |
| Tra giao dịch của 1 SDT | full scan `txn*[]` | **O(T)** với T=5000 |
| Báo cáo doanh thu khoảng ngày | full scan + filter | **O(T)** |
| `orderHistory[U][I]` | dày đặc 1000×20=20k cells, đa số = 0 | RAM lãng phí |
| File format | mỗi store có format binary tay | mỗi store 1 reader/writer riêng |

### Sau refactor
| Operation | Mới | Cost |
|---|---|---|
| Login by phone | `users.byPhone.get(phone)` | **O(1)** |
| Tra menu code | `menu.byCode.get(code)` | **O(1)** |
| Tra giao dịch SDT | `transactions.byUserId.range(userId)` (BTree) | **O(log T + k)** |
| Báo cáo khoảng ngày | `transactions.byTime.range(from, to)` | **O(log T + k)** |
| Top-3 món bán chạy | Fenwick `itemSalesBIT.query()` | **O(M log M)** |
| File format | mỗi `Table` 1 file `.tbl` thống nhất | reader/writer chung |

---

## 2. Class Diagram

```
                    ┌─────────────┐
                    │  Database   │  singleton — owns all tables
                    │  - tables   │
                    │  + open()   │
                    │  + close()  │
                    │  + table()  │
                    └──────┬──────┘
                           │ owns *
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       ┌────────┐    ┌─────────┐    ┌───────────┐
       │ Table  │    │ Schema  │    │  Index    │  abstract
       │--------│    │---------│    │-----------│
       │ schema │◄──►│ columns │    │+ insert() │
       │ rows   │    │ pk      │    │+ erase()  │
       │ idxs[] │    │ uniques │    │+ find()   │
       │--------│    └─────────┘    │+ range()  │
       │+insert │                   └─────┬─────┘
       │+update │                         │
       │+remove │                  ┌──────┼──────┐
       │+find   │                  ▼      ▼      ▼
       │+scan   │           ┌─────────┐ ┌──────┐ ┌────────┐
       │+saveTo │           │HashIdx  │ │BTree │ │Fenwick │
       │+loadFr │           │ <K,RID> │ │<K,..>│ │ <int>  │
       └────────┘           └─────────┘ └──────┘ └────────┘
            │
            ▼ has many
       ┌────────┐
       │  Row   │
       │--------│
       │ values │  vector<Value>
       └────────┘
            │
            ▼
       ┌────────┐
       │ Value  │  variant: int64 / double / string / blob
       └────────┘
```

### Files mới (`shared/db/`)
```
shared/db/
├── value.h / .cpp        — Value (variant int/double/string/blob)
├── column.h              — Column descriptor (name, type, size, flags)
├── schema.h / .cpp       — Schema (columns, PK, unique constraints)
├── row.h / .cpp          — Row (values + dirty flag)
├── index.h               — Index abstract base
├── hash_index.h / .cpp   — HashIndex<K> với separate-chaining manual
├── btree_index.h / .cpp  — BTreeIndex<K> order-31, manual implementation
├── fenwick.h / .cpp      — FenwickTree<int64> cho aggregate queries
├── table.h / .cpp        — Table (schema + rows + indexes + persistence)
├── database.h / .cpp     — Database (registry, open/close all tables)
├── codec.h / .cpp        — binary read/write primitives
└── query.h / .cpp        — RangeQuery, ScanFilter helpers
```

---

## 3. Schema mapping (parallel-arrays → tables)

### Table `menu` (PK=`code`)
| Column | Type | Index |
|---|---|---|
| `code` | STR(3) | **HASH (PK)** |
| `name` | STR(50) | — |
| `price` | DOUBLE | — |
| `category` | STR(1) | — |

### Table `users` (PK=`user_id`, UNIQUE=`phone`)
| Column | Type | Index |
|---|---|---|
| `user_id` | INT64 | **HASH (PK)** |
| `phone` | STR(10) | **HASH (UNIQUE)** ← thay linear scan login |
| `name` | STR(40) | — |
| `description` | STR(80) | — |
| `total_orders` | INT64 | **FenwickTree** (top-N customers by orders) |
| `created_at` | INT64 (unix ts) | — |

### Table `transactions` (PK=`txn_id`)
| Column | Type | Index |
|---|---|---|
| `txn_id` | INT64 | **HASH (PK)** |
| `user_id` | INT64 | **BTree** ← lịch sử khách |
| `session_code` | STR(9) | BTree |
| `ts` | INT64 | **BTree** ← range query theo ngày |
| `subtotal` | DOUBLE | — |
| `discount` | DOUBLE | — |
| `total` | DOUBLE | **FenwickTree** (revenue prefix sum) |

### Table `transaction_items` (PK=composite `txn_id+seq`)
| Column | Type | Index |
|---|---|---|
| `txn_id` | INT64 | **BTree** ← join với transactions |
| `seq` | INT32 | — |
| `item_code` | STR(3) | BTree (best-seller report) |
| `qty` | INT32 | — |
| `price` | DOUBLE | — |

### Table `order_history` (replaces `orderHistory[U][I]` matrix)
**Sparse storage** — chỉ lưu cell ≠ 0.
| Column | Type | Index |
|---|---|---|
| `user_id` | INT64 | **HASH composite (user_id, item_code)** |
| `item_code` | STR(3) | (composite key) |
| `count` | INT32 | — |

→ giảm từ 20.000 cell xuống thực tế ~vài trăm.

### Table `lfm_p` / `lfm_q` (ma trận latent)
| Column | Type | Index |
|---|---|---|
| `id` | INT64 (user_id hoặc item_idx) | **HASH (PK)** |
| `vec` | BLOB(K × float = 40 bytes) | — |

→ thay 2D float array bằng row-per-vector. Load full vào RAM khi training, write khi đóng ca.

### Table `sessions`
| Column | Type | Index |
|---|---|---|
| `code` | STR(9) | **HASH (PK)** |
| `opened_at` | INT64 | BTree |
| `closed_at` | INT64 | — |
| `status` | STR(1) | — |  *(O = open, C = closed)*

---

## 4. File format (`*.tbl`)

Mỗi table → 1 file `data/<name>.tbl`:

```
[16 bytes magic]   "PBL1DBv1\0\0\0\0\0\0\0\0"
[4 bytes]          schema_hash (CRC32 của schema để phát hiện migration)
[4 bytes]          row_count
[4 bytes]          row_size_bytes (fixed-width — string padded)
[row_count × row_size_bytes]   rows in insertion order
```

Indexes **không** lưu xuống file — rebuild từ rows khi `Table.load()` (đơn giản, chấp nhận chi phí khởi động ~ms).

Migration: nếu schema_hash mismatch → in cảnh báo + không load.

---

## 5. API ví dụ (caller-side)

### Trước (parallel arrays)
```cpp
int findUser(const char* phone) {
    for (int i = 0; i < userCount; i++)
        if (strcmp(userPhone[i], phone) == 0) return i;
    return -1;
}
```

### Sau (Table + HashIndex)
```cpp
auto& users = db.table("users");
int64_t uid = users.findOne("phone", phone);  // O(1)
if (uid < 0) {
    uid = users.insert({{"phone", phone}, {"name", ""}, {"created_at", now()}});
}
```

### Range query (mới — chưa có hiện tại)
```cpp
auto& txns = db.table("transactions");
auto rows = txns.range("ts", fromTs, toTs);   // BTree
double revenue = 0;
for (auto& r : rows) revenue += r.get<double>("total");
```

---

## 6. Roadmap implement

| Phase | Việc | File ảnh hưởng |
|---|---|---|
| **A** | Core db layer (Value, Column, Schema, Row, Table, HashIndex, BTreeIndex, Fenwick, codec) | `shared/db/*` (mới) |
| **B** | `Database` singleton + load/save tất cả tables | `shared/db/database.*` |
| **C** | Migration tool: `tools/migrate_v0_to_v1.cpp` đọc cũ → ghi mới | `tools/migrate_*.cpp` |
| **D** | Refactor stores → adapter dùng Database | `server/user_store.*`, `server/menu.*`, `server/transaction_store.*`, `server/order_store.*`, `server/lfm.*`, `server/file_manager.*`, `server/session.*` |
| **E** | Bỏ globals trong `shared/state.{h,cpp}` (giữ shim cho session-only orders nếu cần) | `shared/state.*` |
| **F** | Update `CMakeLists.txt` add `shared/db/*.cpp` | build |
| **G** | Update tests + `tools/seed_data.cpp` | tests, tools |
| **H** | Update knowledge base (`.claude/knowledge/06`, `08`, `10`) | docs |

---

## 7. Decisions cần xác nhận

1. **STL usage:** OK dùng `std::vector`/`std::string`/`std::variant` cho infrastructure, NHƯNG `BTree`/`HashIndex`/`Fenwick` implement **manual** (showcase DSA). ✅ đề xuất
2. **In-place vs alongside:** Tôi sẽ refactor in-place (xóa parallel arrays cũ). Migration tool đọc data cũ trước khi xóa, đảm bảo không mất dữ liệu seed/test. ✅ đề xuất
3. **Session orders (in-memory only):** vẫn lưu trong RAM Table (không persist xuống .tbl) — reset mỗi ca. ✅ đề xuất
4. **CLI/IPC:** giao diện React Ink không thay đổi (vẫn ăn JSON cũ qua stdin/stdout). ✅ đề xuất
