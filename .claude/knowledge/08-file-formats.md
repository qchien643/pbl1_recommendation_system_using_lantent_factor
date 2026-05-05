# 08 · File Formats

Nguồn: [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §16.

## Báo cáo cuối ca — `data/reports/report_YYYY-MM-DD.txt`

Text thuần, encoding UTF-8 hoặc ASCII-safe (không dấu trong phần header / tên món → dễ print). Mỗi ca 1 file.

### Template đầy đủ

```
==============================================================
   BAO CAO NGAY 23/04/2026
   Ma giao dich : 1234
   Ca lam viec  : 07:00 - 22:00
   So may ban   : 3
==============================================================

DON #001 | Ban 02 | SDT: 0901234567 | 09:15
--------------------------------------------------------------
  P01  Pho Bo Tai    x2   65.000d  =  130.000d
  D01  Tra Da        x2   15.000d  =   30.000d
  Tam tinh: 160.000d | Giam: 0d | Tong: 160.000d
  [Goi y LFM duoc dung: D01]

DON #002 | Ban 01 | SDT: 0912345678 | 11:30
--------------------------------------------------------------
  C01  Com Tam Suon Bi   x5   75.000d  =  375.000d
  A01  Cha Gio           x20  80.000d  = 1.600.000d
  G01  Goi Cuon          x10  55.000d  =   550.000d
  Tam tinh: 2.525.000d | Giam 25%: 631.250d | Tong: 1.893.750d
  [Goi y LFM duoc dung: A01, G01]

==============================================================
TONG KET NGAY
  Tong so don        : 12
  Tong doanh thu     : 8.340.000d
  Tong giam gia      : 1.240.000d
  Don duoc giam      : 3 / 12
  So SDT khac nhau   : 9
  Goi y LFM su dung  : 47 / 75 (63%)
  Mon ban chay       : P01 (28 lan), C01 (19 lan), D01 (17 lan)
==============================================================
LFM MODEL STATS
  Tong users da hoc  : 42
  Latent dimensions  : K=10
  Online updates     : 12 (ca nay)
==============================================================
```

### Quy ước format

- **Tiền:** `1.234.567d` (dấu `.` ngăn ngàn, `d` thay cho `đ`).
- **Số lượng:** `x{qty}` ngay sau mã món. Padding 3 chữ số (vd `x20`).
- **Thời gian:** `HH:MM` cho mỗi đơn, `DD/MM/YYYY` cho ngày.
- **Mã gợi ý được dùng:** dòng `[Goi y LFM duoc dung: ...]` — liệt kê các mã trong đơn mà cũng có trong `SUGGEST` gần nhất.

### Thuật toán tính "tỉ lệ chấp nhận gợi ý"

```
acceptRate = (số món trong tất cả đơn mà xuất hiện trong SUGGEST tại thời điểm chọn)
           / (tổng số món được gợi ý trong ca)
```

Log mỗi lần gửi `SUGGEST` vào buffer + mỗi lần nhận `ITEM_ADDED` check xem code có trong SUGGEST gần nhất không.

## Menu — `data/menu.txt`

Text, mỗi dòng 1 món:

```
CODE|NAME|PRICE|CATEGORY
```

```
P01|Pho Bo Tai|65000|P
P02|Pho Ga|55000|P
B01|Bun Bo Hue|60000|B
B02|Bun Rieu|55000|B
C01|Com Tam Suon Bi|75000|C
C02|Com Chien Duong Chau|65000|C
G01|Goi Cuon (5 cuon)|55000|G
A01|Cha Gio (10 cai)|80000|A
D01|Tra Da|15000|D
D02|Nuoc Ngot|20000|D
T01|Che Ba Mau|25000|T
```

Comment: dòng bắt đầu bằng `#` bỏ qua.

## Users — `data/users.dat` (binary)

Layout (format MỚI — không còn orderHistory, thêm name + desc):

```
[ int32 userCount ]
[ char[11]  userPhone[userCount]       ]   // "0901234567\0"
[ char[40]  userName[userCount]        ]   // "Nguyen Van A\0" (rỗng nếu chưa đăng ký)
[ char[80]  userDesc[userCount]        ]   // mô tả tùy chọn, có thể rỗng
[ int32     userTotalOrders[userCount] ]
```

`orderHistory[][]` giờ được derive từ `transactions.dat` qua `rebuildOrderHistory()` ở startup.

## Transactions — `data/transactions.dat` (binary, MỚI)

Source of truth cho per-order history. Persistent xuyên ca.

```
[ int32 txnCount ]
[ int32    txnUserIdx[txnCount]                        ]
[ char[20] txnTime[txnCount]                           ]   // "YYYY-MM-DD HH:MM:SS"
[ char[10] txnSessionCode[txnCount]                    ]
[ int32    txnItemCount[txnCount]                      ]
[ char[4]  txnItemCode[txnCount][MAX_ITEMS]            ]   // MAX_ITEMS=5
[ int32    txnItemQty [txnCount][MAX_ITEMS]            ]
[ float    txnSubtotal[txnCount]                       ]
[ float    txnDiscount[txnCount]                       ]
[ float    txnTotal[txnCount]                          ]
```

**Tạo / load:** [server/transaction_store.cpp](../../server/transaction_store.cpp). `appendTransaction()` push 1 txn; `saveTransactions()` / `loadTransactions()` xử lý binary.

## Transactions log — `data/transactions.log` (text, audit trail)

Append-only text, human-readable, giữ song song với `transactions.dat`. Format mỗi dòng:

```
TIMESTAMP|SESSION_CODE|PHONE|CODE1,QTY1|...|CODEn,QTYn|SUBTOTAL|DISCOUNT|TOTAL
```

Dùng để dev debug / inspect bằng mắt. Dashboard server có thể parse (đơn giản hơn binary) hoặc đọc `transactions.dat`.

## Nhịp độ ghi file (runtime)

Để dashboard đọc được dữ liệu **live** trong phiên, không phải đợi đóng ca:

| Sự kiện | File được ghi ngay |
|---|---|
| `USER_REGISTER` thành công | `users.dat` (name + desc) |
| `ORDER_SUBMIT` thành công | `users.dat` (totalOrders++) + `transactions.dat` + `transactions.log` |
| `closeSession` | `users.dat` + `transactions.dat` + `lfm_P/Q.dat` + `report_*.txt` |
| `openSession` | (nothing — startup data đã load từ đĩa ở process start) |

Xem [server/socket_server.cpp](../../server/socket_server.cpp) `handleOrderSubmit` (gọi `saveTransactions + saveUsers` ngay sau `lfmOnlineUpdate`) và `handleUserRegister` (gọi `saveUsers` sau `setUserName`).

## Tool đọc file binary

[tools/dump_data.mjs](../../tools/dump_data.mjs) — Node.js script parse `users.dat` + `transactions.dat` + `menu.txt`, in ra text human-readable group-by-user (giống `personas.txt` nhưng phản ánh state runtime hiện tại).

```bash
# Từ gốc project:
node tools/dump_data.mjs                        # in ra console
node tools/dump_data.mjs > data/snapshot.txt    # lưu file
```

Khác với `personas.txt`:
- `personas.txt` là **snapshot tạo 1 lần** bởi `seed_data.exe` — không cập nhật runtime.
- `dump_data.mjs` đọc **realtime từ `.dat`** → thấy cả user đăng ký sau seed + đơn đặt runtime.

## LFM models — `data/lfm_P.dat`, `data/lfm_Q.dat` (binary)

`lfm_P.dat`:
```
[ int32 userCount ]
[ int32 K         ]
[ float P[userCount][K]  ]   // row-major
```

`lfm_Q.dat`:
```
[ int32 menuCount ]
[ int32 K         ]
[ float Q[menuCount][K]  ]
```

**Load khi mở ca:**
1. Nếu file không tồn tại → init `P[][] = rand(0, 0.01)`, `Q[][] = rand(0, 0.01)`.
2. Nếu `K` trong file khác `K` hiện tại → bỏ file cũ, init lại.
3. Nếu `userCount` / `menuCount` trong file nhỏ hơn hiện tại → load phần có sẵn, phần dư init ngẫu nhiên + chạy `lfm_add_user()` / `lfm_add_item()`.

**Save khi đóng ca:**
- Ghi atomic: ghi ra `.tmp` → rename → tránh mất dữ liệu nếu crash giữa chừng.

## Naming file báo cáo

- Một ca / ngày: `report_2026-04-23.txt`.
- Hai ca cùng ngày: `report_2026-04-23_01.txt`, `report_2026-04-23_02.txt` (suffix `_NN` theo thứ tự).
- Trong thư mục [data/reports/](../../data/reports/).
