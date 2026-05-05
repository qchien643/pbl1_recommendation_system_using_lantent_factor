# 07 · UX CLI Design

Nguồn: [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §2, §15.

## Nguyên tắc tối thượng (BR16)

> Mọi thao tác nhập liệu chỉ dùng **số** và **mã ASCII không dấu**. Người dùng **không bao giờ** cần gõ tiếng Việt có dấu.

Output (text hiển thị) có thể chứa Vietnamese thuần — chỉ input là bị ràng buộc.

## Chính sách ngôn ngữ hiển thị

| Tầng | UI | Ngôn ngữ | Lý do |
|---|---|---|---|
| **Server (thu ngân)** | [server_dashboard.mjs](../../cli/src/server_dashboard.mjs), [ServerApp.jsx](../../cli/src/ServerApp.jsx) | **English** | Dành cho nhân viên / dev — technical, ngắn gọn (`Session OPEN`, `Revenue`, `Table 05 SUBMIT #7`) |
| **Client (bàn khách)** | [ClientApp.jsx](../../cli/src/ClientApp.jsx) + `components/*.jsx` | **Tiếng Việt có dấu** | Dành cho khách — thân thiện (`Chào mừng trở lại`, `Hóa đơn`, `Cảm ơn quý khách!`) |

**Input vẫn ASCII only** (BR16):
- SĐT = chỉ chữ số
- Mã món = `[PBCGADT][0-9][0-9]` 3 ký tự ASCII
- Mã ca = chỉ chữ số
- **Tên khách (NameInput)** = ASCII printable (32–126), reject Unicode multi-byte để lưu vào `char userName[40]` trong users.dat không bị corrupt. Helper text rõ `"2-35 ký tự ASCII (không dấu)"`.

## Đối chiếu input design cũ vs mới

| Tính năng | Cũ (cần tiếng Việt) | Mới (chỉ số / mã) |
|---|---|---|
| Chọn món | Nhập tên món "Phở Bò" | Nhập **mã mon** `P01` |
| Xác thực khách | Nhập tên khách "Nguyễn Văn A" | Nhập **SDT 10 chữ số** |
| Kết thúc chọn món | Nhập "xong" | Nhập **`00`** hoặc Enter trắng |
| Mở / đóng ca | Nhập mã giao dịch dạng text | Nhập **mã số** (vd `1234`) |
| Chọn tùy chọn | Đọc + nhập text | Chọn **số thứ tự** 1/2/3 |
| Xác nhận hóa đơn | Nhập "có" / "yes" | Nhấn **`Y`** hoặc **Enter** |

## Mockup #1 — Server Dashboard (blessed-contrib, English)

Layout 12×12 grid trong [server_dashboard.mjs](../../cli/src/server_dashboard.mjs):

```
┌ Viet Phong Server ──────────┬─ Session ────────────┐
│ VIET PHONG RESTAURANT       │ ● SESSION OPEN       │
│ Server ● READY  · port 8888 │ Code: 1234           │
│ Time: 2026-04-23 22:40:12   │ Started: 22:38       │
├ Orders Today ─ Revenue ─────┼─ Overall Stats ──────┤
│  ██  ██   ██.█ K VND        │ Clients:  3/20       │
│  ██ ███                     │ Users:    13         │
│                             │ LFM sugg: 18/24 (75%)│
├ Activity Log [Tab→Customers]┴──────────────────────┤
│ 22:40 Server ready · 11 items · 10 users saved     │
│ 22:40 Session OPENED code=1234                     │
│ 22:41 Table 05 login 0901234567 (22 prior orders)  │
│ 22:42 Table 05 SUBMIT #1 · 145.000d                │
└────────────────────────────────────────────────────┘
```

**Tab** → chuyển sang Customers panel (cùng vị trí row 6-11):

```
┌ Customers [↑↓ select] [Enter view] ┬ Transaction History ┐
│ Phone       Name          Orders   │ ◆ Anh Nam · 0901...  │
│ 0901234567  Anh Nam       22  ◀    │ ▲ 22 orders · Spent..│
│ 0912345678  Chi Lan       20       │ ─────────────────── │
│ 0923456789  Bac Hung      18       │ ★ History: 22 orders │
│ ...                                │   ◉ Order #1 · ...   │
└────────────────────────────────────┴──────────────────────┘
```

**Enter** trên 1 khách → focus chuyển sang Transaction History panel (cuộn bằng `↑↓/PgUp/PgDn/g/G`, `Esc/←` về lại danh sách).

Gate: `Tab` chỉ mở được Customers panel khi **session đang mở**; nếu chưa mở, log dòng đỏ `"Must open SESSION before viewing customer history"`.

Widget đã loại bỏ: `clientTable` (Bang khach), `heartbeatSparkline` (Heartbeat 60s) — cả 2 trùng hoặc không cần thiết.

## Mockup #2 — Server (Thu ngân, ASCII wireframe cũ)

```
+----------------------------------------------------------+
|  VIET PHONG RESTAURANT  —  CASHIER SERVER                |
+----------------------------------------------------------+
|  Server: 192.168.1.100:8888    [RUNNING]                 |
|  Clients: [Table 1: OK] [Table 2: OK] [Table 3: WAIT]    |
+----------------------------------------------------------+
|  Nhap MA SO de MO CA (chi so):                           |
|  > [____]                                                |
+----------------------------------------------------------+
|  Don hom nay: 12   Doanh thu: 4.520.000d                 |
|  Goi y LFM dung: 47 lan   Ti le chap nhan: 63%           |
+----------------------------------------------------------+
```

→ Component: `ServerApp.jsx` → compose `WaitingScreen.jsx` + stats panel.

## Mockup #2 — Client nhập SDT

```
+----------------------------------------------+
|   NHA HANG VIET PHONG  —  BAN 02             |
+----------------------------------------------+
|                                              |
|   Chao mung! Vui long nhap so dien thoai:    |
|                                              |
|   SDT (10 chu so):  > [__________]           |
|                                              |
|   Luu y: Chi nhap chu so, khong can go dau   |
|   Vi du: 0901234567                          |
|                                              |
+----------------------------------------------+
```

→ Component: `PhoneInput.jsx`.

**Validate inline:**
- Độ dài khác 10 → `"Loi: Can dung 10 chu so"`.
- Có ký tự không phải số → `"Loi: Chi nhap chu so"`.
- Không bắt đầu bằng `0` → `"Loi: SDT phai bat dau bang 0"`.

## Mockup #2b — Đăng ký khách mới (Name + Desc)

Khi `USER_ACK.isNew=true`, client chuyển sang `NameInput.jsx`:

```
╔══════════════════════════════════════════════════╗
║  ✦  KHACH MOI — VUI LONG CHO BIET TEN  ✦         ║
║                                                    ║
║  ◆ SDT: 0900000001                                 ║
║                                                    ║
║  ▶ Ten cua ban:                                    ║
║     Nguyen Van A_                                  ║
║     ▲ 2-35 ky tu ASCII (khong dau). Enter de tiep. ║
║                                                    ║
║  ▶ Mo ta ngan (tuy chon):                          ║
║     Dan van phong, thich Pho Bo_                   ║
║     ▲ Enter trong de bo qua.                       ║
╚══════════════════════════════════════════════════╝
```

→ Component: `NameInput.jsx`. Gửi `USER_REGISTER` khi xong.

**Validate inline:**
- Tên < 2 ký tự → `"Ten phai co it nhat 2 ky tu ASCII"`.
- Ký tự Unicode/tiếng Việt có dấu → reject tại input (BR16).
- Tên trống sau sanitize server-side → fallback `"Khach"`.

## Mockup #3 — Menu + Gợi ý (khách quen)

```
+----------------------------------------------+
|   NHA HANG VIET PHONG  —  BAN 02             |
+----------------------------------------------+
|   Chao mung tro lai! SDT: 0901234567         |
|   Ban da dat 7 lan. Mon yeu thich: Pho Bo    |
+----------------------------------------------+
|   MA MON  | TEN MON              | GIA       |
|   --------+----------------------+-----------|
|   P01     | Pho Bo Tai           | 65.000d   |
|   P02     | Pho Ga               | 55.000d   |
|   B01     | Bun Bo Hue           | 60.000d   |
|   C01     | Com Tam Suon Bi      | 75.000d   |
|   D01     | Tra Da               | 15.000d   |
|   D02     | Nuoc Ngot            | 20.000d   |
+----------------------------------------------+
|   GOI Y CHO BAN (dua tren lich su):          |
|   C01  Com Tam    ████████░  0.91            |
|   D01  Tra Da     ███████░░  0.85            |
|   T01  Che        █████░░░░  0.72            |
+----------------------------------------------+
|   Da chon: [P01 x1] [D01 x1]  Con lai: 3     |
|   Nhap: [MA MON] [SO LUONG]  00 = Xong       |
|   > [___] [_]                                |
+----------------------------------------------+
```

→ Compose: `MenuDisplay.jsx` + `SuggestPanel.jsx` + `OrderSummary.jsx`.

**SuggestPanel bar chart:** mỗi vị trí bar dùng `'█'` nếu `(i / total_chars) < score` else `'░'`. Ví dụ score 0.91 trên 9 ký tự → 8 ô `█` + 1 ô `░`.

## Mockup #4 — Hóa đơn

```
+--------------------------------------------------+
|              HOA DON — BAN 02                    |
|     Ma GD: 1234   23/04/2026 10:35               |
|     SDT: 0901234567                              |
+------+-----+------------+----+--------+----------+
| STT  | Ma  | Ten mon    | SL | Don gia| T.tien   |
+------+-----+------------+----+--------+----------+
|  1   | P01 | Pho Bo Tai |  2 | 65.000 | 130.000  |
|  2   | D01 | Tra Da     |  2 | 15.000 |  30.000  |
+------+-----+------------+----+--------+----------+
|                     Tam tinh:       160.000d     |
|                     Giam gia:             0d     |
|                     TONG CONG:     160.000d     |
+--------------------------------------------------+
|  Xac nhan gui len Server?  Y = Co / N = Sua lai  |
|  > [_]                                           |
+--------------------------------------------------+
```

→ Component: `Invoice.jsx`.

**Keys chấp nhận:**
- `Y` hoặc `Enter` → gửi `ORDER_SUBMIT`.
- `N` → quay lại trạng thái ORDERING để sửa.

## Mapping mockup → JSX component

| Mockup | Component | Tầng | Ngôn ngữ |
|---|---|---|---|
| Dashboard (default) | `cli/src/server_dashboard.mjs` (blessed-contrib) | Server | English |
| Dashboard (React Ink) | `cli/src/ServerApp.jsx` | Server | English |
| Mockup #2 | `WaitingScreen.jsx` | Client | Vietnamese có dấu |
| Mockup #2 | `PhoneInput.jsx` | Client | VN (hiển thị), ASCII (input) |
| Mockup #2b | `NameInput.jsx` (khách mới) | Client | VN (hiển thị), ASCII (input) |
| Mockup #3 | `MenuDisplay.jsx` + `SuggestPanel.jsx` + `OrderSummary.jsx` | Client | Vietnamese có dấu |
| Mockup #4 | `Invoice.jsx` | Client | Vietnamese có dấu |
| Tổng kết khách | `DailySummary.jsx` | Client | Vietnamese có dấu |

## Rule hiển thị tiền

Format: `1.234.567d` (dấu `.` phân cách ngàn, `d` thay cho `đ` để tránh Unicode trong một số terminal).

## IPC giữa Node.js (UI) và C++ core

Đề xuất: stdio JSON — mỗi dòng 1 JSON object.

**Up (UI → C++ core):**
```json
{"type":"USER_LOGIN","phone":"0901234567"}
{"type":"ITEM_ADDED","code":"P01","qty":1}
{"type":"ORDER_SUBMIT"}
```

**Down (C++ core → UI):**
```json
{"type":"MENU_DATA","items":[{"code":"P01","name":"Pho Bo Tai","price":65000},...]}
{"type":"USER_ACK","userId":5,"isNew":false,"orderCount":3}
{"type":"SUGGEST","items":[{"code":"C01","score":0.91},...]}
```

Module `cli/src/ipc.js`:
```js
export function send(obj) { process.stdout.write(JSON.stringify(obj) + '\n'); }
export function on(type, cb) { /* lắng nghe process.stdin, parse line, match type */ }
```
