# Restaurant Ordering System — LFM Recommendations

Hệ thống đặt món nhà hàng chạy trên **TCP LAN**, dùng **Latent Factor Model (Matrix Factorization)** để gợi ý món cá nhân hoá theo số điện thoại. Đề tài PBL1 — Đại học Bách khoa Đà Nẵng, đề 702.

```
┌─ Server (thu ngân) ──────┐             ┌─ Client (bàn khách) ─┐
│  Menu + Users + Orders   │  TCP 8888   │  Order Builder        │
│  Session Manager         │◄──────────►│  Input Handler         │
│  Latent Factor Engine    │             │  React Ink UI          │
│  File Reports + Models   │             └────────────────────────┘
└──────────────────────────┘
```

---

## 1. Yêu cầu môi trường

| Tool | Phiên bản tối thiểu | Dùng để |
|---|---|---|
| **C++ compiler** | `g++` 9+ hoặc MSVC với C++17 | Biên dịch server + client |
| **CMake** | 3.15+ | Build automation |
| **Node.js** | 18+ (đã test 24.x) | React Ink UI + demo scripts |
| **npm** | 10+ | Cài dependencies |
| **Python 3** | 3.8+ (tuỳ chọn) | Chạy `matrix_factorization.py` reference |

### Windows

Cài Git Bash + [MinGW-W64 (UCRT)](https://winlibs.com/) (đã bao gồm `g++`, `cmake`).
Thêm vào PATH: `C:\mingw64\bin` và Node.js installer path.

```bash
g++ --version        # >= 9
cmake --version      # >= 3.15
node --version       # >= 18
```

### Linux / macOS

```bash
sudo apt install build-essential cmake        # Ubuntu/Debian
# hoặc
brew install gcc cmake                        # macOS
```

Note: server dùng `std::thread`, trên Linux cần `-lpthread` (CMakeLists đã xử lý).

---

## 2. Build

Từ thư mục gốc dự án:

```bash
cmake -S . -B build -G "MinGW Makefiles"     # Windows (Git Bash + MinGW)
# hoặc
cmake -S . -B build                           # Linux / macOS / VS native
cmake --build build
```

Kết quả trong `build/`:

- `build/server.exe` — TCP server (thu ngân)
- `build/client.exe` — TCP client (bàn khách), có 2 chế độ: text + `--json`
- `build/seed_data.exe` — sinh dữ liệu mẫu (10 khách quen)
- `build/phase1_test.exe` — smoke test toàn bộ C++ core

(Trên Linux/macOS bỏ phần `.exe`.)

---

## 3. Cài React Ink UI

```bash
cd cli
npm install        # ~10s, cài ink + tsx + react + deps
cd ..
```

---

## 4. Chạy demo kịch bản (khuyến nghị chạy lần đầu)

Script đã chuẩn bị sẵn sẽ reset data → seed 10 khách quen → mở ca → cho 4 khách đặt món → đóng ca → in báo cáo:

```bash
bash tools/demo.sh
```

Kết quả mong đợi (~20 giây):

- KHÁCH 1 **Anh Nam** `0901234567` → chào khách quen, top-3 = Trà Đá + Phở Bò + Cơm Tấm
- KHÁCH 2 **Chị Lan** `0912345678` → top-3 khác hẳn = Nước Ngọt + Cơm Tấm + Gỏi Cuốn
- KHÁCH 3 SDT mới `0911111111` → cold-start, gợi ý theo global pattern
- KHÁCH 4 SDT mới `0938888888` → đơn lớn 2.795.000đ → tự giảm 25% = **2.096.250đ**
- Báo cáo `data/reports/report_YYYY-MM-DD.txt` sinh đầy đủ

---

## 5. Chạy thủ công (2 terminal)

### Bước 1: Sinh dữ liệu mẫu (chỉ cần lần đầu)

```bash
./build/seed_data.exe
cat data/personas.txt      # xem danh sách 10 SDT mẫu
```

### Bước 2: Terminal A — Server

Có 2 lựa chọn:

**Text mode (đơn giản):**
```bash
./build/server.exe --server
```
Server in `[Server] Ready. Nhap MA SO (1-9 chu so) de MO CA:`  
→ **gõ** `1234` + Enter để mở ca. Gõ lại `1234` để đóng ca.

**blessed-contrib Dashboard (khuyến nghị):**
```bash
cd cli
npm run server
```
Dashboard grid 12x12 có:
- **Header** + panel **Session** (mở/đóng bằng cách gõ số + Enter)
- 2 **LCD gauges** hiển thị số đơn và doanh thu (nghìn đồng) dạng LCD number
- Panel **Stats tổng** (clients, users, discount, LFM accept rate)
- Bảng **Bàng khách** (contrib.table): số bàn, SDT, trạng thái, heartbeat N giây trước
- **Heartbeat sparkline** (contrib.sparkline) cập nhật realtime 60s gần nhất
- **Activity log** (contrib.log) scrollable với 30 dòng, màu theo loại event

Phím: `0-9` nhập mã số ca, `Enter` xác nhận, `Backspace` xoá, `Esc`/`Ctrl-C` thoát.

Nếu muốn bản React Ink (đơn giản hơn, ít khung):
```bash
npm run server-ink
```

### Bước 3: Terminal B — Client (3 lựa chọn)

#### 3a. Client text-mode (đơn giản nhất)

```bash
./build/client.exe --client 127.0.0.1 1
```

Làm theo prompt:
```
Vui long nhap so dien thoai (10 chu so):
> 0901234567

Nhap MA MON va SO LUONG (VD: P01 2) - 00/Enter = Xong
> P01 1
> D01 1
> 00

Xac nhan gui len Server? (Y=Co / N=Huy) [Y]:
> Y
```

#### 3b. Client React Ink (giao diện terminal đẹp — khuyến nghị)

```bash
cd cli
npm start                          # mặc định 127.0.0.1, bàn 1
npm start -- 192.168.1.100 2       # IP khác, bàn 2
```

Giao diện có:
- Header **"Ban 02"** với gradient cristal big text + status bar màu thay đổi theo state
- **PhoneInput** với 10 ô `[_][_]...` nhấp nháy, validate inline, gợi ý SDT mẫu
- **MenuDisplay** màu theo nhóm món (Phở đỏ, Bún tím, Cơm vàng, Đồ uống cyan...), món đã chọn gạch ngang
- **SuggestPanel** với medal 🥇🥈🥉 + bar chart `█████░░░░░` co giãn theo score
- **OrderSummary** thời gian thực: subtotal tính ngay khi thêm món, báo "se duoc giam 25%" khi chạm ngưỡng
- **Invoice** border double cyan, bấm `Y`/`Enter` xác nhận hoặc `N` sửa
- **DailySummary** (sau confirm): "✓ CAM ON QUY KHACH!" với countdown 3s auto-next

**Lưu ý:** React Ink cần terminal thật có TTY. Không chạy được qua pipe/redirect.

#### 3c. Client JSON-mode (cho tích hợp / test)

```bash
./build/client.exe --client 127.0.0.1 1 --json
```

Gõ vào JSON command, mỗi dòng 1 object:
```json
{"cmd":"login","phone":"0901234567"}
{"cmd":"add_item","code":"P01","qty":1}
{"cmd":"finish"}
{"cmd":"confirm"}
{"cmd":"quit"}
```

Server events trả về cũng là JSON, 1 dòng/object. Hữu ích cho automated test.

### Nhiều bàn khách đồng thời — trên cùng 1 máy

Không cần Docker hay VM. TCP socket trên `127.0.0.1` hoạt động giống hệt như mạng LAN thật, chỉ cần **nhiều terminal windows riêng**.

#### Cách 1A — Script tự động UI mode (ĐẸP NHẤT)

```cmd
tools\launch_ui.bat 3
```

Mở **1 SERVER Dashboard (React Ink)** + **3 Client UI (React Ink)**, mỗi cửa sổ là một "máy bàn" độc lập với giao diện màu đẹp. Ở Server Dashboard gõ `1234` + Enter để mở ca → các client tự động nhận START và prompt nhập SDT.

#### Cách 1B — Script tự động text mode (nhẹ hơn)

```cmd
tools\launch_multi.bat 3
```

Mở 1 server + 3 client ở **text mode** (không cần React Ink). Phù hợp nếu muốn chạy nhiều terminal cùng lúc mà CPU yếu.

Hoặc cross-platform (Git Bash):
```bash
bash tools/launch_multi.sh 3
```

#### Cách 2 — Thủ công

Mở **4 cửa sổ Git Bash** (hoặc Windows Terminal tabs):

| Window | Lệnh | Vai trò |
|---|---|---|
| 1 | `./build/server.exe --server` | Máy thu ngân |
| 2 | `./build/client.exe --client 127.0.0.1 1` | Bàn 1 |
| 3 | `./build/client.exe --client 127.0.0.1 2` | Bàn 2 |
| 4 | `./build/client.exe --client 127.0.0.1 3` | Bàn 3 |

Hoặc dùng React Ink UI (mỗi bàn 1 tab):
```bash
cd cli
npm start -- 127.0.0.1 1    # Tab 1
npm start -- 127.0.0.1 2    # Tab 2
npm start -- 127.0.0.1 3    # Tab 3
```

Server hỗ trợ tối đa **20 client đồng thời** (`MAX_CLIENTS` trong `shared/constants.h`).

#### Cách 3 — Mạng LAN thật giữa nhiều máy

Máy bạn làm server, máy khác làm client:

**Máy A (server):**
```bash
# Biết IP LAN của máy A (vd 192.168.1.100)
ipconfig                              # Windows
# hoac: hostname -I                   # Linux
./build/server.exe --server
# Mo Windows Firewall: cho phep TCP port 8888 inbound
```

**Máy B, C, D... (client):**
```bash
./build/client.exe --client 192.168.1.100 2
# Hoac React Ink:
cd cli && npm start -- 192.168.1.100 2
```

Cả A và B đều phải build xong binary + cùng mạng LAN. Firewall Windows có thể chặn lần đầu → cho phép `server.exe` nhận kết nối.

#### Có cần Docker không?

**Không.** Dự án này chỉ cần TCP socket — dùng `127.0.0.1` với nhiều terminal là đủ mô phỏng. Docker chỉ hữu ích nếu:
- Muốn mỗi client có IP riêng trong network isolated (vd `172.20.0.2`, `172.20.0.3`) để test reconnection/ACL scenarios.
- Triển khai thật ra cloud / production.
- Muốn đóng gói để người khác chạy mà không cần cài g++/CMake.

Cho demo PBL1 và test đa client cục bộ, terminal windows là giải pháp đơn giản + chính xác nhất.

---

## 6. SDT mẫu để thử

Sau khi chạy `seed_data.exe`, có 10 khách có sẵn. Xem chi tiết trong [`data/personas.txt`](data/personas.txt):

| SDT | Khách | Gợi ý sẽ thấy |
|---|---|---|
| `0901234567` | Anh Nam (văn phòng) | Trà Đá + Phở Bò + Cơm Tấm |
| `0912345678` | Chị Lan (sinh viên) | Nước Ngọt + Cơm Tấm + Gỏi Cuốn |
| `0923456789` | Bác Hùng (cuối tuần) | Bún Bò + Chè + Bún Riêu |
| `0956789012` | Cô Tư (một mình) | Phở Gà + Chè + Nước Ngọt |
| `0989012345` | Bác Sáu (miền Tây) | Cơm Tấm + Chả Giò + Trà Đá |
| `0990123456` | Anh Khoa (cold-start) | Phở Bò + Trà Đá (điểm thấp) |
| bất kỳ SDT 10 số khác | (khách mới) | Cold-start, score nhỏ |

Menu codes: `P01`/`P02`=Phở, `B01`/`B02`=Bún, `C01`/`C02`=Cơm, `G01`=Gỏi Cuốn, `A01`=Chả Giò, `D01`/`D02`=Đồ Uống, `T01`=Chè.

---

## 7. Lịch sử giao dịch — `data/transactions.log`

Mỗi đơn khi được `ORDER_SUBMIT` sẽ được **append** vào [`data/transactions.log`](data/transactions.log) — một dòng/đơn, format:

```
TIMESTAMP|SESSION_CODE|PHONE|CODE1,QTY1|...|CODEn,QTYn|SUBTOTAL|DISCOUNT|TOTAL
```

Ví dụ:
```
2026-04-23 19:00|1234|0901234567|P01,2|D01,1|145000|0|145000
2026-04-23 19:12|1234|0938888888|A01,20|C01,5|G01,10|2525000|631250|1893750
```

Khác với `orderHistory[u][i]` (chỉ là số đếm cho LFM, không nhớ thứ tự), file này **persist xuyên ca** và giữ chi tiết từng lần mua (món gì, bao nhiêu, tổng bao nhiêu). Không lưu tên khách — chỉ SDT + món.

Dùng để:
- Lookup khách A tháng trước mua gì: `grep "^.*|.*|0901234567|" data/transactions.log`
- Thống kê món bán chạy theo ngày: `awk -F'|' '$1 ~ /2026-04-23/' data/transactions.log`
- Phân tích xu hướng / báo cáo định kỳ

## 8. Reset về "nhà hàng tươi"

```bash
rm -f data/lfm_*.dat data/users.dat data/personas.txt data/reports/*.txt data/transactions.log
./build/seed_data.exe
```

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Khắc phục |
|---|---|---|
| `bind failed` / `Address already in use` | Port 8888 đang bị chiếm (server trước chưa đóng) | `taskkill /F /IM server.exe` (Windows) hoặc `pkill -f server.exe` |
| `Cannot connect` (client) | Server chưa start hoặc firewall | Đảm bảo server chạy trước 1-2 giây, mở port 8888 trong firewall |
| `Raw mode is not supported` (React Ink) | Terminal không phải TTY | Chạy trong terminal thật (PowerShell / Git Bash trực tiếp), không qua pipe |
| Gợi ý toàn ra C01/C02/G01 giống nhau | Dùng init cũ [0, 0.01] positive-only | Đã fix trong Phase 4 — rebuild lại: `cmake --build build --clean-first` |
| `seed_data: Khong mo duoc data/menu.txt` | Chạy sai thư mục | Phải chạy từ thư mục gốc dự án |
| Node `Cannot find package 'tsx'` | Chưa `npm install` trong cli/ | `cd cli && npm install` |

---

## 9. Cấu trúc dự án

```
├── CLAUDE.md                    ← Overview + index cho Claude Code
├── README.md                    ← (file này)
├── CMakeLists.txt               ← Build config
├── phan-tich-du-an-702.md      ← Đặc tả đề tài gốc (1117 dòng)
├── matrix_factorization.py     ← LFM reference Python
│
├── shared/    ← C++ code dùng chung (protocol, state, net, json, utils)
├── server/    ← TCP server + session + LFM + file manager
├── client/    ← TCP client + order builder + display
├── cli/       ← React Ink UI Node.js (JSX) + ipc.js
├── tools/     ← seed_data.cpp + demo.sh + pretty_event.mjs
├── data/      ← menu.txt + users.dat + lfm_*.dat + reports/
├── tests/     ← phase1_test.cpp smoke test
└── docs/      ← ML_ENGINE_DESIGN.md (tài liệu thiết kế ML engine)
```

---

## 10. Tài liệu sâu hơn

- [phan-tich-du-an-702.md](phan-tich-du-an-702.md) — đặc tả gốc: kiến trúc, business rules, protocol, LFM
- [docs/ML_ENGINE_DESIGN.md](docs/ML_ENGINE_DESIGN.md) — thiết kế chi tiết Matrix Factorization (vòng đời, 3 luồng cập nhật, early stopping)
- [matrix_factorization.py](matrix_factorization.py) — implementation LFM bằng Python (reference cho C++ port)
- [.claude/knowledge/](.claude/knowledge/) — knowledge base chia theo chủ đề cho Claude Code subagents
- [CLAUDE.md](CLAUDE.md) — overview dành cho Claude Code / AI assistants

---

## 11. Commands cheat-sheet

```bash
# Build tất cả
cmake -S . -B build -G "MinGW Makefiles" && cmake --build build

# Seed + demo nhanh
./build/seed_data.exe && bash tools/demo.sh

# Smoke test C++ core (không cần mạng)
./build/phase1_test.exe

# Smoke test IPC (cần server đang chạy)
node cli/src/smoke_test.js

# Verify gợi ý seed (cần server đang chạy)
node cli/src/smoke_seed.js

# Python reference (demo 5 bước update model)
python matrix_factorization.py

# React Ink UI thực sự (cần TTY)
cd cli && npm start
```
