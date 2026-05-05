# PBL1 — Restaurant Ordering System with Latent Factor Model

**Đề 702 · DUT · C/C++ + React Ink · TCP LAN · Personalized Menu Recommendations**

Hệ thống đặt món nhà hàng chạy trên LAN: 1 máy **Server** (thu ngân) + N máy **Client** (bàn khách) giao tiếp qua TCP socket (port 8888). Khách nhập **số điện thoại (10 chữ số)** để xác thực → dùng làm `user_id` trong **Latent Factor Model (Matrix Factorization)** để gợi ý món cá nhân hóa theo lịch sử. UI terminal dựng bằng **React Ink**. Core logic / socket / ML port **C++ thuần** với **parallel arrays** (không OOP nặng — ràng buộc đề bài).

**Source of truth:** [phan-tich-du-an-702.md](phan-tich-du-an-702.md) — đặc tả đầy đủ (1117 dòng). File này là overview; chi tiết nằm trong `.claude/knowledge/`.

---

## Stack công nghệ

| Layer | Tech | Ghi chú |
|---|---|---|
| Core + socket | C/C++ 17 + Winsock2 | Platform target: Windows |
| ML (reference) | Python (numpy) | [matrix_factorization.py](matrix_factorization.py) — đã hoàn tất, là reference cho port C++ |
| ML (target) | C++ thuần | K=10, LR=0.01, REG=0.02, MAX_ITER=50 |
| CLI UI | React Ink (Node.js) | Chạy song song C++ core, giao tiếp qua stdio/pipe |
| Build | CMake + Node.js | |
| Lưu trữ | Parallel arrays + file `.txt` / `.dat` | Không dùng DBMS |

---

## Knowledge base — đọc file nào khi?

Toàn bộ ngữ nghĩa dự án được chia nhỏ trong [.claude/knowledge/](.claude/knowledge/). **Chỉ đọc file bạn cần** — không load toàn bộ.

| File | Dùng khi |
|---|---|
| [00-index.md](.claude/knowledge/00-index.md) | Map chủ đề → file KB. Đọc trước tiên nếu không chắc file nào |
| [01-overview.md](.claude/knowledge/01-overview.md) | Hỏi "dự án là gì, có những actor nào" |
| [02-business-rules.md](.claude/knowledge/02-business-rules.md) | 16 business rules — mở/đóng ca, giảm giá, xác thực SDT, mạng, LFM, UX |
| [03-menu-codes.md](.claude/knowledge/03-menu-codes.md) | Bảng mã món `P01`, `B01`, …; validate code |
| [04-network-protocol.md](.claude/knowledge/04-network-protocol.md) | Format message, 10 lệnh, sequence diagram, parser skeleton |
| [05-lfm-algorithm.md](.claude/knowledge/05-lfm-algorithm.md) | LFM: P, Q, SGD, online update, top-K, mapping Python → C++ |
| [06-data-structures.md](.claude/knowledge/06-data-structures.md) | Khai báo parallel arrays C++, ER diagram |
| [07-ux-cli-design.md](.claude/knowledge/07-ux-cli-design.md) | UX CLI mockups, nguyên tắc **không nhập tiếng Việt có dấu** |
| [08-file-formats.md](.claude/knowledge/08-file-formats.md) | Format `report_YYYY-MM-DD.txt`, layout binary `lfm_P.dat` / `lfm_Q.dat` |
| [09-state-machines.md](.claude/knowledge/09-state-machines.md) | State client + server |
| [10-architecture.md](.claude/knowledge/10-architecture.md) | System diagram tổng thể, cây thư mục dự án |

---

## Subagents chuyên biệt

Khi task chỉ liên quan một lĩnh vực, gọi subagent tương ứng để tiết kiệm context:

| Agent | Gọi khi |
|---|---|
| [lfm-expert](.claude/agents/lfm-expert.md) | Làm việc với LFM: train, SGD online update, thêm user/item, port Python → C++ |
| [cpp-socket-expert](.claude/agents/cpp-socket-expert.md) | Viết/debug TCP server-client C++, parser protocol, Winsock2 |
| [react-ink-ui-expert](.claude/agents/react-ink-ui-expert.md) | Xây CLI UI bằng React Ink, component mockup theo §15 |

---

## Quy ước code bắt buộc

1. **Parallel arrays** cho mọi dữ liệu — không dùng `struct` lồng sâu, không `std::vector` / STL nặng. Xem [06-data-structures.md](.claude/knowledge/06-data-structures.md).
2. **Input chỉ số + ASCII không dấu** (BR16) — mọi UI prompt phải tuân thủ. Xem [07-ux-cli-design.md](.claude/knowledge/07-ux-cli-design.md).
3. **Ngôn ngữ hiển thị:** Server dashboard → **English** (staff-facing); Client UI → **Tiếng Việt có dấu** (customer-facing). Input vẫn ASCII only bất kể UI.
4. **Mã món 3 ký tự** `[Prefix][2 chữ số]` (P01, B02, …). Prefix: P=Phở, B=Bún, C=Cơm, G=Gỏi, A=Ăn vặt, D=Đồ uống, T=Tráng miệng.
5. **SDT 10 chữ số bắt đầu bằng 0** — là `user_id` trong LFM, xuyên ca.
6. **Protocol message:** `[LOAI]|[NOI_DUNG]\n` — delimiter `|`, terminator `\n`. Xem [04-network-protocol.md](.claude/knowledge/04-network-protocol.md). Session gate server-side: mọi handler reject khi `!isSessionOpen()`.
7. **Port mặc định:** 8888.
8. **Persist-on-order:** `saveTransactions + saveUsers` chạy ngay sau mỗi `ORDER_SUBMIT` để dashboard đọc live. Xem [08-file-formats.md](.claude/knowledge/08-file-formats.md).

---

## Cấu trúc dự án

Xem [10-architecture.md](.claude/knowledge/10-architecture.md) cho cây đầy đủ. Tóm tắt:

- [server/](server/) — C++ entry `--server`, TCP listener, LFM engine, file I/O
- [client/](client/) — C++ entry `--client <IP>`, order builder, input handler
- [shared/](shared/) — `protocol.h`, `menu_item.h`, `utils.h`
- [cli/](cli/) — React Ink UI (Node.js) wrapper cho server + client
- [data/](data/) — `menu.txt`, `users.dat`, `lfm_P.dat`, `lfm_Q.dat`, `reports/`
- [matrix_factorization.py](matrix_factorization.py) — LFM reference implementation (Python)
- [docs/ML_ENGINE_DESIGN.md](docs/ML_ENGINE_DESIGN.md) — thiết kế ML engine (vòng đời, early stopping, 3 luồng update)
- [README.md](README.md) — hướng dẫn build + run dự án

---

## Commands thường dùng

```bash
python matrix_factorization.py      # Chạy demo Python LFM reference
cmake -S . -B build                 # Configure C++ (khi có code)
cmake --build build                 # Build server + client
cd cli && npm install && npm run dev  # Chạy UI React Ink (khi có code)
```
