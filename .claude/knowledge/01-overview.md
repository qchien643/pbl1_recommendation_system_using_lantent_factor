# 01 · Tổng quan dự án

Nguồn: [phan-tich-du-an-702.md](../../phan-tich-du-an-702.md) §1, §3, §4.1.

## Tóm tắt đề tài

| Thuộc tính | Nội dung |
|---|---|
| Tên | Ứng dụng đặt món ăn và thanh toán đơn hàng |
| Mã đề | 702 |
| Ngôn ngữ | C/C++ (core + socket) + React Ink (CLI UI) |
| Giao diện | Terminal / CMD — hạn chế tối đa nhập tiếng Việt |
| Lưu trữ | Parallel arrays + file `.txt` / `.dat` |
| Kiến trúc | TCP/IP — 1 Server (thu ngân) + N Client (bàn khách) |
| ML | **Latent Factor Model** (Matrix Factorization) gợi ý món cá nhân hóa |
| Xác thực | Số điện thoại 10 chữ số → `user_id` cho LFM |

## 3 Actors chính

| Actor | Vai trò | Máy |
|---|---|---|
| **Thu ngân** | Mở/đóng ca, xem thống kê, quản lý phiên | Máy Server |
| **Khách hàng** | Nhập SDT, xem menu, nhập mã món, nhận hóa đơn | Máy Client (bàn) |
| **Latent Factor Engine** | Gợi ý món dựa trên lịch sử đặt theo SDT | Chạy trên Server |

## Kiến trúc tổng thể (rút gọn)

```
┌─ SERVER (Thu ngân) ─────────────────────┐
│  React Ink UI ↔ C++ Core                │
│  ├─ Socket Server (TCP port 8888)        │
│  ├─ Session Manager                      │
│  ├─ LFM Engine (P, Q matrices)           │
│  ├─ Parallel Arrays: users + orders      │
│  └─ File I/O (.txt, .dat)                │
└────────────┬─────────────────────────────┘
             │ TCP/IP LAN
   ┌─────────┴──────────┬──────────────┐
   ▼                    ▼              ▼
CLIENT (Bàn 01)     CLIENT (Bàn 02)    ...
React Ink + C++     React Ink + C++
```

Xem diagram đầy đủ trong [10-architecture.md](10-architecture.md).

## Flow cốt lõi (rút gọn)

1. Thu ngân mở ca (nhập mã số, vd `1234`) → Server broadcast `START` + `MENU_DATA`.
2. Khách ngồi bàn → nhập SDT 10 số → Client gửi `USER_LOGIN`.
3. Server tra cứu SDT, tính LFM scores, gửi `SUGGEST` top-3.
4. Khách nhập mã món (P01, B02, ...) + số lượng. Max 5 món hoặc nhập `00` để kết thúc.
5. Client tính tổng, áp giảm giá 25% nếu ≥ 2.000.000đ, hiển thị hóa đơn.
6. Khách xác nhận `Y` → Client gửi `ORDER_SUBMIT` → Server online SGD update P, Q.
7. Thu ngân đóng ca (nhập lại mã số) → Server broadcast `STOP`, xuất báo cáo `.txt`, lưu P/Q `.dat`.
