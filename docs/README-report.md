# Báo cáo đồ án PBL1 — Hướng dẫn build và customize

File `report.docx` đã được sinh tự động bằng Python + `python-docx`, bám sát mẫu của Khoa CNTT — ĐHBK ĐN (Times New Roman 13, line-spacing 1.3, justified).

## Cấu trúc thư mục `docs/`

```
docs/
├── KhoaCNTT - Mau baocaoDoAn tinh toan 2020-2021.docx  # Mẫu gốc (tham khảo)
├── ML_ENGINE_DESIGN.md                                  # Tài liệu thiết kế ML
├── report.docx                                          # ★ BÁO CÁO HOÀN CHỈNH
│
├── generate_report.py        # Sinh report.docx từ snippets + figures
├── render_diagrams.py        # Mermaid (.mmd) → PNG qua Kroki API
├── capture_screens.py        # Chụp text output (seed, dump, e2e, server log)
├── mockup_tui.py             # Render mockup ASCII cho các màn TUI
│
├── diagrams/                 # Mermaid source (.mmd) — edit tại đây nếu cần
│   ├── arch-overview.mmd
│   ├── er-diagram.mmd
│   ├── seq-order.mmd
│   └── state-client.mmd
│
├── figures/                  # PNG được sinh bởi 3 script trên
│   ├── arch-overview.png         (Hình 1.1 — kiến trúc tổng thể)
│   ├── er-diagram.png            (Hình 3.1 — sơ đồ ER)
│   ├── state-client.png          (Hình 3.2 — state machine client)
│   ├── seq-order.png             (Hình 3.3 — sequence diagram)
│   ├── srv-startup.png           (server boot log)
│   ├── seed-output.png           (Hình 4.9 — seed_data output)
│   ├── dump-data.png             (Hình 4.11 — dump_data output)
│   ├── e2e-pass.png              (Hình 4.10 — 4/4 test PASS)
│   ├── personas-txt.png          (data/personas.txt excerpt)
│   ├── transactions-log.png      (data/transactions.log excerpt)
│   ├── cli-waiting.png      ★    (Hình 4.3 — màn chờ ca)         [MOCKUP]
│   ├── cli-phone.png        ★    (Hình 4.4 — nhập SĐT)            [MOCKUP]
│   ├── cli-name.png         ★    (Hình 4.5 — đăng ký tên)         [MOCKUP]
│   ├── cli-ordering.png     ★    (Hình 4.6 — màn đặt món)         [MOCKUP]
│   ├── cli-invoice.png      ★    (Hình 4.7 — hoá đơn)             [MOCKUP]
│   ├── cli-thanks.png       ★    (Hình 4.8 — cảm ơn)              [MOCKUP]
│   ├── srv-dashboard-session.png ★ (Hình 4.1 — dashboard log)     [MOCKUP]
│   └── srv-dashboard-customers.png ★ (Hình 4.2 — customers panel) [MOCKUP]
│
└── snippets/                 # Code excerpts cho phụ lục
    ├── parallel-arrays.cpp
    ├── lfm-train.cpp
    ├── online-update.cpp
    ├── top-k.cpp
    ├── handle-order-submit.cpp
    └── client-app-state.jsx
```

`★` = mockup ASCII, **nên replace** bằng screenshot thật.

---

## Quick start: chạy lại từ đầu

```powershell
# Từ gốc dự án
cd e:\code\project\DUT_PBL1\pbl1_recommendation_system_using_lantent_factor

# Tạo venv + cài deps (chỉ chạy 1 lần)
python -m venv .venv-report
.venv-report\Scripts\pip install python-docx requests Pillow

# Step 1: Render Mermaid diagrams → PNG (qua Kroki API, cần Internet)
.venv-report\Scripts\python docs\render_diagrams.py

# Step 2: Capture text outputs (seed_data, dump_data, e2e, server log)
.venv-report\Scripts\python docs\capture_screens.py

# Step 3: Render mockup TUI cho client + dashboard
.venv-report\Scripts\python docs\mockup_tui.py

# Step 4: Build báo cáo
.venv-report\Scripts\python docs\generate_report.py
# → output: docs/report.docx
```

Mở `docs/report.docx` bằng MS Word / LibreOffice / Google Docs để xem.

---

## Customize trước khi nộp

### 1. Điền thông tin trang bìa

Mở `docs/report.docx` trong Word, sửa các placeholder:
- `[Tên giảng viên hướng dẫn]`
- `[Họ và tên sinh viên]`
- `[Mã lớp]`
- `[Số nhóm]`

Hoặc edit trực tiếp trong `docs/generate_report.py` ở hàm `add_cover()`, rồi rerun script.

### 2. Replace mockup screenshots bằng ảnh thật

8 ảnh có tag `[MOCKUP]` ở trên cần được chụp lại từ ứng dụng thật để nộp:

#### Cách chụp TUI screenshots

**Yêu cầu**: Windows Terminal / PowerShell / cmd với theme tối, font Cascadia Code (mặc định) hoặc Consolas.

**Bước chụp:**
1. **Mở Windows Terminal**, đặt kích thước cửa sổ 100×30 hoặc 120×35 (đủ rộng cho dashboard).
2. **Build app** (nếu chưa build): `cmake --build build`
3. **Reseed data** sạch:
   ```powershell
   Remove-Item data\users.dat,data\transactions.dat,data\lfm_*.dat,data\transactions.log -ErrorAction SilentlyContinue
   .\build\seed_data.exe
   ```

**Chụp dashboard máy chủ** (`srv-dashboard-session.png`, `srv-dashboard-customers.png`):
```powershell
cd cli
npm install      # nếu chưa
npm run server
# Trong dashboard:
#   - Nhập "1234" + Enter để mở ca
#   - Để dashboard chạy 30s, có vài log lên
#   - Nhấn Win+Shift+S để cắt vùng dashboard → save với tên srv-dashboard-session.png
#   - Nhấn Tab để chuyển sang Customers panel
#   - Dùng ↑↓ chọn 1 khách (vd "Anh Nam"), Enter để xem chi tiết
#   - Nhấn Win+Shift+S → save với tên srv-dashboard-customers.png
```

**Chụp client UI** (6 ảnh `cli-*.png`):
```powershell
# Mở terminal thứ 2, để dashboard server vẫn chạy ở terminal 1
cd cli
node --import tsx/esm src/ClientApp.jsx 127.0.0.1 5
# Lần lượt chụp:
#   - Khi đang ở WaitingScreen (trước khi mở ca trên server)            → cli-waiting.png
#   - Sau khi mở ca, đang gõ dở SĐT (vd 5/10 chữ số)                   → cli-phone.png
#   - Login với SĐT chưa có (vd 0900000077), đang nhập tên              → cli-name.png
#   - Vào ORDERING, đã thêm 1-2 món, hiện đầy menu + suggest + cart    → cli-ordering.png
#   - Sau khi gõ "00", màn Invoice                                     → cli-invoice.png
#   - Sau khi gõ "Y" xác nhận, màn DailySummary "Cảm ơn quý khách!"    → cli-thanks.png
```

Tất cả ảnh save vào `docs\figures\` (overwrite mockup).

**Sau khi có ảnh thật**, rerun:
```powershell
.venv-report\Scripts\python docs\generate_report.py
```

→ `report.docx` mới sẽ embed ảnh thật.

### 3. Sửa nội dung text

- **Edit nhanh trong Word**: mở `report.docx`, sửa, save.
- **Edit tận gốc** (regenerate-able): sửa các string trong `docs/generate_report.py`, rerun.

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `PermissionError: [Errno 13] Permission denied: 'report.docx'` | Word đang mở file | Đóng Word; script tự động save sang `report-new.docx` nếu vẫn lock |
| Mermaid HTTP error trong `render_diagrams.py` | Không có Internet hoặc Kroki down | Mở https://mermaid.live, paste nội dung từ `diagrams/*.mmd`, export PNG manual vào `figures/` |
| Font Times New Roman bị lỗi (không có dấu) | Hệ thống thiếu Times New Roman | Cài Microsoft Office hoặc tải Times từ MS Core Fonts |
| Tiếng Việt bị "??" trong code excerpt | File `.cpp` lưu sai encoding | Đảm bảo các file `.cpp` lưu UTF-8 (no BOM); script đọc với `encoding="utf-8"` |
| Chữ cover page tràn ra ngoài lề | Tên đề tài quá dài | Sửa trong `add_cover()`, dùng size 16 thay 18 |
| `lfm-train.cpp` v.v. trong appendix dài quá 1 trang | Bình thường | Đó là code thật, đừng cắt; có thể tăng `font-size: Pt(10)` trong `add_code()` |

---

## Phụ thuộc Python

| Package | Version | Vai trò |
|---|---|---|
| python-docx | 1.2.0+ | Tạo `.docx` từ code |
| requests | 2.30+ | Gọi Kroki API |
| Pillow (PIL) | 10+ | Render text → PNG cho mockups |

Cài tất cả:
```powershell
.venv-report\Scripts\pip install python-docx requests Pillow
```

---

## Liên hệ

Source code: https://github.com/[your-handle]/pbl1_recommendation_system_using_lantent_factor (nếu public).

Báo cáo này được tự động sinh — mọi thay đổi nội dung gốc nên thực hiện ở `docs/generate_report.py` để tái tạo được.
